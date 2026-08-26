/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/hy_v3/hy_v3_weight_loader.h"

#include "model_loader/model_loader_utils.h"
#include "models/hy_v3/hy_v3_config.h"

namespace ksana_llm {

HyV3WeightLoader::HyV3WeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                                   std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_shared<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;
  std::shared_ptr<HyV3Config> hy_v3_config = std::dynamic_pointer_cast<HyV3Config>(model_config);
  if (hy_v3_config && hy_v3_config->is_quant) {
    weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_, hy_v3_config->quant_config);
  } else {
    weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
  }
}

bool HyV3WeightLoader::IsFp8() const {
  auto hy_v3_config = std::dynamic_pointer_cast<HyV3Config>(model_config_);
  return hy_v3_config && hy_v3_config->is_quant &&
         (hy_v3_config->quant_config.method == QUANT_FP8_E4M3 ||
          hy_v3_config->quant_config.method == QUANT_BLOCK_FP8_E4M3);
}

Status HyV3WeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  std::shared_ptr<HyV3Config> hy_v3_config = std::dynamic_pointer_cast<HyV3Config>(model_config_);
  const int num_layer = static_cast<int>(hy_v3_config->num_layer);

  // Skip MTP-shipped tensors that are not referenced by any layer we build:
  //   - shared_head is unused (main lm_head is reused; matches vLLM hy_v3_mtp.py:314 which
  //     remaps main lm_head.weight -> shared_head.head.weight).
  // NOTE: final_layernorm IS loaded and used as nextn_shared_head_norm_ (see HyV3Model), since
  // hy_v3 applies a per-MTP-layer final_layernorm before its shared_head (== main lm_head).
  std::vector<std::string> skip_list = {"shared_head."};
  std::vector<std::string> master_only_list = {"model.embed_tokens.weight"};
  std::vector<std::string> tail_only_list = {"lm_head.weight"};

  const int lower_layer_idx = pipeline_config_.lower_layer_idx;
  const int upper_layer_idx = pipeline_config_.upper_layer_idx;
  const int lower_nextn_layer_idx = pipeline_config_.lower_nextn_layer_idx;
  const int upper_nextn_layer_idx = pipeline_config_.upper_nextn_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, MatchMode::PartialMatch)) {
      weight_names.erase(it);
      continue;
    }

    const int layer_idx = GetLayerIdxFromName(*it);

    // MTP layers live at [num_layer, num_layer + num_nextn_predict_layers). Keep them only when
    // MTP is active (mtp_step_num > 0, which sets lower/upper_nextn_layer_idx); otherwise drop.
    if (layer_idx >= num_layer) {
      if (lower_nextn_layer_idx < 0 || upper_nextn_layer_idx < 0 || layer_idx < lower_nextn_layer_idx ||
          layer_idx > upper_nextn_layer_idx) {
        weight_names.erase(it);
        continue;
      }
    }

    // Skip layers outside this worker's pipeline range in distributed mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0 && layer_idx >= 0 &&
        (layer_idx < lower_layer_idx || layer_idx > upper_layer_idx) &&
        (layer_idx < lower_nextn_layer_idx || layer_idx > upper_nextn_layer_idx)) {
      weight_names.erase(it);
      continue;
    }

    if (!context_->global->IsStandalone()) {
      if (context_->global->IsDistributedWorker() && CheckWeightNameMatched(*it, master_only_list, false)) {
        weight_names.erase(it);
        continue;
      }
      if (!context_->global->IsDistributedLastWorker() && CheckWeightNameMatched(*it, tail_only_list, false)) {
        weight_names.erase(it);
        continue;
      }
    }
    ++it;
  }
  return Status();
}

Status HyV3WeightLoader::process_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                              const std::string& weight_prefix_name, int dev_rank) {
  std::shared_ptr<HyV3Config> hy_v3_config = std::dynamic_pointer_cast<HyV3Config>(model_config_);
  const size_t num_experts = hy_v3_config->moe_config.num_experts;

  // FP8 static per-tensor: hand off to the shared method (identical to qwen3_moe / glm4_moe).
  if (IsFp8()) {
    return weight_method_->GetFp8E4m3Method()->process_experts_proj(device_model_weights, weight_prefix_name, dev_rank,
                                                                    static_cast<int>(num_experts));
  }

  // BF16 path: merge per-expert gate/up into a batched up_gate_proj tensor.
  {
    const std::string up_gate_proj_name = fmt::format("{}up_gate_proj.weight", weight_prefix_name);
    std::vector<std::string> concat_names;
    // use_vllm_moe=true -> [gate, up] layout (vLLM convention). Storage key stays "up_gate_proj".
    const bool use_vllm_moe = hy_v3_config->moe_config.use_vllm_moe;
    for (size_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
      const char* first = use_vllm_moe ? "gate_proj" : "up_proj";
      const char* second = use_vllm_moe ? "up_proj" : "gate_proj";
      concat_names.push_back(fmt::format("{}{}.{}.weight", weight_prefix_name, expert_idx, first));
      concat_names.push_back(fmt::format("{}{}.{}.weight", weight_prefix_name, expert_idx, second));
    }
    common_weight_loader_->AutoMergeWeight(concat_names, up_gate_proj_name, device_model_weights, dev_rank);
    const std::vector<size_t> old_shape = device_model_weights.at(up_gate_proj_name).shape;
    const std::vector<size_t> new_shape = {num_experts, 2 * hy_v3_config->moe_config.moe_inter_size / tp_,
                                           hy_v3_config->hidden_units};
    KLLM_CHECK((old_shape[0] == (new_shape[0] * new_shape[1])) && (old_shape[1] == new_shape[2]));
    device_model_weights.at(up_gate_proj_name).shape = new_shape;
  }

  // Merge per-expert down projections.
  {
    const std::string down_proj_name = fmt::format("{}down_proj.weight", weight_prefix_name);
    std::vector<std::string> concat_names;
    for (size_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
      concat_names.push_back(fmt::format("{}{}.down_proj.weight", weight_prefix_name, expert_idx));
    }
    common_weight_loader_->AutoMergeWeight(concat_names, down_proj_name, device_model_weights, dev_rank);
    const std::vector<size_t> old_shape = device_model_weights.at(down_proj_name).shape;
    const std::vector<size_t> new_shape = {num_experts, hy_v3_config->hidden_units,
                                           hy_v3_config->moe_config.moe_inter_size / tp_};
    KLLM_CHECK((old_shape[0] == (new_shape[0] * new_shape[1])) && (old_shape[1] == new_shape[2]));
    device_model_weights.at(down_proj_name).shape = new_shape;
  }

  return Status();
}

Status HyV3WeightLoader::PreProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights) {
#define REGISTER_COMMON(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetCommonMethod().get(), \
                                                        &CommonMethod::key##_##weight_name)
#define REGISTER_FP8E4M3(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetFp8E4m3Method().get(), \
                                                        &Fp8E4m3Method::key##_##weight_name)
#define REGISTER_CUSTOM(key, weight_name) \
  weight_method_->GetRegistry()[#key][#weight_name].Add(this, &HyV3WeightLoader::key##_##weight_name)

  if (IsFp8()) {
    REGISTER_FP8E4M3(load, attn_q_k_v_proj);
    REGISTER_FP8E4M3(load, attn_o_proj);
    REGISTER_FP8E4M3(load, mlp_gate_up_proj);
    REGISTER_FP8E4M3(load, mlp_down_proj);
    REGISTER_FP8E4M3(process, attn_qkv_proj);
    REGISTER_FP8E4M3(process, attn_o_proj);
    REGISTER_FP8E4M3(process, mlp_gate_up_proj);
    REGISTER_FP8E4M3(process, mlp_down_proj);
  } else {
    REGISTER_COMMON(load, attn_q_k_v_proj);
    REGISTER_COMMON(load, attn_o_proj);
    REGISTER_COMMON(load, mlp_gate_up_proj);
    REGISTER_COMMON(load, mlp_down_proj);
    REGISTER_COMMON(process, attn_qkv_proj);
    REGISTER_COMMON(process, attn_o_proj);
    REGISTER_COMMON(process, mlp_gate_up_proj);
    REGISTER_COMMON(process, mlp_down_proj);
  }

  REGISTER_COMMON(load, experts_gate);
  REGISTER_COMMON(load, norm);
  REGISTER_COMMON(load, embed_tokens);
  REGISTER_COMMON(load, lm_head);

  REGISTER_COMMON(process, experts_gate);
  REGISTER_COMMON(process, lm_head);
  REGISTER_CUSTOM(process, experts_proj);

#undef REGISTER_COMMON
#undef REGISTER_FP8E4M3
#undef REGISTER_CUSTOM
  return Status();
}

Status HyV3WeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                             int dev_rank,
                                             std::unordered_map<std::string, Tensor>& device_model_weights,
                                             std::unordered_map<std::string, Tensor>& left_host_weights) {
  auto registry = weight_method_->GetRegistry()["load"];

  for (auto& [host_weight_name, host_weight_tensor] : host_model_weights) {
    // 1. Attention q/k/v and o projections (no QKV bias for hy_v3).
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"},
                               MatchMode::PartialMatch)) {
      registry["attn_q_k_v_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.o_proj"}, MatchMode::PartialMatch)) {
      registry["attn_o_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 2. Pre-RoPE QK-Norm weights (renamed from q_norm/k_norm by the weight map).
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.query_layernorm", ".self_attn.key_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 3. Router gate + aux-loss-free expert bias.
    if (CheckWeightNameMatched(host_weight_name, {".mlp.gate."}, MatchMode::PartialMatch)) {
      if (host_weight_name.find("e_score_correction_bias") != std::string::npos) {
        device_model_weights[host_weight_name] = common_weight_loader_->MoveToDevice(host_weight_tensor, dev_rank);
        continue;
      }
      registry["experts_gate"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 4. Routed expert weights (TP-split along inter_size, same as Qwen3-MoE).
    if (CheckWeightNameMatched(host_weight_name, {".mlp.experts."}, MatchMode::PartialMatch)) {
      if (CheckWeightNameMatched(host_weight_name, {".gate_proj", ".up_proj"}, MatchMode::PartialMatch)) {
        registry["mlp_gate_up_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
        continue;
      }
      if (CheckWeightNameMatched(host_weight_name, {".down_proj"}, MatchMode::PartialMatch)) {
        registry["mlp_down_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
        continue;
      }
    }

    // 5. Shared expert + dense prefix MLP. Both use standard column/row TP split;
    // the fused gate_up tensor is produced in PostProcess.
    if (CheckWeightNameMatched(host_weight_name, {".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj",
                                                  ".mlp.gate_proj", ".mlp.up_proj"},
                               MatchMode::PartialMatch)) {
      registry["mlp_gate_up_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".mlp.shared_expert.down_proj", ".mlp.down_proj"},
                               MatchMode::PartialMatch)) {
      registry["mlp_down_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 6. Norm weights (input_layernorm, post_attention_layernorm, model.norm, MTP enorm/hnorm,
    //    and per-MTP-layer final_layernorm — the last is used as nextn_shared_head_norm_).
    if (CheckWeightNameMatched(host_weight_name,
                               {"model.norm", ".input_layernorm", ".post_attention_layernorm", ".enorm.", ".hnorm.",
                                ".final_layernorm."},
                               MatchMode::PartialMatch)) {
      registry["norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 7. MTP eh_proj — column-parallel Linear. Weight only (BF16); FP8 checkpoints ship this in
    // BF16 too (verified for Hy3-FP8). Split along axis=0, then transpose.
    if (host_weight_name.find(".eh_proj.weight") != std::string::npos) {
      Tensor dev_tensor;
      common_weight_loader_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, tp_, true);
      device_model_weights[host_weight_name] = dev_tensor;
      continue;
    }

    // 8. Embedding + LM head.
    if (CheckWeightNameMatched(host_weight_name, {"model.embed_tokens"}, MatchMode::PartialMatch)) {
      registry["embed_tokens"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {"lm_head"}, MatchMode::PartialMatch)) {
      registry["lm_head"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
  }
  return Status();
}

Status HyV3WeightLoader::PostProcessDecoderLayer(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                 int layer_idx, int dev_rank, int first_k_dense_replace) {
  auto registry = weight_method_->GetRegistry()["process"];

  // Attention: fuse Q/K/V then permute o_proj.
  const std::string q_proj_name = fmt::format("model.layers.{}.self_attn.q_proj.", layer_idx);
  registry["attn_qkv_proj"].Run<Status>(dev_weights_map, q_proj_name, dev_rank);
  const std::string o_proj_name = fmt::format("model.layers.{}.self_attn.o_proj.", layer_idx);
  registry["attn_o_proj"].Run<Status>(dev_weights_map, o_proj_name, dev_rank);

  if (layer_idx >= first_k_dense_replace) {
    // Router gate.
    const std::string gate_name = fmt::format("model.layers.{}.mlp.gate.", layer_idx);
    registry["experts_gate"].Run<Status>(dev_weights_map, gate_name, dev_rank);

    // Routed experts: batch-merge gate_up and down.
    const std::string experts_proj_name = fmt::format("model.layers.{}.mlp.experts.", layer_idx);
    registry["experts_proj"].Run<Status>(dev_weights_map, experts_proj_name, dev_rank);

    // Shared expert: fuse gate_up, permute down.
    const std::string shared_gate_proj_name = fmt::format("model.layers.{}.mlp.shared_expert.gate_proj.", layer_idx);
    registry["mlp_gate_up_proj"].Run<Status>(dev_weights_map, shared_gate_proj_name, dev_rank);
    const std::string shared_down_proj_name = fmt::format("model.layers.{}.mlp.shared_expert.down_proj.", layer_idx);
    registry["mlp_down_proj"].Run<Status>(dev_weights_map, shared_down_proj_name, dev_rank);
  } else {
    // Dense prefix layer.
    const std::string gate_proj_name = fmt::format("model.layers.{}.mlp.gate_proj.", layer_idx);
    registry["mlp_gate_up_proj"].Run<Status>(dev_weights_map, gate_proj_name, dev_rank);
    const std::string down_proj_name = fmt::format("model.layers.{}.mlp.down_proj.", layer_idx);
    registry["mlp_down_proj"].Run<Status>(dev_weights_map, down_proj_name, dev_rank);
  }
  return Status();
}

Status HyV3WeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                 int dev_rank) {
  auto registry = weight_method_->GetRegistry()["process"];
  std::shared_ptr<HyV3Config> hy_v3_config = std::dynamic_pointer_cast<HyV3Config>(model_config_);
  const int first_k_dense_replace = static_cast<int>(hy_v3_config->moe_config.first_k_dense_replace);

  // Main decoder layers.
  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; layer_idx++) {
    STATUS_CHECK_RETURN(PostProcessDecoderLayer(dev_weights_map, layer_idx, dev_rank, first_k_dense_replace));
  }

  // MTP nextn layers (same shape as decoder layers; num_layer .. num_layer+M-1).
  if (pipeline_config_.lower_nextn_layer_idx >= static_cast<int>(hy_v3_config->num_layer)) {
    for (int layer_idx = pipeline_config_.lower_nextn_layer_idx; layer_idx <= pipeline_config_.upper_nextn_layer_idx;
         ++layer_idx) {
      STATUS_CHECK_RETURN(PostProcessDecoderLayer(dev_weights_map, layer_idx, dev_rank, first_k_dense_replace));
    }
  }

  // LM head. Use a std::string (not a string literal): the registry's Run<> mechanism
  // any_casts arguments to reference_wrapper<const std::string>, and a const char[] literal
  // would produce reference_wrapper<const char[]> and throw bad_any_cast.
  const std::string lm_head_name = "lm_head.";
  registry["lm_head"].Run<Status>(dev_weights_map, lm_head_name, dev_rank);

  return Status();
}

}  // namespace ksana_llm
