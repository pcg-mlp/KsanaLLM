/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/glm4_moe/glm4_moe_weight_loader.h"

#include <algorithm>
#include <cctype>

#include "device/nvidia/cuda_utils.h"
#include "kernels/nvidia/basic_kernel_wrapper.h"
#include "model_loader/model_loader_utils.h"
#include "models/glm4_moe/glm4_moe_config.h"
#include "utils/memory_utils.h"
#include "utils/ret_code.h"

namespace ksana_llm {

namespace {
bool TryGetExpertIndex(const std::string& weight_name, int& expert_idx) {
  const std::string kMarker = ".mlp.experts.";
  size_t pos = weight_name.find(kMarker);
  if (pos == std::string::npos) {
    return false;
  }
  pos += kMarker.size();
  size_t end = weight_name.find('.', pos);
  if (end == std::string::npos || end == pos) {
    return false;
  }
  int value = 0;
  for (size_t i = pos; i < end; ++i) {
    unsigned char ch = static_cast<unsigned char>(weight_name[i]);
    if (!std::isdigit(ch)) {
      return false;
    }
    value = value * 10 + (ch - '0');
  }
  expert_idx = value;
  return true;
}
}  // namespace

Glm4MoeWeightLoader::Glm4MoeWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                         std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_shared<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  env->GetExpertParallelConfig(expert_parallel_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;
  enable_full_shared_expert_ = runtime_config_.enable_full_shared_expert;
  // Get quant config from model config
  std::shared_ptr<Glm4MoeConfig> glm4_moe_config = std::dynamic_pointer_cast<Glm4MoeConfig>(model_config);

  if (glm4_moe_config && glm4_moe_config->is_quant) {
    // Use WeightMethod with QuantConfig for FP8 quantization support
    weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_, glm4_moe_config->quant_config);
    // GLM4-MoE enforces TP=EP: routed expert weights are NOT split along inter_size.
    // Create expert_weight_method_ with TP=1 so that FP8/Common load methods don't TP-split expert weights.
    expert_weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, 1, glm4_moe_config->quant_config);
    // Create a separate weight method for shared expert with TP=1 when enable_full_shared_expert is true
    if (enable_full_shared_expert_) {
      shared_expert_weight_method_ =
          std::make_shared<WeightMethod>(common_weight_loader_, 1, glm4_moe_config->quant_config);
    }
  } else {
    weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
    // GLM4-MoE enforces TP=EP: routed expert weights are NOT split along inter_size.
    expert_weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, 1);
    // Create a separate weight method for shared expert with TP=1 when enable_full_shared_expert is true
    if (enable_full_shared_expert_) {
      shared_expert_weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, 1);
    }
  }
}

Status Glm4MoeWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  // Skip MTP dead weights: shared_head.head and shared_head.norm are not referenced by any layer
  // in KsanaLLM (CommonModel reuses main model's lm_head and norm). MTP embed_tokens is skipped
  // separately in the layer index section below.
  std::vector<std::string> skip_list = {"shared_head.head.", "shared_head.norm."};
  std::vector<std::string> master_only_list = {"model.embed_tokens.weight"};
  std::vector<std::string> tail_only_list = {"lm_head.weight"};

  int lower_layer_idx = pipeline_config_.lower_layer_idx;
  int upper_layer_idx = pipeline_config_.upper_layer_idx;
  int lower_nextn_layer_idx = pipeline_config_.lower_nextn_layer_idx;
  int upper_nextn_layer_idx = pipeline_config_.upper_nextn_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, MatchMode::PartialMatch)) {
      weight_names.erase(it);
      continue;
    }

    // Skip some layers in distributed mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      int layer_idx = GetLayerIdxFromName(*it);
      if (layer_idx >= 0 && ((layer_idx < lower_layer_idx || layer_idx > upper_layer_idx) &&
                             (layer_idx < lower_nextn_layer_idx || layer_idx > upper_nextn_layer_idx))) {
        weight_names.erase(it);
        continue;
      }
      // Skip nextn layers on distributed non-sampling nodes (matching DeepSeek V3 pattern).
      if (layer_idx >= 0 && !context_->global->IsSamplingNode() &&
          (layer_idx >= lower_nextn_layer_idx && layer_idx <= upper_nextn_layer_idx)) {
        weight_names.erase(it);
        continue;
      }
      // Skip MTP layer embed_tokens — KsanaLLM reuses main model's embed_tokens for NextN.
      // Guard with layer_idx >= 0: main model's embed_tokens has layer_idx == -1 and must NOT be skipped.
      if (layer_idx >= 0 && layer_idx >= lower_nextn_layer_idx && layer_idx <= upper_nextn_layer_idx &&
          it->find(".embed_tokens.") != std::string::npos) {
        weight_names.erase(it);
        continue;
      }
    }

    if (!context_->global->IsStandalone()) {
      // Skip embedding unless master node or MTP-enabled sampling node.
      // MTP layers need embed_tokens for embedding lookup (same as DeepSeek V3).
      if (context_->global->IsDistributedWorker() &&
          !(context_->global->IsSamplingNode() && runtime_config_.mtp_step_num > 0)) {
        if (CheckWeightNameMatched(*it, master_only_list, false)) {
          weight_names.erase(it);
          continue;
        }
      }

      // Skip lm_head if not tail mode.
      if (!context_->global->IsDistributedLastWorker()) {
        if (CheckWeightNameMatched(*it, tail_only_list, false)) {
          weight_names.erase(it);
          continue;
        }
      }
    }
    ++it;
  }
  return Status();
}

Status Glm4MoeWeightLoader::process_attn_qkv_bias(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                  const std::string& weight_prefix_name, int dev_rank) {
  std::shared_ptr<Glm4MoeConfig> glm4_moe_config = std::dynamic_pointer_cast<Glm4MoeConfig>(model_config_);

  const std::string q_bias_name = weight_prefix_name + "bias";
  const std::string k_bias_name = WeightNameReplace(q_bias_name, "q_proj", "k_proj");
  const std::string v_bias_name = WeightNameReplace(q_bias_name, "q_proj", "v_proj");
  const std::string qkv_bias_name = WeightNameReplace(q_bias_name, "q_proj", "query_key_value");

  common_weight_loader_->AutoMergeWeight({q_bias_name, k_bias_name, v_bias_name}, qkv_bias_name, device_model_weights,
                                         dev_rank);

  // Check if qkv_bias exists before accessing
  if (device_model_weights.find(qkv_bias_name) == device_model_weights.end()) {
    KLLM_LOG_INFO << fmt::format("QKV bias {} not found, skipping shape adjustment", qkv_bias_name);
    return Status();
  }

  const std::vector<size_t> old_shape = device_model_weights.at(qkv_bias_name).shape;
  std::vector<size_t> new_shape = {1, glm4_moe_config->head_num / glm4_moe_config->num_key_value_heads + 2, 0};
  new_shape[2] = old_shape[0] / new_shape[1];
  device_model_weights.at(qkv_bias_name).shape = new_shape;
  return Status();
}

Status Glm4MoeWeightLoader::PreProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights) {
  std::shared_ptr<Glm4MoeConfig> glm4_moe_config = std::dynamic_pointer_cast<Glm4MoeConfig>(model_config_);

  // Determine quantization mode
  bool use_fp8 = glm4_moe_config && glm4_moe_config->is_quant &&
                 (glm4_moe_config->quant_config.method == QUANT_FP8_E4M3 ||
                  glm4_moe_config->quant_config.method == QUANT_BLOCK_FP8_E4M3);
  bool use_w4a8 = glm4_moe_config && glm4_moe_config->is_quant && glm4_moe_config->quant_config.enable_moe_int4;

  KLLM_LOG_DEBUG << fmt::format("PreProcessModelWeights: use_fp8={}, use_w4a8={}, is_quant={}, quant_method={}",
                               use_fp8, use_w4a8, glm4_moe_config ? glm4_moe_config->is_quant : false,
                               glm4_moe_config ? static_cast<int>(glm4_moe_config->quant_config.method) : -1);

#define REGISTER_COMMON(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetCommonMethod().get(), \
                                                        &CommonMethod::key##_##weight_name)
#define REGISTER_FP8(key, weight_name)                                                            \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetFp8E4m3Method().get(), \
                                                        &Fp8E4m3Method::key##_##weight_name)
#define REGISTER_RTN(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetRtnMethod().get(), \
                                                        &RtnMethod::key##_##weight_name)
#define REGISTER_CUSTOM(key, weight_name) \
  weight_method_->GetRegistry()[#key][#weight_name].Add(this, &Glm4MoeWeightLoader::key##_##weight_name)

  // Use FP8 methods for both per-tensor and block-wise FP8 quantization
  // The weight_map handles _scale_inv -> weight_scale naming conversion
  if (use_fp8 && weight_method_->GetFp8E4m3Method()) {
    KLLM_LOG_DEBUG << "Using FP8 weight processing methods for attention and MLP";
    REGISTER_FP8(load, attn_q_k_v_proj);
    REGISTER_FP8(load, attn_o_proj);
    REGISTER_FP8(load, mlp_gate_up_proj);
    REGISTER_FP8(load, mlp_down_proj);

    REGISTER_FP8(process, attn_qkv_proj);
    REGISTER_FP8(process, attn_o_proj);
    REGISTER_FP8(process, mlp_gate_up_proj);
    REGISTER_FP8(process, mlp_down_proj);
  } else {
    // Use common methods for non-quantized weights
    KLLM_LOG_DEBUG << "Using common weight processing methods";
    REGISTER_COMMON(load, attn_q_k_v_proj);
    REGISTER_COMMON(load, attn_o_proj);
    REGISTER_COMMON(load, mlp_gate_up_proj);
    REGISTER_COMMON(load, mlp_down_proj);

    REGISTER_COMMON(process, attn_qkv_proj);
    REGISTER_COMMON(process, attn_o_proj);
    REGISTER_COMMON(process, mlp_gate_up_proj);
    REGISTER_COMMON(process, mlp_down_proj);
  }

  // For RTN MoE experts, register RTN methods
  if (use_w4a8 && weight_method_->GetRtnMethod()) {
    KLLM_LOG_DEBUG << "Using RTN weight processing methods for MoE experts";
    REGISTER_RTN(load, experts_proj);
    REGISTER_RTN(process, experts_proj);
  }

  // Common methods that don't need special quantization handling
  REGISTER_COMMON(load, experts_gate);
  REGISTER_COMMON(load, norm);
  REGISTER_COMMON(load, embed_tokens);
  REGISTER_COMMON(load, lm_head);

  REGISTER_COMMON(process, experts_gate);
  REGISTER_COMMON(process, lm_head);

  // Register GLM4-MoE specific methods
  REGISTER_CUSTOM(process, attn_qkv_bias);

  // Register methods for expert_weight_method_ (TP=1) for routed experts.
  // GLM4-MoE enforces TP=EP: expert weights are NOT split along inter_size.
  if (expert_weight_method_) {
    if (use_fp8 && expert_weight_method_->GetFp8E4m3Method()) {
      expert_weight_method_->GetRegistry()["load"]["mlp_gate_up_proj"].Add(
          expert_weight_method_->GetFp8E4m3Method().get(), &Fp8E4m3Method::load_mlp_gate_up_proj);
      expert_weight_method_->GetRegistry()["load"]["mlp_down_proj"].Add(expert_weight_method_->GetFp8E4m3Method().get(),
                                                                        &Fp8E4m3Method::load_mlp_down_proj);
    } else {
      expert_weight_method_->GetRegistry()["load"]["mlp_gate_up_proj"].Add(
          expert_weight_method_->GetCommonMethod().get(), &CommonMethod::load_mlp_gate_up_proj);
      expert_weight_method_->GetRegistry()["load"]["mlp_down_proj"].Add(expert_weight_method_->GetCommonMethod().get(),
                                                                        &CommonMethod::load_mlp_down_proj);
    }
  }

  // Register methods for shared_expert_weight_method_ (TP=1) when enable_full_shared_expert is true
  // This is needed because shared_expert uses a separate WeightMethod with TP=1
  if (enable_full_shared_expert_ && shared_expert_weight_method_) {
    if (use_fp8 && shared_expert_weight_method_->GetFp8E4m3Method()) {
      shared_expert_weight_method_->GetRegistry()["load"]["mlp_gate_up_proj"].Add(
          shared_expert_weight_method_->GetFp8E4m3Method().get(), &Fp8E4m3Method::load_mlp_gate_up_proj);
      shared_expert_weight_method_->GetRegistry()["load"]["mlp_down_proj"].Add(
          shared_expert_weight_method_->GetFp8E4m3Method().get(), &Fp8E4m3Method::load_mlp_down_proj);
      shared_expert_weight_method_->GetRegistry()["process"]["mlp_gate_up_proj"].Add(
          shared_expert_weight_method_->GetFp8E4m3Method().get(), &Fp8E4m3Method::process_mlp_gate_up_proj);
      shared_expert_weight_method_->GetRegistry()["process"]["mlp_down_proj"].Add(
          shared_expert_weight_method_->GetFp8E4m3Method().get(), &Fp8E4m3Method::process_mlp_down_proj);
    } else {
      shared_expert_weight_method_->GetRegistry()["load"]["mlp_gate_up_proj"].Add(
          shared_expert_weight_method_->GetCommonMethod().get(), &CommonMethod::load_mlp_gate_up_proj);
      shared_expert_weight_method_->GetRegistry()["load"]["mlp_down_proj"].Add(
          shared_expert_weight_method_->GetCommonMethod().get(), &CommonMethod::load_mlp_down_proj);
      shared_expert_weight_method_->GetRegistry()["process"]["mlp_gate_up_proj"].Add(
          shared_expert_weight_method_->GetCommonMethod().get(), &CommonMethod::process_mlp_gate_up_proj);
      shared_expert_weight_method_->GetRegistry()["process"]["mlp_down_proj"].Add(
          shared_expert_weight_method_->GetCommonMethod().get(), &CommonMethod::process_mlp_down_proj);
    }
  }

#undef REGISTER_COMMON
#undef REGISTER_FP8
#undef REGISTER_RTN
#undef REGISTER_CUSTOM
  return Status();
}

Status Glm4MoeWeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                                int dev_rank,
                                                std::unordered_map<std::string, Tensor>& device_model_weights,
                                                std::unordered_map<std::string, Tensor>& left_host_weights) {
  auto registry = weight_method_->GetRegistry()["load"];
  std::shared_ptr<Glm4MoeConfig> glm4_moe_config = std::dynamic_pointer_cast<Glm4MoeConfig>(model_config_);
  int first_k_dense_replace = glm4_moe_config->moe_config.first_k_dense_replace;

  for (auto& [host_weight_name, host_weight_tensor] : host_model_weights) {
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}", dev_rank, host_weight_name,
                                  Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));

    // 1. Attention weights (QKV and O projections)
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"},
                               MatchMode::PartialMatch)) {
      registry["attn_q_k_v_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.o_proj"}, MatchMode::PartialMatch)) {
      registry["attn_o_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 2. QK norm weights (GLM4-MoE specific) - note: after weight map transformation, names become
    // query_layernorm/key_layernorm
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.query_layernorm", ".self_attn.key_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 3. MoE gate weights (router) - only for MoE layers
    if (CheckWeightNameMatched(host_weight_name, {".mlp.gate."}, MatchMode::PartialMatch)) {
      // Load e_score_correction_bias if present (used by MoE routing)
      if (host_weight_name.find("e_score_correction_bias") != std::string::npos) {
        device_model_weights[host_weight_name] = common_weight_loader_->MoveToDevice(host_weight_tensor, dev_rank);
        continue;
      }
      registry["experts_gate"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 4. Routed expert weights (MoE layers)
    if (CheckWeightNameMatched(host_weight_name, {".mlp.experts."}, MatchMode::PartialMatch)) {
      int expert_idx = -1;
      if (TryGetExpertIndex(host_weight_name, expert_idx) &&
          !IsLocalExpert(dev_rank, glm4_moe_config->moe_config.num_experts, expert_idx)) {
        continue;
      }
      // For W4A8 quantized experts, use experts_proj method
      // GLM4-MoE W4A8 format uses: .weight (int8), .weight_scale (fp32), .input_scale (bf16)
      // (DeepSeek uses .qweight/.scales but has its own separate code path)
      // IMPORTANT: Use IsWeightMatchW4A8() to respect ignored_layers from quantize_config.json.
      // For example, MTP layer (layer 92) experts are excluded from W4A8 and use FP8 instead.
      bool is_w4a8_expert = glm4_moe_config->IsWeightMatchW4A8(host_weight_name) &&
                            (host_weight_name.find(".weight") != std::string::npos ||
                             host_weight_name.find(".input_scale") != std::string::npos);

      if (is_w4a8_expert && registry.find("experts_proj") != registry.end()) {
        registry["experts_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
        continue;
      }

      // GLM4-MoE enforces TP=EP: use expert_weight_method_ (TP=1) so expert weights
      // are NOT split along inter_size dimension during loading.
      if (CheckWeightNameMatched(host_weight_name, {".gate_proj", ".up_proj"}, MatchMode::PartialMatch)) {
        auto& expert_registry = expert_weight_method_->GetRegistry();
        expert_registry["load"]["mlp_gate_up_proj"].Run<Status>(device_model_weights, host_weight_name,
                                                                host_weight_tensor, dev_rank);
        continue;
      }
      if (CheckWeightNameMatched(host_weight_name, {".down_proj"}, MatchMode::PartialMatch)) {
        auto& expert_registry = expert_weight_method_->GetRegistry();
        expert_registry["load"]["mlp_down_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor,
                                                             dev_rank);
        continue;
      }
    }

    // 5. Shared expert weights (MoE layers) - note: "shared_expert" (singular, after weight map transformation)
    // Use shared_expert_weight_method_ (TP=1) when enable_full_shared_expert is true
    if (CheckWeightNameMatched(host_weight_name, {".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj"},
                               MatchMode::PartialMatch)) {
      if (enable_full_shared_expert_ && shared_expert_weight_method_) {
        auto& shared_registry = shared_expert_weight_method_->GetRegistry();
        shared_registry["load"]["mlp_gate_up_proj"].Run<Status>(device_model_weights, host_weight_name,
                                                                host_weight_tensor, dev_rank);
      } else {
        registry["mlp_gate_up_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      }
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".mlp.shared_expert.down_proj"}, MatchMode::PartialMatch)) {
      if (enable_full_shared_expert_ && shared_expert_weight_method_) {
        auto& shared_registry = shared_expert_weight_method_->GetRegistry();
        shared_registry["load"]["mlp_down_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor,
                                                             dev_rank);
      } else {
        registry["mlp_down_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      }
      continue;
    }

    // 6. Dense layer MLP weights (layer_idx < first_k_dense_replace)
    // These use .mlp.gate_proj, .mlp.up_proj, .mlp.down_proj (without shared_expert prefix)
    if (CheckWeightNameMatched(host_weight_name, {".mlp.gate_proj", ".mlp.up_proj"}, MatchMode::PartialMatch)) {
      // Make sure it's not from experts or shared_expert
      if (host_weight_name.find(".mlp.experts.") == std::string::npos &&
          host_weight_name.find(".mlp.shared_expert.") == std::string::npos) {
        registry["mlp_gate_up_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
        continue;
      }
    }
    if (CheckWeightNameMatched(host_weight_name, {".mlp.down_proj"}, MatchMode::PartialMatch)) {
      // Make sure it's not from experts or shared_expert
      if (host_weight_name.find(".mlp.experts.") == std::string::npos &&
          host_weight_name.find(".mlp.shared_expert.") == std::string::npos) {
        registry["mlp_down_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
        continue;
      }
    }

    // 7. Norm weights (input_layernorm, post_attention_layernorm, final norm, MTP enorm/hnorm)
    if (CheckWeightNameMatched(host_weight_name,
                               {"model.norm", ".input_layernorm", ".post_attention_layernorm", ".enorm.", ".hnorm."},
                               MatchMode::PartialMatch)) {
      registry["norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 8. MTP eh_proj.weight — split along axis=0 (ColumnParallelLinear), then transpose
    // Shape: [hidden_size, 2*hidden_size] in safetensor (nn.Linear stores [out, in])
    // After SplitOptTrans: split dim0 by TP, then permute(1,0)
    if (host_weight_name.find(".eh_proj.weight") != std::string::npos &&
        host_weight_name.find("weight_scale") == std::string::npos &&
        host_weight_name.find("input_scale") == std::string::npos) {
      Tensor dev_tensor;
      common_weight_loader_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, tp_, true);
      device_model_weights[host_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("Loaded eh_proj weight: {}, shape=[{}, {}]", host_weight_name, dev_tensor.shape[0],
                                    dev_tensor.shape[1]);
      continue;
    }

    // 9. MTP eh_proj scales (FP8) — directly load to device
    if (host_weight_name.find(".eh_proj.weight_scale") != std::string::npos ||
        host_weight_name.find(".eh_proj.input_scale") != std::string::npos) {
      device_model_weights[host_weight_name] = common_weight_loader_->MoveToDevice(host_weight_tensor, dev_rank);
      continue;
    }

    // 10. Main model embedding weights
    if (CheckWeightNameMatched(host_weight_name, {"model.embed_tokens"}, MatchMode::PartialMatch)) {
      registry["embed_tokens"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 11. LM head weights
    if (CheckWeightNameMatched(host_weight_name, {"lm_head"}, MatchMode::PartialMatch)) {
      registry["lm_head"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
  }
  return Status();
}

void Glm4MoeWeightLoader::GetLocalExpertRange(int dev_rank, size_t num_experts, size_t& expert_start,
                                              size_t& expert_end) const {
  size_t expert_para_size = runtime_config_.parallel_basic_config.expert_parallel_size;
  size_t expert_world_size = runtime_config_.parallel_basic_config.expert_world_size;
  size_t global_expert_para_size = expert_para_size * expert_world_size;
  if (global_expert_para_size <= 1) {
    expert_start = 0;
    expert_end = num_experts;
    return;
  }
  size_t num_experts_per_rank = DivRoundUp(num_experts, global_expert_para_size);
  size_t expert_node_rank = expert_parallel_config_.expert_node_rank;
  size_t local_expert_rank = expert_para_size > 0 ? (dev_rank % expert_para_size) : 0;
  size_t rank_expert_offset = expert_node_rank * expert_para_size * num_experts_per_rank;
  size_t expert_offset = local_expert_rank * num_experts_per_rank;
  expert_start = rank_expert_offset + expert_offset;
  expert_end = std::min(num_experts, expert_start + num_experts_per_rank);
}

bool Glm4MoeWeightLoader::IsLocalExpert(int dev_rank, size_t num_experts, int expert_idx) const {
  size_t expert_start = 0;
  size_t expert_end = 0;
  GetLocalExpertRange(dev_rank, num_experts, expert_start, expert_end);
  if (expert_idx < 0) {
    return false;
  }
  return static_cast<size_t>(expert_idx) >= expert_start && static_cast<size_t>(expert_idx) < expert_end;
}

Status Glm4MoeWeightLoader::PostProcessDecoderLayer(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                    int layer_idx, int dev_rank, int first_k_dense_replace) {
  auto registry = weight_method_->GetRegistry()["process"];
  std::shared_ptr<Glm4MoeConfig> glm4_moe_config = std::dynamic_pointer_cast<Glm4MoeConfig>(model_config_);

  // 1. Attention: fuse QKV
  const std::string q_proj_name = fmt::format("model.layers.{}.self_attn.q_proj.", layer_idx);
  const std::string q_weight = q_proj_name + "weight";
  const std::string k_weight = fmt::format("model.layers.{}.self_attn.k_proj.weight", layer_idx);
  const std::string v_weight = fmt::format("model.layers.{}.self_attn.v_proj.weight", layer_idx);

  if (dev_weights_map.count(q_weight) && dev_weights_map.count(k_weight) && dev_weights_map.count(v_weight)) {
    Status status = registry["attn_qkv_proj"].Run<Status>(dev_weights_map, q_proj_name, dev_rank);
    if (!status.OK()) {
      KLLM_LOG_ERROR << fmt::format("Failed to process attn_qkv_proj for layer {}: {}", layer_idx, status.GetMessage());
      return status;
    }
  } else {
    KLLM_LOG_ERROR << fmt::format("Missing Q/K/V weights for layer {}", layer_idx);
    return Status(RetCode::RET_INVALID_ARGUMENT, fmt::format("Missing Q/K/V weights for layer {}", layer_idx));
  }

  // QKV bias
  if (glm4_moe_config->enable_add_qkv_bias) {
    Status status = registry["attn_qkv_bias"].Run<Status>(dev_weights_map, q_proj_name, dev_rank);
    if (!status.OK()) {
      KLLM_LOG_WARNING << fmt::format("Failed to process attn_qkv_bias for layer {}: {}", layer_idx,
                                      status.GetMessage());
    }
  }

  // 2. O_proj
  const std::string o_proj_name = fmt::format("model.layers.{}.self_attn.o_proj.", layer_idx);
  Status status = registry["attn_o_proj"].Run<Status>(dev_weights_map, o_proj_name, dev_rank);
  if (!status.OK()) {
    KLLM_LOG_ERROR << fmt::format("Failed to process attn_o_proj for layer {}: {}", layer_idx, status.GetMessage());
    return status;
  }

  // 3. MLP: MoE or Dense
  if (layer_idx >= first_k_dense_replace) {
    // Router gate
    const std::string gate_name = fmt::format("model.layers.{}.mlp.gate.", layer_idx);
    status = registry["experts_gate"].Run<Status>(dev_weights_map, gate_name, dev_rank);
    if (!status.OK()) {
      KLLM_LOG_ERROR << fmt::format("Failed to process experts_gate for layer {}: {}", layer_idx, status.GetMessage());
      return status;
    }

    // Routed experts: fuse gate_up and down
    const std::string experts_proj_name = fmt::format("model.layers.{}.mlp.experts.", layer_idx);
    status = registry["experts_proj"].Run<Status>(dev_weights_map, experts_proj_name, dev_rank);
    if (!status.OK()) {
      KLLM_LOG_ERROR << fmt::format("Failed to process experts_proj for layer {}: {}", layer_idx, status.GetMessage());
      return status;
    }

    // Shared expert: gate_up
    const std::string shared_expert_gate_proj_name =
        fmt::format("model.layers.{}.mlp.shared_expert.gate_proj.", layer_idx);
    if (enable_full_shared_expert_ && shared_expert_weight_method_) {
      auto& shared_registry = shared_expert_weight_method_->GetRegistry();
      status = shared_registry["process"]["mlp_gate_up_proj"].Run<Status>(dev_weights_map, shared_expert_gate_proj_name,
                                                                          dev_rank);
    } else {
      status = registry["mlp_gate_up_proj"].Run<Status>(dev_weights_map, shared_expert_gate_proj_name, dev_rank);
    }
    if (!status.OK()) {
      KLLM_LOG_ERROR << fmt::format("Failed to process shared_expert gate_up_proj for layer {}: {}", layer_idx,
                                    status.GetMessage());
      return status;
    }

    // Shared expert: down_proj
    const std::string shared_expert_down_proj_name =
        fmt::format("model.layers.{}.mlp.shared_expert.down_proj.", layer_idx);
    if (enable_full_shared_expert_ && shared_expert_weight_method_) {
      auto& shared_registry = shared_expert_weight_method_->GetRegistry();
      status = shared_registry["process"]["mlp_down_proj"].Run<Status>(dev_weights_map, shared_expert_down_proj_name,
                                                                       dev_rank);
    } else {
      status = registry["mlp_down_proj"].Run<Status>(dev_weights_map, shared_expert_down_proj_name, dev_rank);
    }
    if (!status.OK()) {
      KLLM_LOG_ERROR << fmt::format("Failed to process shared_expert down_proj for layer {}: {}", layer_idx,
                                    status.GetMessage());
      return status;
    }
  } else {
    // Dense layers: standard MLP
    const std::string gate_proj_name = fmt::format("model.layers.{}.mlp.gate_proj.", layer_idx);
    status = registry["mlp_gate_up_proj"].Run<Status>(dev_weights_map, gate_proj_name, dev_rank);
    if (!status.OK()) {
      KLLM_LOG_ERROR << fmt::format("Failed to process mlp_gate_up_proj for layer {}: {}", layer_idx,
                                    status.GetMessage());
      return status;
    }

    const std::string down_proj_name = fmt::format("model.layers.{}.mlp.down_proj.", layer_idx);
    status = registry["mlp_down_proj"].Run<Status>(dev_weights_map, down_proj_name, dev_rank);
    if (!status.OK()) {
      KLLM_LOG_ERROR << fmt::format("Failed to process mlp_down_proj for layer {}: {}", layer_idx, status.GetMessage());
      return status;
    }
  }

  return Status();
}

Status Glm4MoeWeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                    int dev_rank) {
  auto registry = weight_method_->GetRegistry()["process"];
  std::shared_ptr<Glm4MoeConfig> glm4_moe_config = std::dynamic_pointer_cast<Glm4MoeConfig>(model_config_);
  int first_k_dense_replace = glm4_moe_config->moe_config.first_k_dense_replace;

  // ===== Generic dtype conversion =====
  // Convert non-quantization-related floating-point tensors to inter_data_type
  // (the dtype used by kernel inputs/outputs, currently equals weight_data_type).
  // Using inter_data_type instead of weight_data_type ensures all weights match
  // the dtype that MatMulLayerFactory::AutoCreateLayer expects (input_type),
  // avoiding builder_map_ lookup failures if weight_data_type and inter_data_type
  // ever diverge in the future (e.g., mixed-precision inference).
  // Skip: quantized weights (INT8/FP8), weight_scale/input_scale (handled by FP8/W4A8 methods),
  //        and e_score_correction_bias (needs FP32).
  DataType target_dtype = runtime_config_.inter_data_type;
  for (auto& [weight_name, weight_tensor] : dev_weights_map) {
    // Skip non-floating-point tensors (INT8 quantized weights, etc.)
    if (!(weight_tensor.dtype == DataType::TYPE_FP32 || weight_tensor.dtype == DataType::TYPE_FP16 ||
          weight_tensor.dtype == DataType::TYPE_BF16)) {
      continue;
    }
    // Skip weight_scale and input_scale tensors — they are consumed by FP8/W4A8 methods
    // which expect specific dtypes (FP8 method expects FP32 scales, W4A8 method handles its own conversion)
    if (weight_name.find("weight_scale") != std::string::npos || weight_name.find("input_scale") != std::string::npos) {
      continue;
    }
    // e_score_correction_bias needs FP32 for better precision (matching DeepSeek V3)
    if (weight_name.find("gate.e_score_correction_bias") != std::string::npos) {
      if (weight_tensor.dtype != DataType::TYPE_FP32) {
        KLLM_LOG_DEBUG << fmt::format("Cast {} from {} to FP32", weight_name, weight_tensor.dtype);
        common_weight_loader_->CastDeviceTensorType(weight_tensor, DataType::TYPE_FP32, dev_rank);
      }
      continue;
    }
    // Convert all other floating-point tensors to target dtype
    if (weight_tensor.dtype != target_dtype) {
      KLLM_LOG_DEBUG << fmt::format("Cast {} from {} to {}", weight_name, weight_tensor.dtype, target_dtype);
      common_weight_loader_->CastDeviceTensorType(weight_tensor, target_dtype, dev_rank);
    }
  }

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; layer_idx++) {
    Status status = PostProcessDecoderLayer(dev_weights_map, layer_idx, dev_rank, first_k_dense_replace);
    if (!status.OK()) return status;
  }

  // ===== MTP (NextN) layer post-processing =====
  if (pipeline_config_.lower_nextn_layer_idx >= static_cast<int>(glm4_moe_config->num_layer)) {
    for (int layer_idx = pipeline_config_.lower_nextn_layer_idx; layer_idx <= pipeline_config_.upper_nextn_layer_idx;
         ++layer_idx) {
      KLLM_LOG_INFO << fmt::format("PostProcess MTP layer {}", layer_idx);

      // Decoder layer weights (attention + MoE) — same processing as main decoder layers
      Status status = PostProcessDecoderLayer(dev_weights_map, layer_idx, dev_rank, first_k_dense_replace);
      if (!status.OK()) return status;

      // Bind FP8 weight_scale/input_scale for eh_proj if present
      const std::string eh_proj_weight_name = fmt::format("model.layers.{}.eh_proj.weight", layer_idx);
      const std::string eh_proj_scale_name = fmt::format("model.layers.{}.eh_proj.weight_scale", layer_idx);
      if (dev_weights_map.count(eh_proj_weight_name) && dev_weights_map.count(eh_proj_scale_name)) {
        dev_weights_map[eh_proj_weight_name].weight_scales = &dev_weights_map[eh_proj_scale_name];
        KLLM_LOG_DEBUG << fmt::format("Bound eh_proj weight_scale for MTP layer {}", layer_idx);
      }
      const std::string eh_proj_input_scale_name = fmt::format("model.layers.{}.eh_proj.input_scale", layer_idx);
      if (dev_weights_map.count(eh_proj_weight_name) && dev_weights_map.count(eh_proj_input_scale_name)) {
        dev_weights_map[eh_proj_weight_name].input_scales = &dev_weights_map[eh_proj_input_scale_name];
        KLLM_LOG_DEBUG << fmt::format("Bound eh_proj input_scale for MTP layer {}", layer_idx);
      }
    }
    KLLM_LOG_INFO << fmt::format("PostProcess MTP layers [{}, {}] completed", pipeline_config_.lower_nextn_layer_idx,
                                 pipeline_config_.upper_nextn_layer_idx);
  }

  // 3. LM head weights
  const std::string lm_head_name = "lm_head.";
  Status status = registry["lm_head"].Run<Status>(dev_weights_map, lm_head_name, dev_rank);
  if (!status.OK()) {
    KLLM_LOG_ERROR << fmt::format("Failed to process lm_head: {}", status.GetMessage());
    return status;
  }

  KLLM_LOG_DEBUG << "PostProcessModelWeights completed successfully";
  return Status();
}

}  // namespace ksana_llm
