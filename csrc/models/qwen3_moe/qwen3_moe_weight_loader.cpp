/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/qwen3_moe/qwen3_moe_weight_loader.h"

#include <torch/torch.h>

#include <algorithm>

#include "model_loader/model_loader_utils.h"
#include "models/base/utils.h"
#include "models/qwen3_moe/qwen3_moe_config.h"

namespace ksana_llm {

Qwen3MoeWeightLoader::Qwen3MoeWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                           std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_shared<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;
  weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
  std::shared_ptr<Qwen3MoeConfig> qwen3_moe_config = std::dynamic_pointer_cast<Qwen3MoeConfig>(model_config);
  if (qwen3_moe_config && qwen3_moe_config->is_quant &&
      (qwen3_moe_config->quant_config.method == QUANT_FP8_E4M3 ||
       qwen3_moe_config->quant_config.method == QUANT_BLOCK_FP8_E4M3)) {
    weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_, qwen3_moe_config->quant_config);
  } else {
    weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
  }
}

Status Qwen3MoeWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  std::vector<std::string> skip_list = {};
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
    }

    if (!context_->global->IsStandalone()) {
      // Skip embedding on if not master mode.
      if (context_->global->IsDistributedWorker()) {
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

void Qwen3MoeWeightLoader::ComputeExpertPartition(int dev_rank, size_t num_experts, size_t& num_experts_per_rank,
                                                  size_t& expert_start, size_t& expert_end, size_t& moe_tp_size,
                                                  size_t& tp_rank) const {
  // Mirrors csrc/models/quant/quant_weight.cpp so the BF16 layout matches what the Kunlun EP kernels expect.
  const size_t expert_world_size = runtime_config_.parallel_basic_config.expert_world_size;
  const size_t expert_para_size = runtime_config_.parallel_basic_config.expert_parallel_size;
  const size_t global_expert_para_size = expert_world_size * expert_para_size;
  moe_tp_size = runtime_config_.parallel_basic_config.moe_tensor_para_size;

  ExpertParallelConfig expert_parallel_config;
  env_->GetExpertParallelConfig(expert_parallel_config);
  const size_t expert_node_rank = expert_parallel_config.expert_node_rank;

  num_experts_per_rank = (num_experts + global_expert_para_size - 1) / global_expert_para_size;
  const size_t rank_expert_offset = expert_node_rank * expert_para_size * num_experts_per_rank;
  const size_t expert_offset =
      (global_expert_para_size > 1) ? ((static_cast<size_t>(dev_rank) % expert_para_size) * num_experts_per_rank) : 0;
  expert_start = rank_expert_offset + expert_offset;
  expert_end = std::min(num_experts, expert_start + num_experts_per_rank);
  tp_rank = (moe_tp_size > 1) ? (static_cast<size_t>(dev_rank) / expert_para_size) : 0;
}

// Parse the global expert index out of a name like
// "model.layers.<L>.mlp.experts.<E>.gate_proj.weight". Returns -1 if not an expert weight.
static int ParseExpertIdx(const std::string& weight_name) {
  const std::string marker = ".mlp.experts.";
  size_t pos = weight_name.find(marker);
  if (pos == std::string::npos) return -1;
  pos += marker.size();
  size_t end = weight_name.find('.', pos);
  if (end == std::string::npos || end == pos) return -1;
  return std::stoi(weight_name.substr(pos, end - pos));
}

Status Qwen3MoeWeightLoader::load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                   const std::string& weight_name, const Tensor& weight_tensor,
                                                   int dev_rank) {
  if (!CheckWeightNameEndMatched(weight_name, {".weight"})) return Status();
  std::shared_ptr<Qwen3MoeConfig> qwen3_moe_config = std::dynamic_pointer_cast<Qwen3MoeConfig>(model_config_);
  const size_t num_experts = qwen3_moe_config->moe_config.num_experts;
  size_t num_experts_per_rank, expert_start, expert_end, moe_tp_size, tp_rank;
  ComputeExpertPartition(dev_rank, num_experts, num_experts_per_rank, expert_start, expert_end, moe_tp_size, tp_rank);

  const int expert_idx = ParseExpertIdx(weight_name);
  // Skip experts that do not belong to this rank's expert-parallel shard.
  if (expert_idx >= 0 &&
      (static_cast<size_t>(expert_idx) < expert_start || static_cast<size_t>(expert_idx) >= expert_end)) {
    return Status();
  }
  // Split the intermediate dim by the moe tensor-parallel group (RowPara), not the full tensor_parallel_size.
  device_model_weights[weight_name] = common_weight_loader_->TensorParallelSplit(
      weight_tensor, static_cast<int>(tp_rank), moe_tp_size, TensorParallelMode::RowPara);
  return Status();
}

Status Qwen3MoeWeightLoader::load_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                const std::string& weight_name, const Tensor& weight_tensor,
                                                int dev_rank) {
  if (!CheckWeightNameEndMatched(weight_name, {".weight"})) return Status();
  std::shared_ptr<Qwen3MoeConfig> qwen3_moe_config = std::dynamic_pointer_cast<Qwen3MoeConfig>(model_config_);
  const size_t num_experts = qwen3_moe_config->moe_config.num_experts;
  size_t num_experts_per_rank, expert_start, expert_end, moe_tp_size, tp_rank;
  ComputeExpertPartition(dev_rank, num_experts, num_experts_per_rank, expert_start, expert_end, moe_tp_size, tp_rank);

  const int expert_idx = ParseExpertIdx(weight_name);
  if (expert_idx >= 0 &&
      (static_cast<size_t>(expert_idx) < expert_start || static_cast<size_t>(expert_idx) >= expert_end)) {
    return Status();
  }
  device_model_weights[weight_name] = common_weight_loader_->TensorParallelSplit(
      weight_tensor, static_cast<int>(tp_rank), moe_tp_size, TensorParallelMode::ColPara);
  return Status();
}

Status Qwen3MoeWeightLoader::process_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                  const std::string& weight_prefix_name, int dev_rank) {
  std::shared_ptr<Qwen3MoeConfig> qwen3_moe_config = std::dynamic_pointer_cast<Qwen3MoeConfig>(model_config_);
  size_t num_experts = qwen3_moe_config->moe_config.num_experts;
  // call Fp8E4m3Method if quant method is fp8_e4m3
  if (qwen3_moe_config->quant_config.method == QUANT_FP8_E4M3) {
    return weight_method_->GetFp8E4m3Method()->process_experts_proj(device_model_weights, weight_prefix_name, dev_rank,
                                                                    num_experts);
  }

  // Expert-parallel partition for this rank. Under EP each rank only holds its local experts
  // [expert_start, expert_end) and splits the intermediate dim by moe_tensor_para_size.
  // With ep=1 this reduces to: all experts, split by tensor_parallel_size (== moe_tensor_para_size).
  size_t num_experts_per_rank, expert_start, expert_end, moe_tp_size, tp_rank;
  ComputeExpertPartition(dev_rank, num_experts, num_experts_per_rank, expert_start, expert_end, moe_tp_size, tp_rank);
  const size_t local_expert_num = expert_end - expert_start;

  // 合并gate_up
  {
    const std::string gate_up_proj_name = fmt::format("{}up_gate_proj.", weight_prefix_name);
    const std::vector<std::string> weight_suffixs = {"weight"};
    for (const std::string& weight_suffix : weight_suffixs) {
      std::vector<std::string> concat_names;
      // use_vllm_moe=true: layout is [gate, up] (vLLM convention); false: [up, gate].
      // Tensor key "up_gate_proj" is a fixed storage name, independent of actual layout.
      // Kunlun: qwen3_moe_config_parser sets use_vllm_moe=true under ENABLE_KUNLUN to get [gate|up]
      //         layout (matches xblas moe_fc_fusion + infer_ops::swiglu turn=true, aligned with xsglang w13).
      const bool use_vllm_moe = qwen3_moe_config->moe_config.use_vllm_moe;
      for (size_t expert_idx = expert_start; expert_idx < expert_end; expert_idx++) {
        const char* first = use_vllm_moe ? "gate_proj" : "up_proj";
        const char* second = use_vllm_moe ? "up_proj" : "gate_proj";
        concat_names.push_back(
            fmt::format("{}{}.{}.{}", weight_prefix_name, expert_idx, first, weight_suffix));
        concat_names.push_back(
            fmt::format("{}{}.{}.{}", weight_prefix_name, expert_idx, second, weight_suffix));
      }
      common_weight_loader_->AutoMergeWeight(concat_names, gate_up_proj_name + weight_suffix, device_model_weights,
                                             dev_rank);
      // shape检查与调整
      const std::vector<size_t> old_shape = device_model_weights.at(gate_up_proj_name + weight_suffix).shape;
      const std::vector<size_t> new_shape = {local_expert_num,
                                             2 * qwen3_moe_config->moe_config.moe_inter_size / moe_tp_size,
                                             qwen3_moe_config->hidden_units};
      KLLM_CHECK((old_shape[0] == (new_shape[0] * new_shape[1])) && (old_shape[1] == new_shape[2]));
      device_model_weights.at(gate_up_proj_name + weight_suffix).shape = new_shape;
    }
  }
  // 合并down
  {
    const std::string down_proj_name = fmt::format("{}down_proj.", weight_prefix_name);
    const std::vector<std::string> weight_suffixs = {"weight"};
    for (const std::string& weight_suffix : weight_suffixs) {
      std::vector<std::string> concat_names;
      for (size_t expert_idx = expert_start; expert_idx < expert_end; expert_idx++) {
        concat_names.push_back(fmt::format("{}{}.down_proj.{}", weight_prefix_name, expert_idx, weight_suffix));
      }
      common_weight_loader_->AutoMergeWeight(concat_names, down_proj_name + weight_suffix, device_model_weights,
                                             dev_rank);
      // shape检查与调整
      const std::vector<size_t> old_shape = device_model_weights.at(down_proj_name + weight_suffix).shape;
      const std::vector<size_t> new_shape = {local_expert_num, qwen3_moe_config->hidden_units,
                                             qwen3_moe_config->moe_config.moe_inter_size / moe_tp_size};
      KLLM_CHECK((old_shape[0] == (new_shape[0] * new_shape[1])) && (old_shape[1] == new_shape[2]));
      device_model_weights.at(down_proj_name + weight_suffix).shape = new_shape;
    }
  }
  // Block-wise FP8: concat per-expert block scales into batched scales matching the merged
  // up_gate_proj/down_proj weights and bind them. Activation is dynamic (no input_scale in
  // checkpoint), so input_scales is left null. No-op for non-block (BF16 / static-FP8) checkpoints.
  if (qwen3_moe_config->quant_config.is_fp8_blockwise) {
    const bool use_vllm_moe = qwen3_moe_config->moe_config.use_vllm_moe;
    const std::string first_proj = use_vllm_moe ? "gate_proj" : "up_proj";
    const std::string second_proj = use_vllm_moe ? "up_proj" : "gate_proj";

    torch::Tensor first_scale = common_weight_loader_->GetTorchTensorFromTensor(
        device_model_weights.at(fmt::format("{}0.{}.weight_scale", weight_prefix_name, first_proj)));
    const int64_t scale_rows = first_scale.size(0);
    const int64_t scale_cols = first_scale.size(1);
    std::vector<torch::Tensor> expert_up_gate_scales;
    for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
      torch::Tensor s0 = common_weight_loader_->GetTorchTensorFromTensor(device_model_weights.at(
          fmt::format("{}{}.{}.weight_scale", weight_prefix_name, expert_idx, first_proj)));
      torch::Tensor s1 = common_weight_loader_->GetTorchTensorFromTensor(device_model_weights.at(
          fmt::format("{}{}.{}.weight_scale", weight_prefix_name, expert_idx, second_proj)));
      expert_up_gate_scales.push_back(torch::cat({s0, s1}, 0));
    }
    torch::Tensor batched_up_gate_scale = torch::stack(expert_up_gate_scales, 0).contiguous();
    const std::string up_gate_scale_name = weight_prefix_name + "up_gate_proj.weight_scale";
    device_model_weights[up_gate_scale_name] = Tensor(
        MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
        {static_cast<size_t>(num_experts), static_cast<size_t>(2 * scale_rows), static_cast<size_t>(scale_cols)});
    common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[up_gate_scale_name])
        .copy_(batched_up_gate_scale);
    device_model_weights[weight_prefix_name + "up_gate_proj.weight"].weight_scales =
        &device_model_weights[up_gate_scale_name];

    std::vector<torch::Tensor> expert_down_scales;
    for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
      expert_down_scales.push_back(common_weight_loader_->GetTorchTensorFromTensor(device_model_weights.at(
          fmt::format("{}{}.down_proj.weight_scale", weight_prefix_name, expert_idx))));
    }
    torch::Tensor batched_down_scale = torch::stack(expert_down_scales, 0).contiguous();
    const std::string down_scale_name = weight_prefix_name + "down_proj.weight_scale";
    device_model_weights[down_scale_name] = Tensor(
        MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
        {static_cast<size_t>(num_experts), static_cast<size_t>(batched_down_scale.size(1)),
         static_cast<size_t>(batched_down_scale.size(2))});
    common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[down_scale_name]).copy_(batched_down_scale);
    device_model_weights[weight_prefix_name + "down_proj.weight"].weight_scales =
        &device_model_weights[down_scale_name];

    for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
      device_model_weights.erase(
          fmt::format("{}{}.{}.weight_scale", weight_prefix_name, expert_idx, first_proj));
      device_model_weights.erase(
          fmt::format("{}{}.{}.weight_scale", weight_prefix_name, expert_idx, second_proj));
      device_model_weights.erase(fmt::format("{}{}.down_proj.weight_scale", weight_prefix_name, expert_idx));
    }
  }

  return Status();
}

Status Qwen3MoeWeightLoader::PreProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights) {
  std::shared_ptr<Qwen3MoeConfig> qwen3_moe_config = std::dynamic_pointer_cast<Qwen3MoeConfig>(model_config_);

#define REGISTER_COMMON(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetCommonMethod().get(), \
                                                        &CommonMethod::key##_##weight_name)

#define REGISTER_FP8E4M3(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetFp8E4m3Method().get(), \
                                                        &Fp8E4m3Method::key##_##weight_name)

#define REGISTER_CUSTOM(key, weight_name) \
  weight_method_->GetRegistry()[#key][#weight_name].Add(this, &Qwen3MoeWeightLoader::key##_##weight_name)

  if (qwen3_moe_config->quant_config.method == QUANT_FP8_E4M3 ||
      qwen3_moe_config->quant_config.method == QUANT_BLOCK_FP8_E4M3) {
    REGISTER_FP8E4M3(load, attn_q_k_v_proj);
    REGISTER_FP8E4M3(load, attn_o_proj);
    REGISTER_FP8E4M3(load, mlp_gate_up_proj);
    REGISTER_FP8E4M3(load, mlp_down_proj);

    REGISTER_FP8E4M3(process, attn_qkv_proj);
    REGISTER_FP8E4M3(process, attn_o_proj);
  } else {
    REGISTER_COMMON(load, attn_q_k_v_proj);
    REGISTER_COMMON(load, attn_o_proj);
    // EP-aware expert loaders (split by moe_tensor_para_size, keep only local experts).
    REGISTER_CUSTOM(load, mlp_gate_up_proj);
    REGISTER_CUSTOM(load, mlp_down_proj);

    REGISTER_COMMON(process, attn_qkv_proj);
    REGISTER_COMMON(process, attn_o_proj);
  }

  REGISTER_COMMON(load, attn_norm);
  REGISTER_COMMON(load, norm);
  REGISTER_COMMON(load, embed_tokens);
  REGISTER_COMMON(load, lm_head);
  REGISTER_COMMON(load, experts_gate);

  REGISTER_COMMON(process, lm_head);
  REGISTER_CUSTOM(process, experts_proj);
  REGISTER_COMMON(process, experts_gate);

#undef REGISTER_COMMON
#undef REGISTER_CUSTOM
  return Status();
}

Status Qwen3MoeWeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                                 int dev_rank,
                                                 std::unordered_map<std::string, Tensor>& device_model_weights,
                                                 std::unordered_map<std::string, Tensor>& left_host_weights) {
  auto registry = weight_method_->GetRegistry()["load"];

  for (auto& [host_weight_name, host_weight_tensor] : host_model_weights) {
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}", dev_rank, host_weight_name,
                                  Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));

    // 1. attn weights
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"},
                               MatchMode::PartialMatch)) {
      registry["attn_q_k_v_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.o_proj"}, MatchMode::PartialMatch)) {
      registry["attn_o_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.key_layernorm", ".self_attn.query_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["attn_norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 2. experts weights
    if (CheckWeightNameMatched(host_weight_name, {".mlp.gate"}, MatchMode::PartialMatch)) {
      registry["experts_gate"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
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

    // 3. norm weights
    if (CheckWeightNameMatched(host_weight_name, {"model.norm", ".input_layernorm", ".post_attention_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 4. embedding weights && lm_head weights
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

Status Qwen3MoeWeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                     int dev_rank) {
  auto registry = weight_method_->GetRegistry()["process"];

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; layer_idx++) {
    // 1. attn weights
    const std::string q_proj_name = fmt::format("model.layers.{}.self_attn.q_proj.", layer_idx);
    registry["attn_qkv_proj"].Run<Status>(dev_weights_map, q_proj_name, dev_rank);

    const std::string o_proj_name = fmt::format("model.layers.{}.self_attn.o_proj.", layer_idx);
    registry["attn_o_proj"].Run<Status>(dev_weights_map, o_proj_name, dev_rank);

    // 2. experts weights
    const std::string gate_name = fmt::format("model.layers.{}.mlp.gate.", layer_idx);
    registry["experts_gate"].Run<Status>(dev_weights_map, gate_name, dev_rank);

    const std::string experts_proj_name = fmt::format("model.layers.{}.mlp.experts.", layer_idx);
    registry["experts_proj"].Run<Status>(dev_weights_map, experts_proj_name, dev_rank);
  }

  // 3. lm_head weights
  const std::string lm_head_name = "lm_head.";
  registry["lm_head"].Run<Status>(dev_weights_map, lm_head_name, dev_rank);

  // FP8 checkpoints store non-quantized weights (e.g. mlp.gate, embed, norm) as FP32; cast them to
  // weight_data_type so downstream matmul/layer layers see BF16. No-op for non-quant (BF16) checkpoints
  // (is_quant is false) and for already-BF16 weights.
  std::shared_ptr<Qwen3MoeConfig> qwen3_moe_config = std::dynamic_pointer_cast<Qwen3MoeConfig>(model_config_);
  if (qwen3_moe_config && qwen3_moe_config->is_quant &&
      qwen3_moe_config->weight_data_type != DataType::TYPE_FP32) {
    for (auto& [name, tensor] : dev_weights_map) {
      if (tensor.dtype != DataType::TYPE_FP32) continue;
      if (name.find("weight_scale") != std::string::npos || name.find("input_scale") != std::string::npos) continue;
      CastDeviceTensorType(tensor, qwen3_moe_config->weight_data_type, dev_rank);
    }
  }

  return Status();
}

}  // namespace ksana_llm