/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/qwen2_moe/new_qwen2_moe_weight_loader.h"

#include <algorithm>

#include "model_loader/model_loader_utils.h"
#include "models/base/utils.h"
#include "models/qwen2_moe/new_qwen2_moe_config.h"

namespace ksana_llm {

NewQwen2MoeWeightLoader::NewQwen2MoeWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                                 std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_shared<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;
  weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
}

Status NewQwen2MoeWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
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

void NewQwen2MoeWeightLoader::ComputeExpertPartition(int dev_rank, size_t num_experts, size_t& num_experts_per_rank,
                                                     size_t& expert_start, size_t& expert_end, size_t& moe_tp_size,
                                                     size_t& tp_rank) const {
  // Mirrors csrc/models/quant/quant_weight.cpp so the BF16 layout matches what the MoE kernels expect.
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
// "model.layers.<L>.mlp.experts.<E>.gate_proj.weight". Returns -1 if not a routed-expert weight
// (e.g. shared_expert / dense MLP), which must keep the standard full-TP split.
static int ParseExpertIdx(const std::string& weight_name) {
  const std::string marker = ".mlp.experts.";
  size_t pos = weight_name.find(marker);
  if (pos == std::string::npos) return -1;
  pos += marker.size();
  size_t end = weight_name.find('.', pos);
  if (end == std::string::npos || end == pos) return -1;
  return std::stoi(weight_name.substr(pos, end - pos));
}

Status NewQwen2MoeWeightLoader::load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                      const std::string& weight_name, const Tensor& weight_tensor,
                                                      int dev_rank) {
  if (!CheckWeightNameEndMatched(weight_name, {".weight"})) return Status();

  const int expert_idx = ParseExpertIdx(weight_name);
  if (expert_idx < 0) {
    // Non-routed weight (shared expert / dense MLP): standard full-TP split (unchanged behavior).
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
    return Status();
  }

  std::shared_ptr<NewQwen2MoeConfig> new_qwen2_moe_config = std::dynamic_pointer_cast<NewQwen2MoeConfig>(model_config_);
  const size_t num_experts = new_qwen2_moe_config->moe_config.num_experts;
  size_t num_experts_per_rank, expert_start, expert_end, moe_tp_size, tp_rank;
  ComputeExpertPartition(dev_rank, num_experts, num_experts_per_rank, expert_start, expert_end, moe_tp_size, tp_rank);

  // Skip routed experts that do not belong to this rank's expert-parallel shard.
  if (static_cast<size_t>(expert_idx) < expert_start || static_cast<size_t>(expert_idx) >= expert_end) {
    return Status();
  }
  // Split the intermediate dim by the moe tensor-parallel group (RowPara), not the full tensor_parallel_size.
  device_model_weights[weight_name] = common_weight_loader_->TensorParallelSplit(
      weight_tensor, static_cast<int>(tp_rank), moe_tp_size, TensorParallelMode::RowPara);
  return Status();
}

Status NewQwen2MoeWeightLoader::load_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                   const std::string& weight_name, const Tensor& weight_tensor,
                                                   int dev_rank) {
  if (!CheckWeightNameEndMatched(weight_name, {".weight"})) return Status();

  const int expert_idx = ParseExpertIdx(weight_name);
  if (expert_idx < 0) {
    // Non-routed weight (shared expert / dense MLP): standard full-TP split (unchanged behavior).
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::ColPara);
    return Status();
  }

  std::shared_ptr<NewQwen2MoeConfig> new_qwen2_moe_config = std::dynamic_pointer_cast<NewQwen2MoeConfig>(model_config_);
  const size_t num_experts = new_qwen2_moe_config->moe_config.num_experts;
  size_t num_experts_per_rank, expert_start, expert_end, moe_tp_size, tp_rank;
  ComputeExpertPartition(dev_rank, num_experts, num_experts_per_rank, expert_start, expert_end, moe_tp_size, tp_rank);

  if (static_cast<size_t>(expert_idx) < expert_start || static_cast<size_t>(expert_idx) >= expert_end) {
    return Status();
  }
  device_model_weights[weight_name] = common_weight_loader_->TensorParallelSplit(
      weight_tensor, static_cast<int>(tp_rank), moe_tp_size, TensorParallelMode::ColPara);
  return Status();
}

Status NewQwen2MoeWeightLoader::process_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                     const std::string& weight_prefix_name, int dev_rank) {
  std::shared_ptr<NewQwen2MoeConfig> new_qwen2_moe_config = std::dynamic_pointer_cast<NewQwen2MoeConfig>(model_config_);
  const size_t num_experts = new_qwen2_moe_config->moe_config.num_experts;

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
      for (size_t expert_idx = expert_start; expert_idx < expert_end; expert_idx++) {
#ifdef ENABLE_KUNLUN
        // Kunlun (xblas moe_fc_fusion + infer_ops::swiglu with turn=true):
        // swiglu expects [gate|up] layout, i.e. SiLU(first_half) * second_half.
        // 对齐 xsglang 的 w13 = [gate_proj(idx=0) | up_proj(idx=1)] 顺序。
        concat_names.push_back(fmt::format("{}{}.gate_proj.{}", weight_prefix_name, expert_idx, weight_suffix));
        concat_names.push_back(fmt::format("{}{}.up_proj.{}", weight_prefix_name, expert_idx, weight_suffix));
#else
        // NVIDIA (InvokeFusedMoe): 保持原始 [up|gate] 顺序不变。
        concat_names.push_back(fmt::format("{}{}.up_proj.{}", weight_prefix_name, expert_idx, weight_suffix));
        concat_names.push_back(fmt::format("{}{}.gate_proj.{}", weight_prefix_name, expert_idx, weight_suffix));
#endif
      }
      common_weight_loader_->AutoMergeWeight(concat_names, gate_up_proj_name + weight_suffix, device_model_weights,
                                             dev_rank);
      // shape检查与调整
      const std::vector<size_t> old_shape = device_model_weights.at(gate_up_proj_name + weight_suffix).shape;
      const std::vector<size_t> new_shape = {local_expert_num,
                                             2 * new_qwen2_moe_config->moe_config.moe_inter_size / moe_tp_size,
                                             new_qwen2_moe_config->hidden_units};
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
      const std::vector<size_t> new_shape = {local_expert_num, new_qwen2_moe_config->hidden_units,
                                             new_qwen2_moe_config->moe_config.moe_inter_size / moe_tp_size};
      KLLM_CHECK((old_shape[0] == (new_shape[0] * new_shape[1])) && (old_shape[1] == new_shape[2]));
      device_model_weights.at(down_proj_name + weight_suffix).shape = new_shape;
    }
  }
  return Status();
}

Status NewQwen2MoeWeightLoader::process_attn_qkv_bias(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                      const std::string& weight_prefix_name, int dev_rank) {
  std::shared_ptr<NewQwen2MoeConfig> new_qwen2_moe_config = std::dynamic_pointer_cast<NewQwen2MoeConfig>(model_config_);

  const std::string q_bias_name = weight_prefix_name + "bias";
  const std::string k_bias_name = WeightNameReplace(q_bias_name, "q_proj", "k_proj");
  const std::string v_bias_name = WeightNameReplace(q_bias_name, "q_proj", "v_proj");
  const std::string qkv_bias_name = WeightNameReplace(q_bias_name, "q_proj", "query_key_value");

  common_weight_loader_->AutoMergeWeight({q_bias_name, k_bias_name, v_bias_name}, qkv_bias_name, device_model_weights,
                                         dev_rank);

  const std::vector<size_t> old_shape = device_model_weights.at(qkv_bias_name).shape;
  std::vector<size_t> new_shape = {1, new_qwen2_moe_config->head_num / new_qwen2_moe_config->num_key_value_heads + 2,
                                   0};
  new_shape[2] = old_shape[0] / new_shape[1];
  device_model_weights.at(qkv_bias_name).shape = new_shape;
  return Status();
}

Status NewQwen2MoeWeightLoader::PreProcessModelWeights(
    const std::unordered_map<std::string, Tensor>& host_model_weights) {
#define REGISTER_COMMON(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetCommonMethod().get(), \
                                                        &CommonMethod::key##_##weight_name)
#define REGISTER_CUSTOM(key, weight_name) \
  weight_method_->GetRegistry()[#key][#weight_name].Add(this, &NewQwen2MoeWeightLoader::key##_##weight_name)

  // Register common weight processing methods
  REGISTER_COMMON(load, attn_q_k_v_proj);
  REGISTER_COMMON(load, attn_o_proj);
  REGISTER_COMMON(load, experts_gate);
  // EP-aware expert loaders (keep only local routed experts, split by moe_tensor_para_size).
  // Shared-expert / dense-MLP weights fall back to the standard full-TP split inside these.
  REGISTER_CUSTOM(load, mlp_gate_up_proj);
  REGISTER_CUSTOM(load, mlp_down_proj);
  REGISTER_COMMON(load, norm);
  REGISTER_COMMON(load, embed_tokens);
  REGISTER_COMMON(load, lm_head);

  REGISTER_COMMON(process, attn_qkv_proj);
  REGISTER_COMMON(process, attn_o_proj);
  REGISTER_COMMON(process, experts_gate);
  REGISTER_COMMON(process, mlp_gate_up_proj);
  REGISTER_COMMON(process, mlp_down_proj);
  REGISTER_COMMON(process, lm_head);

  REGISTER_CUSTOM(process, attn_qkv_bias);
  REGISTER_CUSTOM(process, experts_proj);

#undef REGISTER_COMMON
#undef REGISTER_CUSTOM
  return Status();
}

Status NewQwen2MoeWeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
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

    // 3. shared expert weights
    if (CheckWeightNameMatched(host_weight_name, {".mlp.shared_expert_gate."}, MatchMode::PartialMatch)) {
      registry["experts_gate"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj"},
                               MatchMode::PartialMatch)) {
      registry["mlp_gate_up_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".mlp.shared_expert.down_proj"}, MatchMode::PartialMatch)) {
      registry["mlp_down_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 4. norm weights
    if (CheckWeightNameMatched(host_weight_name, {"model.norm", ".input_layernorm", ".post_attention_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 5. embedding weights && lm_head weights
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

Status NewQwen2MoeWeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                        int dev_rank) {
  auto registry = weight_method_->GetRegistry()["process"];

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; layer_idx++) {
    // 1. attn weights
    const std::string q_proj_name = fmt::format("model.layers.{}.self_attn.q_proj.", layer_idx);
    registry["attn_qkv_proj"].Run<Status>(dev_weights_map, q_proj_name, dev_rank);
    registry["attn_qkv_bias"].Run<Status>(dev_weights_map, q_proj_name, dev_rank);

    const std::string o_proj_name = fmt::format("model.layers.{}.self_attn.o_proj.", layer_idx);
    registry["attn_o_proj"].Run<Status>(dev_weights_map, o_proj_name, dev_rank);

    // 2. experts weights
    const std::string gate_name = fmt::format("model.layers.{}.mlp.gate.", layer_idx);
    registry["experts_gate"].Run<Status>(dev_weights_map, gate_name, dev_rank);

    const std::string experts_proj_name = fmt::format("model.layers.{}.mlp.experts.", layer_idx);
    registry["experts_proj"].Run<Status>(dev_weights_map, experts_proj_name, dev_rank);

    // 3. shared expert weights
    const std::string shared_expert_gate_name = fmt::format("model.layers.{}.mlp.shared_expert_gate.", layer_idx);
    registry["experts_gate"].Run<Status>(dev_weights_map, shared_expert_gate_name, dev_rank);

    const std::string shared_expert_gate_proj_name =
        fmt::format("model.layers.{}.mlp.shared_expert.gate_proj.", layer_idx);
    registry["mlp_gate_up_proj"].Run<Status>(dev_weights_map, shared_expert_gate_proj_name, dev_rank);

    const std::string shared_expert_down_proj_name =
        fmt::format("model.layers.{}.mlp.shared_expert.down_proj.", layer_idx);
    registry["mlp_down_proj"].Run<Status>(dev_weights_map, shared_expert_down_proj_name, dev_rank);
  }

  // 4. lm_head weights
  const std::string lm_head_name = "lm_head.";
  registry["lm_head"].Run<Status>(dev_weights_map, lm_head_name, dev_rank);

  return Status();
}

}  // namespace ksana_llm