/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/qwen2_moe/new_qwen2_moe_weight_loader.h"

#include "model_loader/model_loader_utils.h"
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

Status NewQwen2MoeWeightLoader::process_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                     const std::string& weight_prefix_name, int dev_rank) {
  std::shared_ptr<NewQwen2MoeConfig> new_qwen2_moe_config = std::dynamic_pointer_cast<NewQwen2MoeConfig>(model_config_);
  // 合并gate_up
  {
    const std::string gate_up_proj_name = fmt::format("{}up_gate_proj.", weight_prefix_name);
    const std::vector<std::string> weight_suffixs = {"weight"};
    for (const std::string& weight_suffix : weight_suffixs) {
      std::vector<std::string> concat_names;
      for (int expert_idx = 0; expert_idx < new_qwen2_moe_config->moe_config.num_experts; expert_idx++) {
        concat_names.push_back(fmt::format("{}{}.up_proj.{}", weight_prefix_name, expert_idx, weight_suffix));
        concat_names.push_back(fmt::format("{}{}.gate_proj.{}", weight_prefix_name, expert_idx, weight_suffix));
      }
      common_weight_loader_->AutoMergeWeight(concat_names, gate_up_proj_name + weight_suffix, device_model_weights,
                                             dev_rank);
      // shape检查与调整
      const std::vector<size_t> old_shape = device_model_weights.at(gate_up_proj_name + weight_suffix).shape;
      const std::vector<size_t> new_shape = {new_qwen2_moe_config->moe_config.num_experts,
                                             2 * new_qwen2_moe_config->moe_config.moe_inter_size / tp_,
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
      for (int expert_idx = 0; expert_idx < new_qwen2_moe_config->moe_config.num_experts; expert_idx++) {
        concat_names.push_back(fmt::format("{}{}.down_proj.{}", weight_prefix_name, expert_idx, weight_suffix));
      }
      common_weight_loader_->AutoMergeWeight(concat_names, down_proj_name + weight_suffix, device_model_weights,
                                             dev_rank);
      // shape检查与调整
      const std::vector<size_t> old_shape = device_model_weights.at(down_proj_name + weight_suffix).shape;
      const std::vector<size_t> new_shape = {new_qwen2_moe_config->moe_config.num_experts,
                                             new_qwen2_moe_config->hidden_units,
                                             new_qwen2_moe_config->moe_config.moe_inter_size / tp_};
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
  REGISTER_COMMON(load, mlp_gate_up_proj);
  REGISTER_COMMON(load, mlp_down_proj);
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