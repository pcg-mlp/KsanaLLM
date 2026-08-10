/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/gpt_moe/gpt_moe_weight_loader.h"

#include "model_loader/model_loader_utils.h"
#include "models/gpt_moe/gpt_moe_config.h"

namespace ksana_llm {

GptMoeWeightLoader::GptMoeWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                                       std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_shared<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;
  weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
}

Status GptMoeWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  std::vector<std::string> skip_list = {};
  std::vector<std::string> master_and_tail_only_list = {"model.embed_tokens.weight"};

  int lower_layer_idx = pipeline_config_.lower_layer_idx;
  int upper_layer_idx = pipeline_config_.upper_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, false)) {
      weight_names.erase(it);
      continue;
    }

    // Skip some layers in distributed mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      int layer_idx = GetLayerIdxFromName(*it);
      if (layer_idx >= 0 && (layer_idx < lower_layer_idx || layer_idx > upper_layer_idx)) {
        weight_names.erase(it);
        continue;
      }
    }

    if (!context_->global->IsStandalone()) {
      // tie-word-embeddings of gpt_moe model is true
      if (context_->global->IsDistributedWorker() && !context_->global->IsDistributedLastWorker()) {
        if (CheckWeightNameMatched(*it, master_and_tail_only_list, false)) {
          weight_names.erase(it);
          continue;
        }
      }
    }
    ++it;
  }
  return Status();
}

Status GptMoeWeightLoader::load_attn_fused_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                    const std::string& weight_name, const Tensor& weight_tensor,
                                                    int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    auto gpt_moe_config = std::dynamic_pointer_cast<GptMoeConfig>(model_config_);
    KLLM_CHECK(weight_tensor.shape[0] % tp_ == 0);

    const size_t head_num = gpt_moe_config->head_num;
    const size_t num_kv_heads = gpt_moe_config->num_key_value_heads;
    KLLM_CHECK(head_num % num_kv_heads == 0);

    const std::vector<size_t> qkv_proj_shape = {weight_tensor.shape[0] / tp_, weight_tensor.shape[1]};
    Tensor qkv_proj_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, weight_tensor.dtype, qkv_proj_shape);
    const size_t qkv_pitch = qkv_proj_tensor.GetTotalBytes() / ((head_num / num_kv_heads) + 2);
    const size_t q_size = (head_num / num_kv_heads) * qkv_pitch;
    const size_t q_para_offset = dev_rank * q_size;
    const size_t kv_para_offset = dev_rank * qkv_pitch;

    MemcpyAsync(qkv_proj_tensor.GetPtr<void>(), weight_tensor.GetPtr<void>() + q_para_offset, q_size,
                MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
    MemcpyAsync(qkv_proj_tensor.GetPtr<void>() + q_size, weight_tensor.GetPtr<void>() + q_size * tp_ + kv_para_offset,
                qkv_pitch, MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
    MemcpyAsync(qkv_proj_tensor.GetPtr<void>() + q_size + qkv_pitch,
                weight_tensor.GetPtr<void>() + kv_para_offset + (q_size + qkv_pitch) * tp_, qkv_pitch,
                MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

    device_model_weights[weight_name] = qkv_proj_tensor;
  }

  return Status();
}

Status GptMoeWeightLoader::process_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                const std::string& weight_prefix_name, int dev_rank) {
  auto gpt_moe_config = std::dynamic_pointer_cast<GptMoeConfig>(model_config_);
  KLLM_CHECK(gpt_moe_config->moe_config.moe_inter_size % tp_ == 0);
  // 合并gate
  {
    const std::string gate_proj_name = fmt::format("{}gate_proj.", weight_prefix_name);
    const std::vector<std::string> weight_suffixs = {"weight"};
    for (const std::string& weight_suffix : weight_suffixs) {
      std::vector<std::string> concat_names;
      for (size_t expert_idx = 0; expert_idx < gpt_moe_config->moe_config.num_experts; expert_idx++) {
        concat_names.push_back(fmt::format("{}{}.gate_proj.{}", weight_prefix_name, expert_idx, weight_suffix));
      }
      common_weight_loader_->AutoMergeWeight(concat_names, gate_proj_name + weight_suffix, device_model_weights,
                                             dev_rank);
      // shape检查与调整
      auto& old_shape = device_model_weights.at(gate_proj_name + weight_suffix).shape;
      const std::vector<size_t> new_shape = {gpt_moe_config->moe_config.num_experts,
                                             gpt_moe_config->moe_config.moe_inter_size / tp_,
                                             gpt_moe_config->hidden_units};
      KLLM_CHECK((old_shape[0] == (new_shape[0] * new_shape[1])) && (old_shape[1] == new_shape[2]));
      old_shape = new_shape;
    }
  }
  // 合并down
  {
    const std::string down_proj_name = fmt::format("{}down_proj.", weight_prefix_name);
    const std::vector<std::string> weight_suffixs = {"weight"};
    for (const std::string& weight_suffix : weight_suffixs) {
      std::vector<std::string> concat_names;
      for (size_t expert_idx = 0; expert_idx < gpt_moe_config->moe_config.num_experts; expert_idx++) {
        concat_names.push_back(fmt::format("{}{}.down_proj.{}", weight_prefix_name, expert_idx, weight_suffix));
      }
      common_weight_loader_->AutoMergeWeight(concat_names, down_proj_name + weight_suffix, device_model_weights,
                                             dev_rank);
      // shape检查与调整
      auto& old_shape = device_model_weights.at(down_proj_name + weight_suffix).shape;
      const std::vector<size_t> new_shape = {gpt_moe_config->moe_config.num_experts, gpt_moe_config->hidden_units,
                                             gpt_moe_config->moe_config.moe_inter_size / tp_};
      KLLM_CHECK((old_shape[0] == (new_shape[0] * new_shape[1])) && (old_shape[1] == new_shape[2]));
      old_shape = new_shape;
    }
  }
  return Status();
}

Status GptMoeWeightLoader::PreProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights) {
#define REGISTER_COMMON(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetCommonMethod().get(), \
                                                        &CommonMethod::key##_##weight_name)
#define REGISTER_CUSTOM(key, weight_name) \
  weight_method_->GetRegistry()[#key][#weight_name].Add(this, &GptMoeWeightLoader::key##_##weight_name)

  // Register common weight methods
  REGISTER_COMMON(load, attn_o_proj);
  REGISTER_COMMON(load, mlp_gate_up_proj);
  REGISTER_COMMON(load, mlp_down_proj);
  REGISTER_COMMON(load, experts_gate);
  REGISTER_COMMON(load, norm);
  REGISTER_COMMON(load, embed_tokens);
  REGISTER_COMMON(load, lm_head);

  REGISTER_COMMON(process, attn_fused_qkv_proj);
  REGISTER_COMMON(process, attn_o_proj);
  REGISTER_COMMON(process, mlp_down_proj);
  REGISTER_COMMON(process, lm_head);

  // Register custom weight methods
  REGISTER_CUSTOM(load, attn_fused_qkv_proj);

  REGISTER_CUSTOM(process, experts_proj);

#undef REGISTER_COMMON
#undef REGISTER_CUSTOM
  return Status();
}

Status GptMoeWeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                               int dev_rank,
                                               std::unordered_map<std::string, Tensor>& device_model_weights,
                                               std::unordered_map<std::string, Tensor>& left_host_weights) {
  auto registry = weight_method_->GetRegistry()["load"];
  auto gpt_moe_config = std::dynamic_pointer_cast<GptMoeConfig>(model_config_);

  for (auto& [host_weight_name, host_weight_tensor] : host_model_weights) {
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}", dev_rank, host_weight_name,
                                  Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));

    // 1. attn weights
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.W_pack"}, MatchMode::PartialMatch)) {
      const std::string qkv_proj_name = WeightNameReplace(host_weight_name, "W_pack", "query_key_value");
      registry["attn_fused_qkv_proj"].Run<Status>(device_model_weights, qkv_proj_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.o_proj"}, MatchMode::PartialMatch)) {
      registry["attn_o_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 2. expert weights
    if (CheckWeightNameMatched(host_weight_name, {".mlp.gate"}, MatchMode::PartialMatch)) {
      registry["experts_gate"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".mlp.experts."}, MatchMode::PartialMatch)) {
      if (CheckWeightNameMatched(host_weight_name, {".up_proj"}, MatchMode::PartialMatch)) {
        const std::string gate_proj_name = WeightNameReplace(host_weight_name, "up_proj", "gate_proj");
        registry["mlp_gate_up_proj"].Run<Status>(device_model_weights, gate_proj_name, host_weight_tensor, dev_rank);
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

    // 4. embedding weight && lm_head weight
    if (CheckWeightNameMatched(host_weight_name, {"model.embed_tokens"}, MatchMode::PartialMatch)) {
      registry["embed_tokens"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      KLLM_CHECK(gpt_moe_config->tie_word_embeddings);
      const std::string lm_head_name = "lm_head.weight";
      registry["lm_head"].Run<Status>(device_model_weights, lm_head_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {"model.embed_positions"}, MatchMode::PartialMatch)) {
      registry["embed_tokens"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
  }

  return Status();
}

Status GptMoeWeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                   int dev_rank) {
  auto registry = weight_method_->GetRegistry()["process"];
  auto gpt_moe_config = std::dynamic_pointer_cast<GptMoeConfig>(model_config_);

  // Cast fp32 weights to the model's dtype
  for (auto& [weight_name, dev_tensor] : dev_weights_map) {
    if (!CheckWeightNameMatched(weight_name, {".weight_scale", ".input_scale"}, MatchMode::PartialMatch) &&
        dev_tensor.dtype == TYPE_FP32) {
      common_weight_loader_->CastDeviceTensorType(dev_tensor, gpt_moe_config->weight_data_type, dev_rank);
    }
  }

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; layer_idx++) {
    // 1. attn weights
    const std::string qkv_proj_name = fmt::format("model.layers.{}.self_attn.query_key_value.", layer_idx);
    registry["attn_fused_qkv_proj"].Run<Status>(dev_weights_map, qkv_proj_name, dev_rank);

    const std::string o_proj_name = fmt::format("model.layers.{}.self_attn.o_proj.", layer_idx);
    registry["attn_o_proj"].Run<Status>(dev_weights_map, o_proj_name, dev_rank);

    // 2. expert weights
    const std::string gate_name = fmt::format("model.layers.{}.mlp.gate.", layer_idx);
    registry["mlp_down_proj"].Run<Status>(dev_weights_map, gate_name, dev_rank);

    const std::string experts_proj_name = fmt::format("model.layers.{}.mlp.experts.", layer_idx);
    registry["experts_proj"].Run<Status>(dev_weights_map, experts_proj_name, dev_rank);
  }

  // 3. lm_head weight
  const std::string lm_head_name = "lm_head.";
  registry["lm_head"].Run<Status>(dev_weights_map, lm_head_name, dev_rank);

  return Status();
}

}  // namespace ksana_llm
