/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/bge_reranker_minicpm/bge_reranker_minicpm_weight_loader.h"

#include "model_loader/model_loader_utils.h"
#include "models/bge_reranker_minicpm/bge_reranker_minicpm_config.h"

namespace ksana_llm {

BgeRerankerMinicpmWeightLoader::BgeRerankerMinicpmWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                                               std::shared_ptr<Environment> env,
                                                               std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_shared<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;

  // Initialize WeightMethod with QuantConfig for FP8 support
  auto bge_reranker_minicpm_config = std::dynamic_pointer_cast<BgeRerankerMinicpmConfig>(model_config_);
  if (bge_reranker_minicpm_config->is_quant && bge_reranker_minicpm_config->quant_config.method == QUANT_FP8_E4M3) {
    weight_method_ =
        std::make_shared<WeightMethod>(common_weight_loader_, tp_, bge_reranker_minicpm_config->quant_config);
  } else {
    weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
  }
}

Status BgeRerankerMinicpmWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  std::vector<std::string> skip_list = {};
  std::vector<std::string> master_only_list = {"model.embed_tokens.weight"};
  std::vector<std::string> tail_only_list = {"linear_head.weight"};

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

Status BgeRerankerMinicpmWeightLoader::PreProcessModelWeights(
    const std::unordered_map<std::string, Tensor>& host_model_weights) {
  auto bge_reranker_minicpm_config = std::dynamic_pointer_cast<BgeRerankerMinicpmConfig>(model_config_);
#define REGISTER_COMMON(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetCommonMethod().get(), \
                                                        &CommonMethod::key##_##weight_name)
#define REGISTER_FP8E4M3(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetFp8E4m3Method().get(), \
                                                        &Fp8E4m3Method::key##_##weight_name)

  if (bge_reranker_minicpm_config->quant_config.method == QUANT_FP8_E4M3) {
    // Register fp8_e4m3 weight methods
    REGISTER_FP8E4M3(load, attn_q_k_v_proj);
    REGISTER_FP8E4M3(load, attn_o_proj);
    REGISTER_FP8E4M3(load, mlp_gate_up_proj);
    REGISTER_FP8E4M3(load, mlp_down_proj);

    REGISTER_FP8E4M3(process, attn_qkv_proj);
    REGISTER_FP8E4M3(process, attn_o_proj);
    REGISTER_FP8E4M3(process, mlp_gate_up_proj);
    REGISTER_FP8E4M3(process, mlp_down_proj);
  } else {
    // Register common weight methods
    REGISTER_COMMON(load, attn_q_k_v_proj);
    REGISTER_COMMON(load, attn_o_proj);
    REGISTER_COMMON(load, mlp_gate_up_proj);
    REGISTER_COMMON(load, mlp_down_proj);

    REGISTER_COMMON(process, attn_qkv_proj);
    REGISTER_COMMON(process, attn_o_proj);
    REGISTER_COMMON(process, mlp_gate_up_proj);
    REGISTER_COMMON(process, mlp_down_proj);
  }

  REGISTER_COMMON(load, norm);
  REGISTER_COMMON(load, embed_tokens);

  REGISTER_COMMON(process, lm_head);

#undef REGISTER_COMMON
#undef REGISTER_FP8E4M3
  return Status();
}

Status BgeRerankerMinicpmWeightLoader::ProcessModelWeights(
    const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
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

    // 2. mlp weights
    if (CheckWeightNameMatched(host_weight_name, {".mlp.gate_proj", ".mlp.up_proj"}, MatchMode::PartialMatch)) {
      registry["mlp_gate_up_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".mlp.down_proj"}, MatchMode::PartialMatch)) {
      registry["mlp_down_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
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
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {"lm_head"}, MatchMode::PartialMatch)) {
      registry["norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
  }

  return Status();
}

Status BgeRerankerMinicpmWeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                               int dev_rank) {
  auto registry = weight_method_->GetRegistry()["process"];
  auto bge_reranker_minicpm_config = std::dynamic_pointer_cast<BgeRerankerMinicpmConfig>(model_config_);

  // Cast fp32 weights to the model's dtype
  for (auto& [weight_name, dev_tensor] : dev_weights_map) {
    if (!CheckWeightNameMatched(weight_name, {".weight_scale", ".input_scale"}, MatchMode::PartialMatch) &&
        dev_tensor.dtype == TYPE_FP32) {
      common_weight_loader_->CastDeviceTensorType(dev_tensor, bge_reranker_minicpm_config->weight_data_type, dev_rank);
    }
  }

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; layer_idx++) {
    // 1. attn weights
    const std::string q_proj_name = fmt::format("model.layers.{}.self_attn.q_proj.", layer_idx);
    registry["attn_qkv_proj"].Run<Status>(dev_weights_map, q_proj_name, dev_rank);

    const std::string o_proj_name = fmt::format("model.layers.{}.self_attn.o_proj.", layer_idx);
    registry["attn_o_proj"].Run<Status>(dev_weights_map, o_proj_name, dev_rank);

    // 2. mlp weights
    const std::string gate_proj_name = fmt::format("model.layers.{}.mlp.gate_proj.", layer_idx);
    registry["mlp_gate_up_proj"].Run<Status>(dev_weights_map, gate_proj_name, dev_rank);

    const std::string down_proj_name = fmt::format("model.layers.{}.mlp.down_proj.", layer_idx);
    registry["mlp_down_proj"].Run<Status>(dev_weights_map, down_proj_name, dev_rank);
  }

  for (uint32_t layer_idx = 0;
       layer_idx <= bge_reranker_minicpm_config->num_layer - bge_reranker_minicpm_config->start_layer; layer_idx++) {
    // 3. lm_head weights
    const std::string lm_head_name = fmt::format("lm_head.{}.linear_head.", layer_idx);
    registry["lm_head"].Run<Status>(dev_weights_map, lm_head_name, dev_rank);
  }

  return Status();
}

}  // namespace ksana_llm
