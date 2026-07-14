/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <string>

#include "device/device_types.h"
#include "kernels/trans_layout.h"
#include "model_loader/model_loader_utils.h"
#include "models/base/model_arch.h"
#include "models/base/model_format.h"
#include "models/qwen/qwen_config.h"
#include "models/qwen/qwen_weight_loader.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/ret_code.h"
#include "utils/status.h"
#include "utils/string_utils.h"
#include "utils/tensor.h"

#ifdef ENABLE_CUDA
#  include "kernels/nvidia/kernel_wrapper.h"
#endif

namespace ksana_llm {

QwenWeightLoader::QwenWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                                   std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_shared<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;

  // Initialize WeightMethod with QuantConfig for FP8/AWQ/GPTQ support
  std::shared_ptr<QwenConfig> qwen_config = std::dynamic_pointer_cast<QwenConfig>(model_config);
  if (qwen_config && qwen_config->is_quant) {
    weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_, qwen_config->quant_config);
  } else {
    weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
  }
}

QwenWeightLoader::~QwenWeightLoader() {}

Status QwenWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq"};
  std::vector<std::string> master_only_list = {"model.embed_tokens.weight"};
  std::vector<std::string> tail_only_list = {"lm_head.weight"};

  std::shared_ptr<QwenConfig> qwen_config = std::dynamic_pointer_cast<QwenConfig>(model_config_);

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
      // Skip embedding on if it's not master mode or tail mode when tie_word_embeddings is true.
      if (context_->global->IsDistributedWorker()) {
        if (CheckWeightNameMatched(*it, master_only_list, false)) {
          if (!(qwen_config && qwen_config->tie_word_embeddings && context_->global->IsDistributedLastWorker())) {
            weight_names.erase(it);
            continue;
          }
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

Status QwenWeightLoader::PreProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights) {
  std::shared_ptr<QwenConfig> qwen_config = std::dynamic_pointer_cast<QwenConfig>(model_config_);

#define REGISTER_COMMON(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetCommonMethod().get(), \
                                                        &CommonMethod::key##_##weight_name)
#define REGISTER_AWQ(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetAwqMethod().get(), \
                                                        &AwqMethod::key##_##weight_name)
#define REGISTER_W4A8AWQ(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetW4A8AWQMethod().get(), \
                                                        &W4A8AWQMethod::key##_##weight_name)
#define REGISTER_GPTQ(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetGptqMethod().get(), \
                                                        &GptqMethod::key##_##weight_name)
#define REGISTER_FP8E4M3(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetFp8E4m3Method().get(), \
                                                        &Fp8E4m3Method::key##_##weight_name)

  if (qwen_config->quant_config.method == QUANT_AWQ) {
    REGISTER_AWQ(load, attn_q_k_v_proj);
    REGISTER_AWQ(load, attn_o_proj);
    REGISTER_AWQ(load, mlp_gate_up_proj);
    REGISTER_AWQ(load, mlp_down_proj);
    REGISTER_AWQ(process, attn_qkv_proj);
    REGISTER_AWQ(process, attn_o_proj);
    REGISTER_AWQ(process, mlp_gate_up_proj);
    REGISTER_AWQ(process, mlp_down_proj);
  } else if (qwen_config->quant_config.method == QUANT_W4A8_AWQ) {
    REGISTER_W4A8AWQ(load, attn_q_k_v_proj);
    REGISTER_W4A8AWQ(load, attn_o_proj);
    REGISTER_W4A8AWQ(load, mlp_gate_up_proj);
    REGISTER_W4A8AWQ(load, mlp_down_proj);
    REGISTER_W4A8AWQ(process, attn_qkv_proj);
    REGISTER_W4A8AWQ(process, attn_o_proj);
    REGISTER_W4A8AWQ(process, mlp_gate_up_proj);
    REGISTER_W4A8AWQ(process, mlp_down_proj);
  } else if (qwen_config->quant_config.method == QUANT_GPTQ) {
    REGISTER_GPTQ(load, attn_q_k_v_proj);
    REGISTER_GPTQ(load, attn_o_proj);
    REGISTER_GPTQ(load, mlp_gate_up_proj);
    REGISTER_GPTQ(load, mlp_down_proj);
    REGISTER_GPTQ(process, attn_qkv_proj);
    REGISTER_GPTQ(process, attn_o_proj);
    REGISTER_GPTQ(process, mlp_gate_up_proj);
    REGISTER_GPTQ(process, mlp_down_proj);
  } else if (qwen_config->quant_config.method == QUANT_FP8_E4M3) {
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

  REGISTER_COMMON(load, attn_norm);
  REGISTER_COMMON(load, norm);
  REGISTER_COMMON(load, embed_tokens);
  REGISTER_COMMON(load, lm_head);

  REGISTER_COMMON(process, attn_qkv_bias);
  REGISTER_COMMON(process, lm_head);

#undef REGISTER_COMMON
#undef REGISTER_AWQ
#undef REGISTER_W4A8AWQ
#undef REGISTER_GPTQ
#undef REGISTER_FP8E4M3

  return Status();
}

Status QwenWeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                             int dev_rank,
                                             std::unordered_map<std::string, Tensor>& device_model_weights,
                                             std::unordered_map<std::string, Tensor>& left_host_weights) {
  auto registry = weight_method_->GetRegistry()["load"];
  auto qwen_config = std::dynamic_pointer_cast<QwenConfig>(model_config_);

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

    // 4. embedding weights && lm_head weights
    if (CheckWeightNameMatched(host_weight_name, {"model.embed_tokens"}, MatchMode::PartialMatch)) {
      // Load embed_tokens if it's the master node
      if (!context_->global->IsDistributedWorker()) {
        registry["embed_tokens"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      }
      // When tie_word_embeddings is true, standalone or tail worker needs to load lm_head as well
      if (qwen_config->tie_word_embeddings &&
          (context_->global->IsDistributedLastWorker() || context_->global->IsStandalone())) {
        const std::string lm_head_weight_name = "lm_head.weight";
        registry["lm_head"].Run<Status>(device_model_weights, lm_head_weight_name, host_weight_tensor, dev_rank);
      }
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name, {"lm_head"}, MatchMode::PartialMatch)) {
      registry["lm_head"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // TODO(jinxcwu) 后续找到具体模型再优化这一部份
    if (CheckWeightNameMatched(host_weight_name,
                               {".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"},
                               MatchMode::PartialMatch)) {
      const std::string query_key_value_bias_name =
          host_weight_name.substr(0, host_weight_name.find(".self_attn.")) + ".self_attn.query_key_value.bias";
      size_t host_shape0_split = DivRoundUp(host_weight_tensor.shape[0], tp_);
      if (device_model_weights.find(query_key_value_bias_name) == device_model_weights.end()) {
        Tensor query_key_value_bias =
            Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, {1, 3, host_shape0_split});
        device_model_weights[query_key_value_bias_name] = query_key_value_bias;
      }

      Tensor& query_key_value_bias_tensor = device_model_weights.at(query_key_value_bias_name);
      size_t host_offset = host_shape0_split * GetTypeSize(host_weight_tensor.dtype) * dev_rank;
      size_t dev_offset = 0;
      if (host_weight_name.find("q_proj.bias") != std::string::npos) {
        dev_offset = 0;
      } else if (host_weight_name.find("k_proj.bias") != std::string::npos) {
        dev_offset = host_shape0_split * GetTypeSize(host_weight_tensor.dtype);
      } else if (host_weight_name.find("v_proj.bias") != std::string::npos) {
        dev_offset = host_shape0_split * GetTypeSize(host_weight_tensor.dtype) * 2;
      }
      MemcpyAsync(query_key_value_bias_tensor.GetPtr<void>() + dev_offset,
                  host_weight_tensor.GetPtr<void>() + host_offset,
                  host_shape0_split * GetTypeSize(host_weight_tensor.dtype), MEMCPY_HOST_TO_DEVICE,
                  context_->device->GetMemoryManageStream());
      continue;
    }
  }
  return Status();
}

Status QwenWeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                 int dev_rank) {
  auto registry = weight_method_->GetRegistry()["process"];

  auto qwen_config = std::dynamic_pointer_cast<QwenConfig>(model_config_);

  // Cast weights that need to match config.torch_dtype.
  for (auto& [weight_name, dev_tensor] : dev_weights_map) {
    // Get the filter suffixes for current quantization method.
    // Skip quantization-specific weights based on the filter dictionary.
    if (qwen_config->is_quant) {
      auto it = kQuantWeightSuffixFilters.find(qwen_config->quant_config.method);
      if (it != kQuantWeightSuffixFilters.end() &&
          CheckWeightNameMatched(weight_name, it->second, MatchMode::PartialMatch)) {
        continue;
      }
    }

    // Only cast FP32/FP16/BF16 weights to the target weight_data_type.
    if (!(dev_tensor.dtype == DataType::TYPE_FP16 || dev_tensor.dtype == DataType::TYPE_BF16 ||
          dev_tensor.dtype == DataType::TYPE_FP32)) {
      continue;
    }

    KLLM_LOG_DEBUG << fmt::format("Dev_rank {}, cast weight {}, from {} to {}", dev_rank, weight_name, dev_tensor.dtype,
                                  qwen_config->weight_data_type);
    common_weight_loader_->CastDeviceTensorType(dev_tensor, qwen_config->weight_data_type, dev_rank);
  }

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; layer_idx++) {
    // 1. attn weights
    const std::string q_proj_name = fmt::format("model.layers.{}.self_attn.q_proj.", layer_idx);
    registry["attn_qkv_proj"].Run<Status>(dev_weights_map, q_proj_name, dev_rank);
    registry["attn_qkv_bias"].Run<Status>(dev_weights_map, q_proj_name, dev_rank);

    const std::string o_proj_name = fmt::format("model.layers.{}.self_attn.o_proj.", layer_idx);
    registry["attn_o_proj"].Run<Status>(dev_weights_map, o_proj_name, dev_rank);

    // 2. mlp weights
    const std::string gate_proj_name = fmt::format("model.layers.{}.mlp.gate_proj.", layer_idx);
    registry["mlp_gate_up_proj"].Run<Status>(dev_weights_map, gate_proj_name, dev_rank);

    // Should process gate_proj and up_proj separately in awq quant mode
    // TODO(winminkong): int4量化合并gate_proj和up_proj同时计算
    if (qwen_config->quant_config.method == QUANT_AWQ || qwen_config->quant_config.method == QUANT_GPTQ) {
      const std::string up_proj_name = fmt::format("model.layers.{}.mlp.up_proj.", layer_idx);
      registry["mlp_gate_up_proj"].Run<Status>(dev_weights_map, up_proj_name, dev_rank);
    }

    const std::string down_proj_name = fmt::format("model.layers.{}.mlp.down_proj.", layer_idx);
    registry["mlp_down_proj"].Run<Status>(dev_weights_map, down_proj_name, dev_rank);
  }

  // 3. lm_head weights
  const std::string lm_head_name = "lm_head.";
  registry["lm_head"].Run<Status>(dev_weights_map, lm_head_name, dev_rank);

  return Status();
}

}  // namespace ksana_llm
