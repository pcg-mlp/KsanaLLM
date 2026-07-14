/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/internlmxcomposer2/new_internlmxcomposer2_weight_loader.h"

#include <torch/torch.h>

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

NewInternlmxComposer2WeightLoader::NewInternlmxComposer2WeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                                                     std::shared_ptr<Environment> env,
                                                                     std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_shared<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;
  weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
}

NewInternlmxComposer2WeightLoader::~NewInternlmxComposer2WeightLoader() {}

Status NewInternlmxComposer2WeightLoader::FilterWeightNames(std::vector<std::string> &weight_names) {
  std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq"};
  std::vector<std::string> master_only_list = {"model.embed_tokens.weight", "model.tok_embeddings.weight"};
  std::vector<std::string> tail_only_list = {"lm_head.weight", "output.weight"};

  int lower_layer_idx = pipeline_config_.lower_layer_idx;
  int upper_layer_idx = pipeline_config_.upper_layer_idx;
  int lower_nextn_layer_idx = pipeline_config_.lower_nextn_layer_idx;
  int upper_nextn_layer_idx = pipeline_config_.upper_nextn_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, MatchMode::PartialMatch)) {
      weight_names.erase(it);
      continue;
    }

    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      int layer_idx = GetLayerIdxFromName(*it);
      if (layer_idx >= 0 && ((layer_idx < lower_layer_idx || layer_idx > upper_layer_idx) &&
                             (layer_idx < lower_nextn_layer_idx || layer_idx > upper_nextn_layer_idx))) {
        weight_names.erase(it);
        continue;
      }
    }

    if (!context_->global->IsStandalone()) {
      if (context_->global->IsDistributedWorker()) {
        if (CheckWeightNameMatched(*it, master_only_list, false)) {
          weight_names.erase(it);
          continue;
        }
      }

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

Status NewInternlmxComposer2WeightLoader::PreProcessModelWeights(
    const std::unordered_map<std::string, Tensor> &host_model_weights) {
  std::shared_ptr<InternlmxComposer2Config> config =
      std::dynamic_pointer_cast<InternlmxComposer2Config>(model_config_);

#define REGISTER_COMMON(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetCommonMethod().get(), \
                                                        &CommonMethod::key##_##weight_name)
#define REGISTER_W4A8AWQ(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetW4A8AWQMethod().get(), \
                                                        &W4A8AWQMethod::key##_##weight_name)

  if (config->quant_config.method == QUANT_W4A8_AWQ) {
    REGISTER_W4A8AWQ(load, attn_q_k_v_proj);
    REGISTER_W4A8AWQ(load, attn_o_proj);
    REGISTER_W4A8AWQ(load, mlp_gate_up_proj);
    REGISTER_W4A8AWQ(load, mlp_down_proj);
    REGISTER_W4A8AWQ(process, attn_qkv_proj);
    REGISTER_W4A8AWQ(process, attn_o_proj);
    REGISTER_W4A8AWQ(process, mlp_gate_up_proj);
    REGISTER_W4A8AWQ(process, mlp_down_proj);
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

  REGISTER_COMMON(process, lm_head);

#undef REGISTER_COMMON
#undef REGISTER_W4A8AWQ

  return Status();
}

Status NewInternlmxComposer2WeightLoader::ProcessInterleavedQKV(
    const std::string &host_weight_name, const Tensor &host_weight_tensor,
    std::unordered_map<std::string, Tensor> &device_model_weights, int dev_rank) {
  std::shared_ptr<InternlmxComposer2Config> config =
      std::dynamic_pointer_cast<InternlmxComposer2Config>(model_config_);

  int head_num = config->head_num;
  int hidden_units = config->hidden_units;
  int num_kv_heads = config->num_key_value_heads;
  int head_dim = hidden_units / head_num;
  int num_key_value_groups = head_num / num_kv_heads;

  // Get torch dtype
  torch::ScalarType torch_dtype;
  if (host_weight_tensor.dtype == TYPE_FP32) {
    torch_dtype = torch::kFloat32;
  } else if (host_weight_tensor.dtype == TYPE_FP16) {
    torch_dtype = torch::kFloat16;
  } else if (host_weight_tensor.dtype == TYPE_BF16) {
    torch_dtype = torch::kBFloat16;
  } else {
    KLLM_THROW(fmt::format("Unsupported weight data type: {}", static_cast<int>(host_weight_tensor.dtype)));
  }

  auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch_dtype);
  std::vector<size_t> weight_shape = host_weight_tensor.shape;

  torch::Tensor in = torch::from_blob(host_weight_tensor.GetPtr<void>(),
                                      std::vector<int64_t>(weight_shape.begin(), weight_shape.end()), options);
  in = in.to(options);

  // De-interleave QKV: InternLM2/InternLMXComposer2 stores QKV in interleaved format
  // Shape: [num_kv_heads, 2 + num_key_value_groups, head_dim, hidden_size]
  in = in.view({num_kv_heads, 2 + num_key_value_groups, head_dim, -1});
  auto splits = torch::split(in, {num_key_value_groups, 1, 1}, 1);
  auto query_states = splits[0].contiguous().view({num_kv_heads * num_key_value_groups * head_dim, weight_shape[1]});
  auto key_states = splits[1].contiguous().view({num_kv_heads * head_dim, weight_shape[1]});
  auto value_states = splits[2].contiguous().view({num_kv_heads * head_dim, weight_shape[1]});
  in = torch::cat({query_states, key_states, value_states}, 0);

  size_t pos = host_weight_name.find(".self_attn.");
  std::string qkv_name;
  if (pos != std::string::npos) {
    qkv_name = host_weight_name.substr(0, pos) + ".self_attn.query_key_value.weight";
  } else {
    qkv_name = host_weight_name;
  }

  const int k = static_cast<int>(weight_shape[1]);  // hidden size
  const int q_size = head_num * head_dim;           // total Q out features (before TP)
  const int kv_size = num_kv_heads * head_dim;      // total K/V out features (before TP)
  KLLM_CHECK_WITH_INFO(q_size % static_cast<int>(tp_) == 0, "q_size must be divisible by tp");
  KLLM_CHECK_WITH_INFO(kv_size % static_cast<int>(tp_) == 0, "kv_size must be divisible by tp");

  const int q_split_size = q_size / static_cast<int>(tp_);
  const int kv_split_size = kv_size / static_cast<int>(tp_);
  const int n = q_split_size + 2 * kv_split_size;  // per-rank out features

  torch::Tensor in_t = in.transpose(0, 1).contiguous();  // [k, n_total]
  const int q_start = q_split_size * dev_rank;
  const int k_start = q_size + kv_split_size * dev_rank;
  const int v_start = q_size + kv_size + kv_split_size * dev_rank;

  torch::Tensor q_block = in_t.narrow(1, q_start, q_split_size);
  torch::Tensor k_block = in_t.narrow(1, k_start, kv_split_size);
  torch::Tensor v_block = in_t.narrow(1, v_start, kv_split_size);
  torch::Tensor packed = torch::cat({q_block, k_block, v_block}, 1).contiguous();  // [k, n]

  if (device_model_weights.find(qkv_name) == device_model_weights.end()) {
    device_model_weights[qkv_name] =
        Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
          {static_cast<size_t>(k), static_cast<size_t>(n)});
  }

  Tensor& qkv_tensor = device_model_weights.at(qkv_name);
  const size_t copy_bytes = static_cast<size_t>(k) * static_cast<size_t>(n) * GetTypeSize(host_weight_tensor.dtype);
  MemcpyAsync(qkv_tensor.GetPtr<void>(), packed.data_ptr(), copy_bytes, MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());

  return Status();
}

Status NewInternlmxComposer2WeightLoader::ProcessInterleavedPLoraQKV(
    const std::string &host_weight_name, const Tensor &host_weight_tensor,
    std::unordered_map<std::string, Tensor> &device_model_weights, int dev_rank) {
  std::shared_ptr<InternlmxComposer2Config> config =
      std::dynamic_pointer_cast<InternlmxComposer2Config>(model_config_);

  int head_num = config->head_num;
  int hidden_units = config->hidden_units;
  int num_kv_heads = config->num_key_value_heads;
  int head_dim = hidden_units / head_num;
  int num_key_value_groups = head_num / num_kv_heads;

  // Get torch dtype
  torch::ScalarType torch_dtype;
  if (host_weight_tensor.dtype == TYPE_FP32) {
    torch_dtype = torch::kFloat32;
  } else if (host_weight_tensor.dtype == TYPE_FP16) {
    torch_dtype = torch::kFloat16;
  } else if (host_weight_tensor.dtype == TYPE_BF16) {
    torch_dtype = torch::kBFloat16;
  } else {
    KLLM_THROW(fmt::format("Unsupported weight data type: {}", static_cast<int>(host_weight_tensor.dtype)));
  }

  auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch_dtype);
  std::vector<size_t> weight_shape = host_weight_tensor.shape;

  torch::Tensor in = torch::from_blob(host_weight_tensor.GetPtr<void>(),
                                      std::vector<int64_t>(weight_shape.begin(), weight_shape.end()), options);
  in = in.to(options);

  in = in.view({num_kv_heads, 2 + num_key_value_groups, head_dim, -1});
  auto splits = torch::split(in, {num_key_value_groups, 1, 1}, 1);
  auto query_states = splits[0].contiguous().view({num_kv_heads * num_key_value_groups * head_dim, weight_shape[1]});
  auto key_states = splits[1].contiguous().view({num_kv_heads * head_dim, weight_shape[1]});
  auto value_states = splits[2].contiguous().view({num_kv_heads * head_dim, weight_shape[1]});
  in = torch::cat({query_states, key_states, value_states}, 0);

  std::string output_name = host_weight_name;

  std::vector<size_t> output_shape = {static_cast<size_t>(in.size(0)), static_cast<size_t>(in.size(1))};
  Tensor output_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, output_shape);
  device_model_weights[output_name] = output_tensor;

  size_t copy_size = in.numel() * GetTypeSize(host_weight_tensor.dtype);
  MemcpyAsync(output_tensor.GetPtr<void>(), in.data_ptr(), copy_size, MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());

  return Status();
}

Status NewInternlmxComposer2WeightLoader::ProcessPLoraWeight(
    const std::string &host_weight_name, const Tensor &host_weight_tensor,
    std::unordered_map<std::string, Tensor> &device_model_weights, int dev_rank) {
  // PLora weights are not split by TP, copy full tensor to device
  Tensor output_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape);
  device_model_weights[host_weight_name] = output_tensor;

  size_t copy_size = host_weight_tensor.GetTotalBytes();
  MemcpyAsync(output_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), copy_size, MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());

  return Status();
}

Status NewInternlmxComposer2WeightLoader::ProcessModelWeights(
    const std::unordered_map<std::string, Tensor> &host_model_weights, int dev_rank,
    std::unordered_map<std::string, Tensor> &device_model_weights,
    std::unordered_map<std::string, Tensor> &left_host_weights) {
  auto registry = weight_method_->GetRegistry()["load"];

  for (auto &[host_weight_name, host_weight_tensor] : host_model_weights) {
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}", dev_rank, host_weight_name,
                                  Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));

    if (CheckWeightNameMatched(host_weight_name, {".self_attn.wqkv.weight", ".self_attn.W_rqkv_pack.weight"},
                               MatchMode::PartialMatch)) {
      ProcessInterleavedQKV(host_weight_name, host_weight_tensor, device_model_weights, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name, {".self_attn.qkv_rproj.Plora_B.weight"}, MatchMode::PartialMatch)) {
      ProcessInterleavedPLoraQKV(host_weight_name, host_weight_tensor, device_model_weights, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name, {"Plora_A.weight", "Plora_B.weight"}, MatchMode::PartialMatch)) {
      ProcessPLoraWeight(host_weight_name, host_weight_tensor, device_model_weights, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name,
                               {".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"},
                               MatchMode::PartialMatch)) {
      registry["attn_q_k_v_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name, {".self_attn.wo.weight", ".self_attn.o_proj.weight"},
                               MatchMode::PartialMatch)) {
      registry["attn_o_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name, {".self_attn.key_layernorm", ".self_attn.query_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["attn_norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name,
                               {".mlp.w1.weight", ".mlp.w3.weight", ".mlp.gate_proj.weight", ".mlp.up_proj.weight"},
                               MatchMode::PartialMatch)) {
      registry["mlp_gate_up_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name, {".mlp.w2.weight", ".mlp.down_proj.weight"},
                               MatchMode::PartialMatch)) {
      registry["mlp_down_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name,
                               {"model.norm", ".attention_norm", ".ffn_norm", ".input_layernorm",
                                ".post_attention_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name, {"model.embed_tokens.weight", "model.tok_embeddings.weight"},
                               MatchMode::PartialMatch)) {
      registry["embed_tokens"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name, {"lm_head.weight", "output.weight"}, MatchMode::PartialMatch)) {
      registry["lm_head"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
  }
  return Status();
}

Status NewInternlmxComposer2WeightLoader::PostProcessModelWeights(
    std::unordered_map<std::string, Tensor> &dev_weights_map, int dev_rank) {
  auto registry = weight_method_->GetRegistry()["process"];

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; layer_idx++) {
    const std::string o_proj_name = fmt::format("model.layers.{}.self_attn.o_proj.", layer_idx);
    registry["attn_o_proj"].Run<Status>(dev_weights_map, o_proj_name, dev_rank);

    const std::string gate_proj_weight_name = fmt::format("model.layers.{}.mlp.gate_proj.weight", layer_idx);
    const std::string up_proj_weight_name = fmt::format("model.layers.{}.mlp.up_proj.weight", layer_idx);
    const std::string down_proj_weight_name = fmt::format("model.layers.{}.mlp.down_proj.weight", layer_idx);

    if (dev_weights_map.find(gate_proj_weight_name) != dev_weights_map.end()) {
      common_weight_loader_->Permute2D(dev_weights_map.at(gate_proj_weight_name), dev_rank);
    }
    if (dev_weights_map.find(up_proj_weight_name) != dev_weights_map.end()) {
      common_weight_loader_->Permute2D(dev_weights_map.at(up_proj_weight_name), dev_rank);
    }
    if (dev_weights_map.find(down_proj_weight_name) != dev_weights_map.end()) {
      common_weight_loader_->Permute2D(dev_weights_map.at(down_proj_weight_name), dev_rank);
    }
  }

  const std::string lm_head_name = "lm_head.";
  registry["lm_head"].Run<Status>(dev_weights_map, lm_head_name, dev_rank);

  return Status();
}

}  // namespace ksana_llm

