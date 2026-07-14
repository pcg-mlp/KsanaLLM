/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/chatglm/chatglm_model_weight_loader.h"
#include <string>
#include <unordered_set>

#include "configure/model_config.h"
#include "device/device_types.h"
#include "kernels/trans_layout.h"
#include "model_loader/model_loader_utils.h"
#include "models/base/model_arch.h"
#include "models/base/model_format.h"
#include "models/chatglm/chatglm_config.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/ret_code.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

ChatglmModelWeightLoader::ChatglmModelWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                                   std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_shared<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;
  weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
}

ChatglmModelWeightLoader::~ChatglmModelWeightLoader() {}

Status ChatglmModelWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq", "rotary_pos_emb.inv_freq"};
  std::vector<std::string> master_only_list = {"model.embed_tokens.weight",
                                               "transformer.embedding.word_embeddings.weight"};
  std::vector<std::string> tail_only_list = {"lm_head.weight", "transformer.output_layer.weight"};

  int lower_layer_idx = pipeline_config_.lower_layer_idx;
  int upper_layer_idx = pipeline_config_.upper_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, false)) {
      weight_names.erase(it);
      continue;
    }

    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      int layer_idx = GetLayerIdxFromName(*it);
      if (layer_idx >= 0 && (layer_idx < lower_layer_idx || layer_idx > upper_layer_idx)) {
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

Status ChatglmModelWeightLoader::PreProcessModelWeights(
    const std::unordered_map<std::string, Tensor>& host_model_weights) {
#define REGISTER_COMMON(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetCommonMethod().get(), \
                                                        &CommonMethod::key##_##weight_name)
#define REGISTER_CUSTOM(key, weight_name) \
  weight_method_->GetRegistry()[#key][#weight_name].Add(this, &ChatglmModelWeightLoader::key##_##weight_name)

  // Register load methods
  REGISTER_CUSTOM(load, chatglm_qkv_weight);
  REGISTER_CUSTOM(load, chatglm_qkv_bias);
  REGISTER_CUSTOM(load, chatglm_gate_up);
  REGISTER_COMMON(load, attn_o_proj);
  REGISTER_COMMON(load, mlp_down_proj);
  REGISTER_COMMON(load, norm);
  REGISTER_COMMON(load, attn_norm);
  REGISTER_COMMON(load, embed_tokens);
  REGISTER_COMMON(load, lm_head);

  // Register process methods
  REGISTER_CUSTOM(process, chatglm_qkv);
  REGISTER_COMMON(process, attn_o_proj);
  REGISTER_COMMON(process, mlp_down_proj);
  REGISTER_COMMON(process, lm_head);

#undef REGISTER_COMMON
#undef REGISTER_CUSTOM
  return Status();
}

Status ChatglmModelWeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                                     int dev_rank,
                                                     std::unordered_map<std::string, Tensor>& device_model_weights,
                                                     std::unordered_map<std::string, Tensor>& left_host_weights) {
  if (model_config_->model_format == ModelFormat::GGUF) {
    return Status(RET_INVALID_ARGUMENT, "Not support GGUF format yet.");
  }

  ModelConfig model_config;
  STATUS_CHECK_RETURN(env_->GetModelConfig(model_config));
  DataType weight_data_type = model_config.weight_data_type;

  auto registry = weight_method_->GetRegistry()["load"];

  for (auto& [file_weight_name, host_weight_tensor] : host_model_weights) {
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}", dev_rank, file_weight_name,
                                  Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));

    // ChatGLM-specific: QKV weight (query_key_value)
    if (file_weight_name.find("self_attn.query_key_value.weight") != std::string::npos) {
      registry["chatglm_qkv_weight"].Run<Status>(device_model_weights, file_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // ChatGLM-specific: QKV bias
    if (file_weight_name.find("self_attn.query_key_value.bias") != std::string::npos) {
      registry["chatglm_qkv_bias"].Run<Status>(device_model_weights, file_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // ChatGLM-specific: fused gate_up (dense_h_to_4h)
    if (file_weight_name.find("mlp.gate_proj.weight") != std::string::npos) {
      registry["chatglm_gate_up"].Run<Status>(device_model_weights, file_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (file_weight_name.find("self_attn.o_proj.weight") != std::string::npos) {
      registry["attn_o_proj"].Run<Status>(device_model_weights, file_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (file_weight_name.find("mlp.down_proj.weight") != std::string::npos) {
      registry["mlp_down_proj"].Run<Status>(device_model_weights, file_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(file_weight_name,
                               {"input_layernorm.weight", "post_attention_layernorm.weight", "final_layernorm.weight",
                                "model.norm.weight"},
                               MatchMode::PartialMatch)) {
      registry["norm"].Run<Status>(device_model_weights, file_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (file_weight_name.find("attn_norm") != std::string::npos) {
      registry["attn_norm"].Run<Status>(device_model_weights, file_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (file_weight_name.find("model.embed_tokens.weight") != std::string::npos) {
      registry["embed_tokens"].Run<Status>(device_model_weights, file_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (file_weight_name.find("lm_head.weight") != std::string::npos) {
      registry["lm_head"].Run<Status>(device_model_weights, file_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // Handle bias weights that don't need TP split
    if (file_weight_name.find("self_attn.o_proj.bias") != std::string::npos ||
        file_weight_name.find("mlp.down_proj.bias") != std::string::npos) {
      Tensor dev_tensor;
      STATUS_CHECK_RETURN(CopyHostTensorToDevice(host_weight_tensor, dev_rank, dev_tensor));
      STATUS_CHECK_RETURN(CastDeviceTensorType(dev_tensor, weight_data_type, dev_rank));
      StreamSynchronize(context_->device->GetMemoryManageStream());
      device_model_weights[file_weight_name] = dev_tensor;
      continue;
    }

    // Handle bias weights that need TP split (RowPara)
    if (file_weight_name.find("mlp.gate_proj_bias") != std::string::npos ||
        file_weight_name.find("mlp.up_proj_bias") != std::string::npos) {
      Tensor dev_tensor = common_weight_loader_->TensorParallelSplit(host_weight_tensor, dev_rank, tp_, RowPara);
      STATUS_CHECK_RETURN(CastDeviceTensorType(dev_tensor, weight_data_type, dev_rank));
      StreamSynchronize(context_->device->GetMemoryManageStream());
      device_model_weights[file_weight_name] = dev_tensor;
      continue;
    }

    KLLM_LOG_WARNING << "Unhandled weight: " << file_weight_name << ", skipping.";
  }

  return Status();
}

Status ChatglmModelWeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                         int dev_rank) {
  auto registry = weight_method_->GetRegistry()["process"];

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; layer_idx++) {
    // Process QKV weight (transpose)
    const std::string qkv_name = fmt::format("model.layers.{}.self_attn.query_key_value.", layer_idx);
    STATUS_CHECK_RETURN(registry["chatglm_qkv"].Run<Status>(dev_weights_map, qkv_name, dev_rank));

    // Process o_proj weight (transpose)
    const std::string o_proj_name = fmt::format("model.layers.{}.self_attn.o_proj.", layer_idx);
    STATUS_CHECK_RETURN(registry["attn_o_proj"].Run<Status>(dev_weights_map, o_proj_name, dev_rank));

    // Process down_proj weight (transpose)
    const std::string down_proj_name = fmt::format("model.layers.{}.mlp.down_proj.", layer_idx);
    STATUS_CHECK_RETURN(registry["mlp_down_proj"].Run<Status>(dev_weights_map, down_proj_name, dev_rank));
  }

  // Process lm_head (transpose)
  const std::string lm_head_name = "lm_head.";
  STATUS_CHECK_RETURN(registry["lm_head"].Run<Status>(dev_weights_map, lm_head_name, dev_rank));

  return Status();
}

Status ChatglmModelWeightLoader::load_chatglm_qkv_weight(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                         const std::string& weight_name, const Tensor& weight_tensor,
                                                         int dev_rank) {
  ModelConfig model_config;
  STATUS_CHECK_RETURN(env_->GetModelConfig(model_config));

  size_t head_num = model_config.head_num;
  size_t num_kv_heads = model_config.num_key_value_heads;
  size_t size_per_head = model_config.size_per_head;

  // Calculate dimensions (K and V have same dimension in GQA/MQA)
  size_t q_dim_full = head_num * size_per_head;
  size_t kv_dim_full = num_kv_heads * size_per_head;
  size_t q_dim_rank = q_dim_full / tp_;
  size_t kv_dim_rank = kv_dim_full / tp_;

  size_t qkv_dim = q_dim_rank + 2 * kv_dim_rank;
  size_t hidden_size = weight_tensor.shape[1];

  Tensor qkv_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, weight_tensor.dtype, {qkv_dim, hidden_size});

  size_t type_size = GetTypeSize(weight_tensor.dtype);
  size_t row_bytes = hidden_size * type_size;

  // Get TP rank within the TP group (dev_rank may be global device ID in PP scenarios)
  size_t tp_rank = dev_rank % tp_;

  // Source offsets: layout is [Q_all, K_all, V_all]
  size_t q_src_offset = tp_rank * q_dim_rank * row_bytes;
  size_t k_src_offset = (q_dim_full + tp_rank * kv_dim_rank) * row_bytes;
  size_t v_src_offset = k_src_offset + kv_dim_full * row_bytes;

  // Destination offsets
  size_t q_dst_offset = 0;
  size_t k_dst_offset = q_dim_rank * row_bytes;
  size_t v_dst_offset = (q_dim_rank + kv_dim_rank) * row_bytes;

  // Copy Q, K, V parts (all on same stream, no intermediate sync needed)
  Memcpy2DAsync(qkv_tensor.GetPtr<void>() + q_dst_offset, row_bytes, weight_tensor.GetPtr<void>() + q_src_offset,
                row_bytes, row_bytes, q_dim_rank, MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

  Memcpy2DAsync(qkv_tensor.GetPtr<void>() + k_dst_offset, row_bytes, weight_tensor.GetPtr<void>() + k_src_offset,
                row_bytes, row_bytes, kv_dim_rank, MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

  Memcpy2DAsync(qkv_tensor.GetPtr<void>() + v_dst_offset, row_bytes, weight_tensor.GetPtr<void>() + v_src_offset,
                row_bytes, row_bytes, kv_dim_rank, MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

  StreamSynchronize(context_->device->GetMemoryManageStream());

  KLLM_LOG_DEBUG << "Stored QKV weight: " << weight_name << ", shape: [" << qkv_dim << ", " << hidden_size << "]";
  device_model_weights[weight_name] = qkv_tensor;
  return Status();
}

Status ChatglmModelWeightLoader::load_chatglm_qkv_bias(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                       const std::string& weight_name, const Tensor& weight_tensor,
                                                       int dev_rank) {
  ModelConfig model_config;
  STATUS_CHECK_RETURN(env_->GetModelConfig(model_config));

  size_t head_num = model_config.head_num;
  size_t num_kv_heads = model_config.num_key_value_heads;
  size_t size_per_head = model_config.size_per_head;

  // Calculate dimensions (K and V have same dimension in GQA/MQA)
  size_t q_dim_full = head_num * size_per_head;
  size_t kv_dim_full = num_kv_heads * size_per_head;
  size_t q_dim_rank = q_dim_full / tp_;
  size_t kv_dim_rank = kv_dim_full / tp_;

  size_t qkv_dim = q_dim_rank + 2 * kv_dim_rank;
  size_t type_size = GetTypeSize(weight_tensor.dtype);

  Tensor bias_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, weight_tensor.dtype, {qkv_dim});

  // Get TP rank within the TP group (dev_rank may be global device ID in PP scenarios)
  size_t tp_rank = dev_rank % tp_;

  // Source offsets: layout is [Q_all, K_all, V_all]
  size_t q_src_offset = tp_rank * q_dim_rank * type_size;
  size_t k_src_offset = (q_dim_full + tp_rank * kv_dim_rank) * type_size;
  size_t v_src_offset = k_src_offset + kv_dim_full * type_size;

  // Destination offsets
  size_t q_dst_offset = 0;
  size_t k_dst_offset = q_dim_rank * type_size;
  size_t v_dst_offset = (q_dim_rank + kv_dim_rank) * type_size;

  // All on same stream, no intermediate sync needed
  MemcpyAsync(bias_tensor.GetPtr<void>() + q_dst_offset, weight_tensor.GetPtr<void>() + q_src_offset,
              q_dim_rank * type_size, MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

  MemcpyAsync(bias_tensor.GetPtr<void>() + k_dst_offset, weight_tensor.GetPtr<void>() + k_src_offset,
              kv_dim_rank * type_size, MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

  MemcpyAsync(bias_tensor.GetPtr<void>() + v_dst_offset, weight_tensor.GetPtr<void>() + v_src_offset,
              kv_dim_rank * type_size, MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

  StreamSynchronize(context_->device->GetMemoryManageStream());

  KLLM_LOG_DEBUG << "Stored QKV bias: " << weight_name;
  device_model_weights[weight_name] = bias_tensor;
  return Status();
}

Status ChatglmModelWeightLoader::load_chatglm_gate_up(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                      const std::string& weight_name, const Tensor& weight_tensor,
                                                      int dev_rank) {
  // Get TP rank within the TP group (dev_rank may be global device ID in PP scenarios)
  size_t tp_rank = dev_rank % tp_;

  // Original shape: [inter_size*2, hidden_size]
  size_t fused_inter_size = weight_tensor.shape[0];
  size_t inter_size = fused_inter_size / 2;
  size_t hidden_size = weight_tensor.shape[1];

  // Copy to device and transpose to [hidden_size, inter_size*2]
  // All operations on same stream - no intermediate sync needed
  Tensor full_dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, weight_tensor.dtype, weight_tensor.shape);
  MemcpyAsync(full_dev_tensor.GetPtr<void>(), weight_tensor.GetPtr<void>(), weight_tensor.GetTotalBytes(),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

  std::vector<size_t> permuted_shape = {hidden_size, fused_inter_size};
  Tensor permute_dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, weight_tensor.dtype, permuted_shape);
  PermuteDeviceTensor(full_dev_tensor, {1, 0}, dev_rank, permute_dev_tensor);

  // TP split gate and up separately
  size_t type_size = GetTypeSize(weight_tensor.dtype);
  size_t tp_slice_size = DivRoundUp(inter_size, tp_);
  std::vector<size_t> final_shape = {hidden_size, tp_slice_size};

  Tensor gate_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, weight_tensor.dtype, final_shape);
  Tensor up_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, weight_tensor.dtype, final_shape);

  size_t src_pitch = fused_inter_size * type_size;
  size_t dst_pitch = tp_slice_size * type_size;
  size_t gate_col_offset = tp_rank * tp_slice_size * type_size;
  size_t up_col_offset = inter_size * type_size + tp_rank * tp_slice_size * type_size;

  size_t actual_slice_width = tp_slice_size * type_size;
  if (tp_rank == tp_ - 1) {
    actual_slice_width = inter_size * type_size - tp_rank * tp_slice_size * type_size;
  }

  Memcpy2DAsync(gate_tensor.GetPtr<void>(), dst_pitch, permute_dev_tensor.GetPtr<void>() + gate_col_offset, src_pitch,
                actual_slice_width, hidden_size, MEMCPY_DEVICE_TO_DEVICE, context_->device->GetMemoryManageStream());

  Memcpy2DAsync(up_tensor.GetPtr<void>(), dst_pitch, permute_dev_tensor.GetPtr<void>() + up_col_offset, src_pitch,
                actual_slice_width, hidden_size, MEMCPY_DEVICE_TO_DEVICE, context_->device->GetMemoryManageStream());

  StreamSynchronize(context_->device->GetMemoryManageStream());

  device_model_weights[weight_name] = gate_tensor;

  const std::string gate_proj_str = "gate_proj";
  std::string up_proj_name = weight_name;
  size_t gate_pos = up_proj_name.find(gate_proj_str);
  KLLM_CHECK_WITH_INFO(gate_pos != std::string::npos,
                        fmt::format("Weight name '{}' does not contain 'gate_proj'", weight_name));
  up_proj_name.replace(gate_pos, gate_proj_str.length(), "up_proj");
  device_model_weights[up_proj_name] = up_tensor;

  KLLM_LOG_DEBUG << "Split fused gate_up: " << weight_name << " and " << up_proj_name << ", shape: [" << hidden_size
                 << ", " << tp_slice_size << "]";
  return Status();
}

Status ChatglmModelWeightLoader::process_chatglm_qkv(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                     const std::string& weight_prefix_name, int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
  auto itr = device_model_weights.find(weight_name);
  if (itr == device_model_weights.end()) {
    return Status();
  }

  Tensor& weight_tensor = itr->second;
  STATUS_CHECK_RETURN(common_weight_loader_->Permute2D(weight_tensor, dev_rank));
  return Status();
}

}  // namespace ksana_llm
