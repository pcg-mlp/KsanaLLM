/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/common/common_weight.h"

#include <regex>
#include "nlohmann/json.hpp"

#include "ksana_llm/utils/common_device.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/kernels/permute.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/optional_file.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/nn/functional/normalization.h>

namespace ksana_llm {

// weight_name 与存储的模型文件的关联映射关系
// gather_embedding                   : model.wte.weight.bin
// input_layernorm          + layer x : model.layers.x.input_layernorm.weight.bin
// post_attention_layernorm + layer x : model.layers.x.post_attention_layernorm.weight.bin
// attention.dense + rank r + layer x : model.layers.x.attention.dense.weight.r.bin
// attention.query_key_value + r +  x : model.layers.x.attention.query_key_value.weight.r.bin
// mlp.gate_proj +  rank r  + layer x : model.layers.x.mlp.gate_proj.weight.r.bin
// mlp.up_proj   +  rank r  + layer x : model.layers.x.mlp.up_proj.weight.r.bin
// mlp.down_proj +  rank r  + layer x : model.layers.x.mlp.down_proj.weight.r.bin
// norm                               : model.final_layernorm.weight
// lm_head                            : model.lm_head.weight
template <typename T>
CommonWeight<T>::~CommonWeight() {
  GetBlockManager()->SetDeviceId(rank_);
  for (auto& [key, tensor] : weights_map_) {
    const int block_id = tensor.GetBlockId();
    GetBlockManager()->FreeContiguous(block_id);
  }
}

template <typename T>
CommonWeight<T>::CommonWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : context_(context), model_config_(model_config) {
  model_path_ = model_config.path;
  rank_ = rank;
  if (!GetModelInfo(model_config).OK()) {
    NLLM_LOG_ERROR << fmt::format("Load model config file error.");
    exit(-1);
  }
}

int CheckQKVWeight(const std::string& str, const int head_num, const int num_kv_heads) {
  std::string suffix = "_proj.weight";
  if (str.find("_proj.bias") != std::string::npos) {
    suffix = "_proj.bias";
  }
  if (str.length() < suffix.length() + 1 || str.compare(str.length() - suffix.length(), suffix.length(), suffix)) {
    return -1;
  }
  std::vector<char> qkv_list = {'q', 'k', 'v'};
  std::vector<int> qkv_offset = {0, head_num / num_kv_heads, head_num / num_kv_heads + 1};
  for (int i = 0; i < 3; ++i) {
    if (str[str.length() - suffix.length() - 1] == qkv_list[i]) {
      return qkv_offset[i];
    }
  }
  return -1;
}

template <typename T>
Status CommonWeight<T>::GetCustomNameList(std::vector<std::string>& weight_name_list,
                                          std::vector<std::string>& custom_name_list) {
  // In the default case, the tensor name is consistent with the weight name.
  custom_name_list.assign(weight_name_list.begin(), weight_name_list.end());

  // Search for the optional_weight_map.json file
  auto optional_file = Singleton<OptionalFile>::GetInstance();
  std::string& weight_path =
      optional_file->GetOptionalFile(model_config_.path, "weight_map", model_config_.type + "_weight_map.json");
  if (weight_path == "") {
    return Status();
  }

  nlohmann::json weight_map_json;
  std::ifstream file(weight_path);
  if (!file.is_open()) {
    NLLM_LOG_ERROR << fmt::format("Load weight map json: {} error.", weight_path) << std::endl;
    return Status(RetCode::RET_INVALID_ARGUMENT, fmt::format("Load weight map json: {} error.", weight_path));
  } else {
    file >> weight_map_json;
    file.close();
  }
  for (size_t idx = 0; idx < weight_name_list.size(); ++idx) {
    std::string weight_name = weight_name_list[idx];
    for (auto it = weight_map_json.begin(); it != weight_map_json.end(); ++it) {
      std::regex pattern(it.key());
      std::string format = it.value();
      if (std::regex_search(weight_name, pattern)) {
        custom_name_list[idx] = std::regex_replace(weight_name, pattern, format);
        break;
      }
    }
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::PrepareLoadOpMeta(size_t& tensor_para_offset, std::vector<size_t>& weight_shape,
                                          bool& transpose_first, const std::string& tensor_name) {
  // EmbedTokensUseCpu does not require slicing
  if (Singleton<Environment>::GetInstance()->EmbedTokensUseCpu() &&
      tensor_name.find("embed_tokens") != std::string::npos) {
    return Status();
  }
  if (tensor_name.find("_proj.weight") != std::string::npos || tensor_name.find(".bias") != std::string::npos ||
      tensor_name.find("self_attn.W_pack") != std::string::npos ||
      tensor_name.find("embed_tokens") != std::string::npos ||
      tensor_name.find("lm_head.weight") != std::string::npos) {
    tensor_para_offset = rank_;
    if (tensor_name.find("o_proj") != std::string::npos || tensor_name.find("down_proj") != std::string::npos ||
        tensor_name.find("embed_tokens") != std::string::npos) {
      transpose_first = true;
      weight_shape[1] /= tensor_para_size_;
    } else {
      weight_shape[0] = DivRoundUp(weight_shape[0], tensor_para_size_);
    }
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader) {
  GetBlockManager()->SetDeviceId(rank_);
  std::vector<std::string> weight_name_list = weights_loader->GetTensorNameList();
  std::vector<std::string> custom_name_list;
  STATUS_CHECK_RETURN(GetCustomNameList(weight_name_list, custom_name_list));
  for (size_t idx = 0; idx < weight_name_list.size(); ++idx) {
    // tensor_para_offset 用于标记读取 weights_data 时是否做分卡处理:
    //     input_layernorm:          不做分卡处理
    //     post_attention_layernorm: 不做分卡处理
    //     self_attn.o_proj:         先转置,再按 axis=0 切分
    //     self_attn.qkv_proj:       先按 axis=0 切分, 再 permute((2, 0, 1))
    //     mlp.down_proj:            先转置,再按 axis=0 切分
    //     mlp.up_proj:              先按 axis=0 切分, 再转置
    //     mlp.gate_proj:            先按 axis=0 切分, 再转置
    //     lm_head:                  不做分卡处理, 需转置
    //     norm:                     不做分卡处理
    //     embedding:                不做分卡处理
    std::string& tensor_name = custom_name_list[idx];
    std::string& weight_name = weight_name_list[idx];
    bool transpose_first = false;  // 使用 transpose_first 表明转置(若存在)是否在分卡(若存在)之前
    size_t tensor_para_offset = 0;
    std::vector<size_t> weight_shape = weights_loader->GetTensorShape(weight_name);
    STATUS_CHECK_RETURN(PrepareLoadOpMeta(tensor_para_offset, weight_shape, transpose_first, tensor_name));

    // get weight's data ptr
    void* weight_ptr;
    size_t weight_size;
    std::tie(weight_ptr, weight_size) = weights_loader->GetTensor(weight_name);
    DataType weight_data_type = weights_loader->GetTensorDataType(weight_name);

    torch::Tensor weight_cpu_tensor;
    if (weight_data_type == TYPE_FP32) {
      // cast TYPE_FP32 to weight_data_type_.
      auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
      torch::Tensor in = torch::from_blob(weight_ptr, {(int64_t)(weight_size / sizeof(float))}, options);
      weight_size /= sizeof(float) / GetTypeSize(weight_data_type_);
      if (weight_data_type_ == TYPE_FP16) {
        weight_cpu_tensor = in.to(torch::kFloat16);
        weight_ptr = weight_cpu_tensor.data_ptr();
      } else if (weight_data_type_ == TYPE_BF16) {
        weight_cpu_tensor = in.to(torch::kBFloat16);
        weight_ptr = weight_cpu_tensor.data_ptr();
      } else {
        NLLM_LOG_WARNING << "Weight " << tensor_name << " data type " << weight_data_type << " can't cast to type "
                         << weight_data_type_;
      }
    } else if (weight_data_type != TYPE_FP16 && weight_data_type != TYPE_BF16) {
      NLLM_LOG_WARNING << "Weight " << tensor_name << " data type is " << weight_data_type;
    }

    int head_num = model_config_.head_num;
    int num_kv_heads = model_config_.num_key_value_heads;
    // copy host data to device
    int qkv_offset = CheckQKVWeight(tensor_name, head_num, num_kv_heads);
    if (qkv_offset >= 0) {
      bool is_bias = (tensor_name.find("_proj.bias") != std::string::npos);
      std::string qkv_name = tensor_name.substr(0, tensor_name.find_last_of('_') - 1) + "query_key_value" +
                             (is_bias ? ".bias" : ".weight");
      if (!weights_map_.count(qkv_name)) {
        if (qkv_offset == 0) {
          // For q_proj in the GQA scenario, the weight_shape is first transformed into k_proj.
          weight_shape[0] /= head_num / num_kv_heads;
        }
        weight_shape.insert(weight_shape.begin(), ((head_num / num_kv_heads) + 2));
        if (is_bias) {
          // The Add-Bias-Residual Kernel uses the shape[0] of the input tensor to determine whether
          // broadcasting is required.
          weight_shape.insert(weight_shape.begin(), 1);
        }
        AddWeightTensor(qkv_name, weight_shape, weight_data_type_);
      }
      weights_data_type_map_[qkv_name] = weight_data_type;
      Tensor& qkv_weight_tensor = weights_map_[qkv_name];
      size_t single_proj_size = qkv_weight_tensor.GetTotalBytes() / (head_num / num_kv_heads + 2);
      size_t saved_offset = qkv_offset * single_proj_size;
      if (qkv_offset == 0) {
        single_proj_size *= head_num / num_kv_heads;
      }
      tensor_para_offset *= single_proj_size;
      MemcpyAsync(qkv_weight_tensor.GetPtr<void>() + saved_offset, weight_ptr + tensor_para_offset, single_proj_size,
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    } else if (tensor_name.find("_proj.weight") != std::string::npos ||
               tensor_name.find("_proj.bias") != std::string::npos ||
               tensor_name.find("_layernorm.weight") != std::string::npos || tensor_name == "model.norm.weight" ||
               (tensor_name == "lm_head.weight" && !model_config_.tie_word_embeddings)) {
      LoadRegularTensor(weight_ptr, tensor_name, weight_shape, weight_data_type, transpose_first, tensor_para_offset,
                        weight_size);
    } else if (tensor_name == "model.embed_tokens.weight") {
      LoadRegularTensor(weight_ptr, tensor_name, weight_shape, weight_data_type, transpose_first, tensor_para_offset,
                        weight_size);
      if (model_config_.tie_word_embeddings) {
        /* When the "tie-word-embeddings" is set to True in the model's config.json, the model's
         * "model.embed_tokens.weight" and "lm_head.weight" share the same data space. Therefore, it is necessary
         * to load the data from "weight_ptr" twice and store it in the corresponding device spaces of the two weights.
         */
        NLLM_LOG_DEBUG << "tie_word_embeddings = true, lm_head.weight = model.embed_tokens.weight";
        std::vector<size_t> embed_tokens_shape = weights_map_["model.embed_tokens.weight"].shape;
        std::vector<size_t> lm_head_shape = {(size_t)(DivRoundUp(embed_tokens_shape[0], tensor_para_size_)),
                                             embed_tokens_shape[1] * tensor_para_size_};
        LoadRegularTensor(weight_ptr, "lm_head.weight", lm_head_shape, weight_data_type, /*transpose_first*/ false,
                          tensor_para_offset, weight_size);
      }
    } else if (tensor_name.find("self_attn.W_pack.weight") != std::string::npos) {
      std::string qkv_name = tensor_name.substr(0, tensor_name.find_last_of('_') - 1) + "query_key_value.weight";
      weights_data_type_map_[qkv_name] = weight_data_type;
      if (!weights_map_.count(qkv_name)) {
        weight_shape.insert(weight_shape.begin(), 3);
        weight_shape[1] /= 3;
        AddWeightTensor(qkv_name, weight_shape, weight_data_type_);
      }
      Tensor& qkv_weight_tensor = weights_map_[qkv_name];
      size_t src_pitch = weight_shape[1] * weight_shape[2] * tensor_para_size_ * sizeof(T);
      size_t dst_pitch = weight_shape[1] * weight_shape[2] * sizeof(T);
      tensor_para_offset *= dst_pitch;
      Memcpy2DAsync(qkv_weight_tensor.GetPtr<void>(), dst_pitch, weight_ptr + tensor_para_offset, src_pitch, dst_pitch,
                    weight_shape[0], MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    } else if (tensor_name.find("query_key_value.bias") != std::string::npos) {
      weights_data_type_map_[tensor_name] = weight_data_type;
      AddWeightTensor(tensor_name, weight_shape, weight_data_type_);
      Tensor& qkv_bias_tensor = weights_map_[tensor_name];
      size_t src_pitch = weight_shape[0] / 3 * tensor_para_size_ * sizeof(T);
      size_t dst_pitch = weight_shape[0] / 3 * sizeof(T);
      tensor_para_offset *= dst_pitch;
      Memcpy2DAsync(qkv_bias_tensor.GetPtr<void>(), dst_pitch, weight_ptr + tensor_para_offset, src_pitch, dst_pitch, 3,
                    MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    } else {
      NLLM_LOG_DEBUG << "state_dict[" << tensor_name << "] will not be used";
    }
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::PermuteSingleTensorOfQKVWeight(void* qkv_src, void* qkv_dst, Tensor& q_in_tensor,
                                                       Tensor& q_out_tensor, std::vector<size_t>& data_shape,
                                                       std::vector<size_t>& qkv_dst_shape) {
  q_in_tensor.shape = data_shape;
  q_out_tensor.shape = data_shape;
  MemcpyAsync(q_in_tensor.GetPtr<void>(), qkv_src, q_in_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
              context_->GetMemoryManageStreams()[rank_]);
  Permute(q_in_tensor, q_out_tensor, {2, 0, 1}, context_->GetMemoryManageStreams()[rank_]);

#ifdef ENALBE_CUDA
  Memcpy2DAsync(qkv_dst, qkv_dst_shape[1] * sizeof(T), q_out_tensor.GetPtr<void>(), data_shape[1] * sizeof(T),
                data_shape[1] * sizeof(T), data_shape[2], MEMCPY_DEVICE_TO_DEVICE,
                context_->GetMemoryManageStreams()[rank_]);
#else
  // NOTE(karlluo): for ascend, there is a issue that can not use Memcpy2DAsync.
  // will fix it when it work.
  for (size_t row_idx = 0; row_idx < data_shape[2]; ++row_idx) {
    MemcpyAsync(qkv_dst + row_idx * qkv_dst_shape[1] * sizeof(T),
                q_out_tensor.GetPtr<void>() + row_idx * data_shape[1] * sizeof(T), data_shape[1] * sizeof(T),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
  }
#endif

  return Status();
}

template <typename T>
Status CommonWeight<T>::PermuteQKVWeight(Tensor& last_qkv_tensor, Tensor& q_in_tensor, Tensor& q_out_tensor,
                                         const int num_layer) {
  GetBlockManager()->SetDeviceId(rank_);

  // src tensor: qkv_weight_tensor[head_num / num_kv_heads + 2, d1, d2]
  // after split: q[head_num / num_kv_heads, d1, d2], k[1, d1, d2], v[1, d1, d2]
  // after permute: q[d2, head_num / num_kv_heads, d1], k[d2, 1, d1], v[d2, 1, d1]
  // dst tensor: last_kv_tensor[d2, head_num / num_kv_heads * d1 + d1 + d1]
  int head_num = model_config_.head_num;
  int num_kv_heads = model_config_.num_key_value_heads;
  for (size_t layer_idx = 0; layer_idx < (size_t)num_layer; ++layer_idx) {
    std::string qkv_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.query_key_value.weight";
    Tensor& qkv_weight_tensor = weights_map_[qkv_name];
    auto qkv_shape = qkv_weight_tensor.shape;
    std::vector<size_t> q_shape = {1, head_num / num_kv_heads * qkv_shape[1], qkv_shape[2]};
    std::vector<size_t> kv_shape = {1, qkv_shape[1], qkv_shape[2]};
    size_t q_size = q_shape[1] * q_shape[2] * sizeof(T);
    size_t kv_size = kv_shape[1] * kv_shape[2] * sizeof(T);
    std::vector<size_t> qkv_dst_shape = {qkv_shape[2], qkv_shape[0] * qkv_shape[1]};
    last_qkv_tensor.shape = qkv_dst_shape;

    void* qkv_src = qkv_weight_tensor.GetPtr<void>();
    void* qkv_dst = last_qkv_tensor.GetPtr<void>();
    PermuteSingleTensorOfQKVWeight(qkv_src, qkv_dst, q_in_tensor, q_out_tensor, q_shape, qkv_dst_shape);

    qkv_src = qkv_src + q_size;
    qkv_dst = qkv_dst + q_shape[1] * sizeof(T);
    PermuteSingleTensorOfQKVWeight(qkv_src, qkv_dst, q_in_tensor, q_out_tensor, kv_shape, qkv_dst_shape);

    qkv_src = qkv_src + kv_size;
    qkv_dst = qkv_dst + kv_shape[1] * sizeof(T);
    PermuteSingleTensorOfQKVWeight(qkv_src, qkv_dst, q_in_tensor, q_out_tensor, kv_shape, qkv_dst_shape);

    Tensor t = last_qkv_tensor;
    last_qkv_tensor = qkv_weight_tensor;
    weights_map_[qkv_name] = t;
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::PermuteMLPWeight(Tensor& last_mlp_tensor, const int num_layer) {
  GetBlockManager()->SetDeviceId(rank_);
  for (size_t layer_idx = 0; layer_idx < (size_t)num_layer; ++layer_idx) {
    std::string down_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight";
    Tensor& down_weight_tensor = weights_map_[down_proj_name];
    Permute(down_weight_tensor, last_mlp_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
    Tensor t = last_mlp_tensor;
    last_mlp_tensor = down_weight_tensor;
    t.shape = {down_weight_tensor.shape[1], down_weight_tensor.shape[0]};
    weights_map_[down_proj_name] = t;

    std::string gate_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight";
    Tensor& gate_weight_tensor = weights_map_[gate_proj_name];
    Permute(gate_weight_tensor, last_mlp_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
    t = last_mlp_tensor;
    last_mlp_tensor = gate_weight_tensor;
    t.shape = {gate_weight_tensor.shape[1], gate_weight_tensor.shape[0]};
    weights_map_[gate_proj_name] = t;

    std::string up_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight";
    Tensor& up_weight_tensor = weights_map_[up_proj_name];
    Permute(up_weight_tensor, last_mlp_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
    t = last_mlp_tensor;
    last_mlp_tensor = up_weight_tensor;
    t.shape = {up_weight_tensor.shape[1], up_weight_tensor.shape[0]};
    weights_map_[up_proj_name] = t;
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::PermuteOutputProjectWeight(Tensor& last_o_proj_tensor, const int num_layer) {
  GetBlockManager()->SetDeviceId(rank_);
  for (size_t layer_idx = 0; layer_idx < (size_t)num_layer; ++layer_idx) {
    std::string o_proj_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight";
    Tensor& o_proj_weight_tensor = weights_map_[o_proj_name];
    Permute(o_proj_weight_tensor, last_o_proj_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
    Tensor t = last_o_proj_tensor;
    last_o_proj_tensor = o_proj_weight_tensor;
    t.shape = {o_proj_weight_tensor.shape[1], o_proj_weight_tensor.shape[0]};
    weights_map_[o_proj_name] = t;
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::LoadRegularTensor(void* weight_ptr, std::string tensor_name, std::vector<size_t>& weight_shape,
                                          DataType& weight_data_type, bool transpose_first, size_t tensor_para_offset,
                                          size_t& weight_size) {
  AddWeightTensor(tensor_name, weight_shape, weight_data_type_);
  weights_data_type_map_[tensor_name] = weight_data_type;
  if (transpose_first) {
    size_t src_pitch = weights_map_[tensor_name].shape[1] * tensor_para_size_ * sizeof(T);
    size_t dst_pitch = weights_map_[tensor_name].shape[1] * sizeof(T);
    tensor_para_offset *= dst_pitch;
    Memcpy2DAsync(weights_map_[tensor_name].GetPtr<void>(), dst_pitch, weight_ptr + tensor_para_offset, src_pitch,
                  dst_pitch, weights_map_[tensor_name].shape[0], MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);
  } else {
    tensor_para_offset *= weights_map_[tensor_name].GetTotalBytes();
    GetBlockManager()->SetDeviceId(rank_);
    size_t sub_bytes = 0;
    if (rank_ == (tensor_para_size_ - 1) && tensor_name == "lm_head.weight") {
      sub_bytes = weights_map_[tensor_name].GetTotalBytes() * tensor_para_size_ - weight_size;
    }
    MemcpyAsync(weights_map_[tensor_name].GetPtr<void>(), weight_ptr + tensor_para_offset,
                weights_map_[tensor_name].GetTotalBytes() - sub_bytes, MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[rank_]);
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::PermuteTensor(int hidden_units, int inter_size, int num_layer, int vocab_size) {
  GetBlockManager()->SetDeviceId(rank_);

  // permute qkv_tensor: permute((2, 0, 1))
  CreateTensorWithSameShape("model.layers.0.self_attn.query_key_value.weight", "empty_qkv_tensor");
  auto shape = weights_map_["model.layers.0.self_attn.query_key_value.weight"].shape;
  auto dtype = weights_map_["model.layers.0.self_attn.query_key_value.weight"].dtype;
  shape[0] = shape[0] * model_config_.head_num / (model_config_.head_num + 2 * model_config_.num_key_value_heads);
  AddWeightTensor("empty_q_in_tensor", shape, dtype);
  AddWeightTensor("empty_q_out_tensor", shape, dtype);
  Tensor& last_qkv_tensor = weights_map_["empty_qkv_tensor"];
  Tensor& q_in_tensor = weights_map_["empty_q_in_tensor"];
  Tensor& q_out_tensor = weights_map_["empty_q_out_tensor"];
  STATUS_CHECK_RETURN(PermuteQKVWeight(last_qkv_tensor, q_in_tensor, q_out_tensor, num_layer));
  GetBlockManager()->FreeContiguous(last_qkv_tensor.GetBlockId());
  GetBlockManager()->FreeContiguous(q_in_tensor.GetBlockId());
  GetBlockManager()->FreeContiguous(q_out_tensor.GetBlockId());

  // permute gate_proj, up_proj, down_proj: permute(1, 0)
  CreateTensorWithSameShape("model.layers.0.mlp.down_proj.weight", "empty_mlp_tensor");
  Tensor& last_mlp_tensor = weights_map_["empty_mlp_tensor"];
  STATUS_CHECK_RETURN(PermuteMLPWeight(last_mlp_tensor, num_layer));
  GetBlockManager()->FreeContiguous(last_mlp_tensor.GetBlockId());

  // permute o_proj: permute(1, 0)
  CreateTensorWithSameShape("model.layers.0.self_attn.o_proj.weight", "empty_o_proj_tensor");
  Tensor& last_o_proj_tensor = weights_map_["empty_o_proj_tensor"];
  STATUS_CHECK_RETURN(PermuteOutputProjectWeight(last_o_proj_tensor, num_layer));
  GetBlockManager()->FreeContiguous(last_o_proj_tensor.GetBlockId());

  // permute lm_head: permute(1, 0)
  CreateTensorWithSameShape("lm_head.weight", "empty_lm_head_tensor");
  Tensor& lm_head_tensor = weights_map_["lm_head.weight"];
  Tensor& lm_head_transpose_tensor = weights_map_["empty_lm_head_tensor"];
  Permute(lm_head_tensor, lm_head_transpose_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
  Tensor t = lm_head_transpose_tensor;
  lm_head_transpose_tensor = lm_head_tensor;
  t.shape = {lm_head_tensor.shape[1], lm_head_tensor.shape[0]};
  weights_map_["lm_head.weight"] = t;
  GetBlockManager()->FreeContiguous(lm_head_transpose_tensor.GetBlockId());

  weights_map_.erase("empty_qkv_tensor");
  weights_map_.erase("empty_q_in_tensor");
  weights_map_.erase("empty_q_out_tensor");
  weights_map_.erase("empty_mlp_tensor");
  weights_map_.erase("empty_lm_head_tensor");
  weights_map_.erase("empty_o_proj_tensor");
  return Status();
}

template <typename T>
bool CommonWeight<T>::IsLoaded() {
  return weights_had_loaded_;
}

template <typename T>
Status CommonWeight<T>::AddWeightTensor(std::string weight_name, std::vector<size_t> shapes, DataType dtype) {
  if (weights_map_.count(weight_name)) {
    NLLM_LOG_WARNING << fmt::format("The weight named {} has already been created. Skip creating the weight tensor.",
                                    weight_name);
    return Status();
  }
  size_t length = GetTypeSize(dtype);
  for (auto& dim : shapes) {
    length *= dim;
  }
  int block_id;
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->AllocateContiguous(length, block_id);

  weights_map_.emplace(weight_name, Tensor(MemoryDevice::MEMORY_DEVICE, dtype, shapes, block_id));
  return Status();
}

template <typename T>
Status CommonWeight<T>::CreateTensorWithSameShape(const std::string& origin_tensor_name,
                                                  const std::string& copy_tensor_name) {
  if (!weights_map_.count(origin_tensor_name)) {
    NLLM_LOG_ERROR << fmt::format("Create tensor {} faild: tensor {} not in weights map", copy_tensor_name,
                                  origin_tensor_name);
    exit(-1);
  }
  Tensor& origin_tensor = weights_map_[origin_tensor_name];
  AddWeightTensor(copy_tensor_name, origin_tensor.shape, origin_tensor.dtype);
  return Status();
}

template <typename T>
std::string CommonWeight<T>::ConcatLayerName(std::string layer_flag, int& layer_index, bool is_bias) {
  std::string layer_name =
      "model.layers." + std::to_string(layer_index) + "." + layer_flag + (is_bias ? ".bias" : ".weight");
  return layer_name;
}

template <typename T>
Tensor CommonWeight<T>::GetModelWeights(const std::string& weight_name) {
  if (!weights_map_.count(weight_name)) {
    NLLM_LOG_WARNING << fmt::format("weight name {} not in weights map", weight_name);
    return Tensor();
  }
  return weights_map_[weight_name];
}

template <typename T>
Status CommonWeight<T>::GetModelInfo(const ModelConfig& model_config) {
  weight_data_type_ = model_config.weight_data_type;
  model_name_ = model_config.name;
  tensor_para_size_ = model_config.tensor_para_size;
  return Status();
}

template <typename T>
void CommonWeight<T>::ProcessWeights() {
  int hidden_units = model_config_.hidden_units;
  int inter_size = model_config_.inter_size;
  int num_layer = model_config_.num_layer;
  int vocab_size = model_config_.vocab_size;
  // Convert of BFP16 and FP16
  if (model_config_.weight_data_type == TYPE_FP16 || model_config_.weight_data_type == TYPE_BF16) {
    for (auto& data_type_iter : weights_data_type_map_) {
      if (data_type_iter.second == TYPE_FP16 || data_type_iter.second == TYPE_BF16) {
        Tensor& tensor = weights_map_[data_type_iter.first];
        tensor.dtype = data_type_iter.second;
        GetBlockManager()->SetDeviceId(rank_);
        CastInplace(tensor, model_config_.weight_data_type, context_->GetMemoryManageStreams()[rank_]);
        tensor.dtype = model_config_.weight_data_type;
      }
    }
  }

  if (Singleton<Environment>::GetInstance()->EmbedTokensUseCpu()) {
    NLLM_LOG_INFO << "Enable EmbedTokensUseCpu";
    auto weight_name = "model.embed_tokens.weight";
    Tensor& tensor = weights_map_[weight_name];
    int block_id = 0;
    size_t length = tensor.GetTotalBytes();
    GetBlockManager()->AllocateHostContiguous(length, block_id);
    Tensor cpu_tensor(MemoryDevice::MEMORY_HOST, tensor.dtype, tensor.shape, block_id);
    MemcpyAsync(cpu_tensor.GetPtr<void>(), tensor.GetPtr<void>(), length, MEMCPY_DEVICE_TO_HOST,
                context_->GetMemoryManageStreams()[rank_]);
    GetBlockManager()->FreeContiguous(tensor.GetBlockId());
    weights_map_.insert_or_assign(weight_name, cpu_tensor);
    StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
  }

  PermuteTensor(hidden_units, inter_size, num_layer, vocab_size);

  // We use vocab_size to determine whether it is the Baichuan2 model.
  // If vocab_size is equal to 125,696, then it is the Baichuan2 model.
  // And Unlike Baichuan1, the Baichuan2 model requires normalizing the head weights. Refer to
  // repo: https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
  // commit: 84603cde5ebffb6084e476cfaeceaf0b8b91fe54
  // file: modeling_baichuan.py#L508
  if (model_config_.type == "baichuan" && vocab_size == 125696) {
    if (weights_data_type_map_.find("lm_head.weight") != weights_data_type_map_.end()) {
      Tensor& tensor = weights_map_["lm_head.weight"];
      GetBlockManager()->SetDeviceId(rank_);
      StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
      torch::ScalarType torch_dtype;
      if (tensor.dtype == DataType::TYPE_FP32) {
        torch_dtype = torch::kFloat32;
      } else if (tensor.dtype == DataType::TYPE_FP16) {
        torch_dtype = torch::kFloat16;
      }
#ifdef ENABLE_BFLOAT16
      else if (tensor.dtype == DataType::TYPE_BF16) {
        torch_dtype = torch::kBFloat16;
      }
#endif
      auto options = torch::TensorOptions().device(torch::kCUDA, rank_).dtype(torch_dtype);
      torch::Tensor in =
          torch::from_blob(tensor.GetPtr<void>(), {(int64_t)tensor.shape[0], (int64_t)tensor.shape[1]}, options);
      auto out = torch::nn::functional::normalize(in, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
      MemcpyAsync(tensor.GetPtr<void>(), out.data_ptr(), sizeof(T) * tensor.shape[0] * tensor.shape[1],
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    }
  }

  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
}

template class CommonWeight<float>;
template class CommonWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class CommonWeight<bfloat16>;
#endif

}  // namespace ksana_llm
