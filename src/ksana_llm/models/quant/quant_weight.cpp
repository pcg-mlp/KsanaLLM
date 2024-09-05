/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM
 * Copyright (c) 2024, Tencent Inc.  All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ksana_llm/models/quant/quant_weight.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/nn/functional/normalization.h>

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

namespace ksana_llm {

template <typename T>
QuantWeight<T>::QuantWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context,
                            std::unordered_map<std::string, Tensor>& weights_map)
    : weights_map_(weights_map), rank_(rank), context_(context), model_config_(model_config) {
  enable_ = CheckQuantModel();
  tensor_manager_ = std::make_shared<TensorManager>(rank, weights_map_);
  tensor_para_size_ = model_config.tensor_para_size;
}

template <typename T>
QuantWeight<T>::~QuantWeight() {}

template <typename T>
bool QuantWeight<T>::IsEnable() {
  return enable_;
}

template <typename T>
bool QuantWeight<T>::FilterOutQuantWeight(const std::string& tensor_name) {
  if (!enable_) {
    return false;
  }

  std::vector<std::string> skip_lists = {".g_idx", ".o_proj.bias", ".gate_proj.bias", ".up_proj.bias",
                                         ".down_proj.bias"};
  if (model_config_.quant_config.method == QUANT_GPTQ) {
    skip_lists.push_back(".qzeros");
  }
  for (const std::string& skip : skip_lists) {
    if (tensor_name.find(skip) != std::string::npos) {
      return true;
    }
  }
  return false;
}

template <typename T>
bool QuantWeight<T>::CheckQuantModel() {
  // TODO(jinxcwu): make a struct to store different quant type: gptq, awq, ...
  if (model_config_.is_quant) {
    if (model_config_.quant_config.method == QUANT_GPTQ) {
      return true;
    }
    if (model_config_.quant_config.method == QUANT_AWQ) {
      return true;
    }
    if (model_config_.quant_config.method == QUANT_FP8_E4M3) {
      if (context_->IsGemmFp8Supported()) {
        KLLM_LOG_INFO << "Device is sufficient to support FP8 GEMM.";
      } else {
        KLLM_THROW("Device is insufficient to support FP8 GEMM.");
      }
    }
  }
  return false;
}

template <typename T>
bool QuantWeight<T>::LoadQuantWeight(std::string& tensor_name, std::vector<size_t>& weight_shape,
                                     DataType& weight_data_type, void* weight_ptr) {
  if (!enable_) {
    return false;
  }

#ifdef ENABLE_CUDA
  if (tensor_name.find(".qweight") != std::string::npos || tensor_name.find(".scales") != std::string::npos ||
      tensor_name.find(".qzeros") != std::string::npos) {
    if (tensor_name.find("W_pack") != std::string::npos) {
      size_t q_proj_size = model_config_.size_per_head * model_config_.head_num;
      size_t kv_proj_size = model_config_.size_per_head * model_config_.num_key_value_heads;
      if (q_proj_size % tensor_para_size_ != 0 || kv_proj_size % tensor_para_size_ != 0) {
        KLLM_THROW(
            fmt::format("Model can't run with tensor_para_size == {}. "
                        "The size of q_proj_size {} or kv_proj_size {} cannot be evenly divided by the size of "
                        "tensor_parallel_size_.",
                        tensor_para_size_, q_proj_size, kv_proj_size));
      }
      auto options = torch::TensorOptions().device(torch::kCPU).dtype(GetTorchTypeFromDataType(weight_data_type));
      torch::Tensor tensor =
          torch::from_blob(weight_ptr, std::vector<int64_t>(weight_shape.begin(), weight_shape.end()), options);

      std::string q_proj_name = std::regex_replace(tensor_name, std::regex("W_pack"), "q_proj");
      std::string k_proj_name = std::regex_replace(tensor_name, std::regex("W_pack"), "k_proj");
      std::string v_proj_name = std::regex_replace(tensor_name, std::regex("W_pack"), "v_proj");

      tensor = AutoUnpack(tensor_name, tensor);
      size_t s = (q_proj_size + kv_proj_size + kv_proj_size) / tensor.size(1);
      q_proj_size /= s;
      kv_proj_size /= s;
      auto tensors = torch::split(
          tensor,
          {static_cast<int64_t>(q_proj_size), static_cast<int64_t>(kv_proj_size), static_cast<int64_t>(kv_proj_size)},
          1);

      q_proj_size /= tensor_para_size_;
      kv_proj_size /= tensor_para_size_;
      tensors[0] = tensors[0].slice(1, rank_ * q_proj_size, (rank_ + 1) * q_proj_size).contiguous();
      tensors[1] = tensors[1].slice(1, rank_ * kv_proj_size, (rank_ + 1) * kv_proj_size).contiguous();
      tensors[2] = tensors[2].slice(1, rank_ * kv_proj_size, (rank_ + 1) * kv_proj_size).contiguous();

      AddWeightFromTorchTensor(q_proj_name, tensors[0]);
      AddWeightFromTorchTensor(k_proj_name, tensors[1]);
      AddWeightFromTorchTensor(v_proj_name, tensors[2]);
    } else if (tensor_name.find("o_proj") != std::string::npos || tensor_name.find("down_proj") != std::string::npos) {
      if (weight_shape[0] % tensor_para_size_ != 0) {
        KLLM_THROW(
            fmt::format("Model can't run with tensor_para_size == {}."
                        "The size of weight_shape[0] {} cannot be evenly divided by the size of tensor_para_size_",
                        tensor_para_size_, weight_shape[0]));
      }
      auto options = torch::TensorOptions().device(torch::kCPU).dtype(GetTorchTypeFromDataType(weight_data_type));
      torch::Tensor tensor =
          torch::from_blob(weight_ptr, std::vector<int64_t>(weight_shape.begin(), weight_shape.end()), options);

      tensor = AutoUnpack(tensor_name, tensor);

      size_t single_size = tensor.size(0) / tensor_para_size_;
      tensor = tensor.slice(0, rank_ * single_size, (rank_ + 1) * single_size).contiguous();
      AddWeightFromTorchTensor(tensor_name, tensor);
    } else {
      if (weight_shape[1] % tensor_para_size_ != 0) {
        KLLM_THROW(
            fmt::format("Model can't run with tensor_para_size == {}."
                        "The size of weight_shape[1] {} cannot be evenly divided by the size of tensor_para_size_",
                        tensor_para_size_, weight_shape[1]));
      }
      auto options = torch::TensorOptions().device(torch::kCPU).dtype(GetTorchTypeFromDataType(weight_data_type));
      torch::Tensor tensor =
          torch::from_blob(weight_ptr, std::vector<int64_t>(weight_shape.begin(), weight_shape.end()), options);

      tensor = AutoUnpack(tensor_name, tensor);

      if (model_config_.type == "chatglm" && tensor_name.find("gate_proj") != std::string::npos) {
        const std::string gate_name = tensor_name;
        const std::string up_name = std::regex_replace(gate_name, std::regex("gate"), "up");

        auto tensors = torch::chunk(tensor, 2, -1);

        const size_t single_size = tensors[0].size(1) / tensor_para_size_;
        torch::Tensor gate_tensor = tensors[0].slice(1, rank_ * single_size, (rank_ + 1) * single_size).contiguous();
        torch::Tensor up_tensor = tensors[1].slice(1, rank_ * single_size, (rank_ + 1) * single_size).contiguous();

        AddWeightFromTorchTensor(gate_name, gate_tensor);
        AddWeightFromTorchTensor(up_name, up_tensor);
      } else {
        size_t single_size = tensor.size(1) / tensor_para_size_;
        tensor = tensor.slice(1, rank_ * single_size, (rank_ + 1) * single_size).contiguous();
        AddWeightFromTorchTensor(tensor_name, tensor);
      }
    }
    return true;
  }
#endif
  return false;
}

#ifdef ENABLE_CUDA
template <typename T>
torch::Tensor QuantWeight<T>::AutoUnpack(const std::string& tensor_name, torch::Tensor& tensor) {
  if (tensor_name.find(".qweight") != std::string::npos) {
    if (model_config_.quant_config.method == QUANT_GPTQ) {
      tensor = UnpackGPTQ(tensor);
    } else if (model_config_.quant_config.method == QUANT_AWQ) {
      tensor = UnpackAWQ(tensor, model_config_.quant_config.bits, model_config_.quant_config.group_size);
    }
    int8_t zero = std::pow(2, model_config_.quant_config.bits - 1);
    tensor = (tensor - zero).contiguous();
    tensor = PackInt8ToPackedInt4(tensor);
  }
  if (tensor_name.find(".qzeros") != std::string::npos && model_config_.quant_config.method == QUANT_AWQ) {
    tensor = UnpackAWQ(tensor, model_config_.quant_config.bits, model_config_.quant_config.group_size);
    tensor = tensor.to(torch::kHalf);
  }
  return tensor;
}

// Follow the logic from https://github.com/casper-hansen/AutoAWQ/blob/v0.2.6/awq/utils/packing_utils.py
template <typename T>
torch::Tensor QuantWeight<T>::UnpackAWQ(const torch::Tensor& qweight, int bits, int group_size) {
  torch::Tensor iweight = UnpackQWeight(qweight, bits);
  torch::Tensor reverse_order_tensor = GetReverseOrder(iweight, bits);
  iweight = iweight.index_select(1, reverse_order_tensor).contiguous();
  auto mask = (1 << bits) - 1;
  iweight = torch::bitwise_and(iweight, mask);
  return iweight.contiguous();
}

template <typename T>
torch::Tensor QuantWeight<T>::GetReverseOrder(const torch::Tensor& iweights, int bits) {
  torch::Tensor reverse_order_tensor = torch::arange(iweights.size(-1), torch::kInt64);
  reverse_order_tensor = reverse_order_tensor.view({-1, 32 / bits});
  reverse_order_tensor =
      reverse_order_tensor.index_select(1, torch::tensor({0, 4, 1, 5, 2, 6, 3, 7}, torch::kInt64)).contiguous();
  reverse_order_tensor = reverse_order_tensor.view({-1});
  return reverse_order_tensor;
}

template <typename T>
torch::Tensor QuantWeight<T>::UnpackQWeight(const torch::Tensor& qtensor, int bits) {
  torch::Tensor shifts = torch::arange(0, 32, bits).unsqueeze(0).unsqueeze(0);
  torch::Tensor itensor = torch::bitwise_right_shift(qtensor.unsqueeze(-1), shifts).to(torch::kInt8);
  itensor = itensor.view({itensor.size(0), -1});
  return itensor;
}

// Unpack [k/groupsize,n]int32 to [k,n]int4
template <typename T>
torch::Tensor QuantWeight<T>::UnpackGPTQ(const torch::Tensor& w_packed) {
  auto w_packed_contiguous = w_packed.t().contiguous();
  auto w_packed_int4x2 = w_packed_contiguous.view(torch::kUInt8);
  auto w_unpacked = torch::zeros({w_packed_int4x2.size(0), w_packed_int4x2.size(1) * 2}, torch::kInt8);
  w_unpacked.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)},
                        w_packed_int4x2 % 16);
  w_unpacked.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)},
                        w_packed_int4x2 / 16);
  return w_unpacked.t().contiguous();
}

// Pack [k,n]int4 to [k,n/2]int8, each byte save 2 int4 weight
template <typename T>
torch::Tensor QuantWeight<T>::PackInt8ToPackedInt4(torch::Tensor weight) {
  std::vector<int64_t> packed_tensor_size(weight.dim());
  for (int i = 0; i < weight.dim(); ++i) {
    packed_tensor_size[i] = weight.size(i);
  }
  packed_tensor_size[weight.dim() - 1] = (packed_tensor_size[weight.dim() - 1] + 1) / 2;

  torch::Tensor packed_weight =
      torch::zeros(packed_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

  int8_t* unpacked_ptr = reinterpret_cast<int8_t*>(weight.data_ptr());
  int8_t* packed_ptr = reinterpret_cast<int8_t*>(packed_weight.data_ptr());

  for (int64_t packed_idx = 0; packed_idx < packed_weight.numel(); ++packed_idx) {
    int8_t packed_int4s = 0;
    int8_t elt_0 = unpacked_ptr[2 * packed_idx + 0];
    int8_t elt_1 = unpacked_ptr[2 * packed_idx + 1];

    packed_int4s |= ((elt_0 & 0x0F));
    packed_int4s |= int8_t(elt_1 << 4);

    packed_ptr[packed_idx] = packed_int4s;
  }
  return packed_weight;
}

template <typename T>
torch::Tensor QuantWeight<T>::PreprocessWeightsForMixedGemmWarpper(torch::Tensor row_major_quantized_weight,
                                                                   llm_kernels::nvidia::QuantType quant_type) {
  const size_t bits_in_quant_type = GetBitsInQuantType(quant_type);

  const size_t num_experts = row_major_quantized_weight.dim() == 2 ? 1 : row_major_quantized_weight.size(0);
  const size_t num_rows = row_major_quantized_weight.size(-2);
  const size_t num_cols = (8 / bits_in_quant_type) * row_major_quantized_weight.size(-1);

  torch::Tensor processed_tensor = torch::zeros_like(row_major_quantized_weight);
  int8_t* input_byte_ptr = reinterpret_cast<int8_t*>(row_major_quantized_weight.data_ptr());
  int8_t* output_byte_ptr = reinterpret_cast<int8_t*>(processed_tensor.data_ptr());

  PreprocessWeightsForMixedGemm(output_byte_ptr, input_byte_ptr, {num_experts, num_rows, num_cols}, quant_type);

  return processed_tensor;
}
#endif

template <typename T>
Status QuantWeight<T>::ConvertGroupTensor(int hidden_units, int inter_size, int num_layer) {
  if (!enable_) {
    return Status();
  }

#ifdef ENABLE_CUDA
  GetBlockManager()->SetDeviceId(rank_);

  // pack q, k, v to qkv
  std::vector<std::string> needed_slove_weights_name = {"qweight", "scales"};
  if (model_config_.quant_config.method == QUANT_AWQ) {
    needed_slove_weights_name.push_back("qzeros");
  }
  for (std::string& needed_slove_weight_name : needed_slove_weights_name) {
    for (size_t layer_idx = 0; layer_idx < (size_t)num_layer; ++layer_idx) {
      std::string q_name = fmt::format("model.layers.{}.self_attn.q_proj.{}", layer_idx, needed_slove_weight_name);
      std::string k_name = fmt::format("model.layers.{}.self_attn.k_proj.{}", layer_idx, needed_slove_weight_name);
      std::string v_name = fmt::format("model.layers.{}.self_attn.v_proj.{}", layer_idx, needed_slove_weight_name);
      std::string qkv_name =
          fmt::format("model.layers.{}.self_attn.query_key_value.{}", layer_idx, needed_slove_weight_name);

      auto options =
          torch::TensorOptions().device(torch::kCUDA).dtype(GetTorchTypeFromDataType(weights_map_[q_name].dtype));
      torch::Tensor q_tensor_gpu = torch::from_blob(
          weights_map_[q_name].GetPtr<void>(),
          std::vector<int64_t>(weights_map_[q_name].shape.begin(), weights_map_[q_name].shape.end()), options);
      torch::Tensor k_tensor_gpu = torch::from_blob(
          weights_map_[k_name].GetPtr<void>(),
          std::vector<int64_t>(weights_map_[k_name].shape.begin(), weights_map_[k_name].shape.end()), options);
      torch::Tensor v_tensor_gpu = torch::from_blob(
          weights_map_[v_name].GetPtr<void>(),
          std::vector<int64_t>(weights_map_[v_name].shape.begin(), weights_map_[v_name].shape.end()), options);
      torch::Tensor qkv_tensor_gpu = torch::cat({q_tensor_gpu, k_tensor_gpu, v_tensor_gpu}, -1);
      torch::Tensor qkv_tensor = qkv_tensor_gpu.to(torch::kCPU);

      AddWeightFromTorchTensor(qkv_name, qkv_tensor);

      GetBlockManager()->FreeContiguous(weights_map_[q_name].GetBlockId());
      GetBlockManager()->FreeContiguous(weights_map_[k_name].GetBlockId());
      GetBlockManager()->FreeContiguous(weights_map_[v_name].GetBlockId());
      weights_map_.erase(q_name);
      weights_map_.erase(k_name);
      weights_map_.erase(v_name);
    }
  }

  // convert qzeros
  if (model_config_.quant_config.method == QUANT_AWQ) {
    needed_slove_weights_name = {"self_attn.query_key_value", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                                 "mlp.down_proj"};
    for (std::string& needed_slove_weight_name : needed_slove_weights_name) {
      for (size_t layer_idx = 0; layer_idx < (size_t)num_layer; ++layer_idx) {
        std::string scales_name = fmt::format("model.layers.{}.{}.scales", layer_idx, needed_slove_weight_name);
        std::string qzeros_name = fmt::format("model.layers.{}.{}.qzeros", layer_idx, needed_slove_weight_name);
        std::string zeros_name = fmt::format("model.layers.{}.{}.zeros", layer_idx, needed_slove_weight_name);

        auto scales_options = torch::TensorOptions()
                                  .device(torch::kCUDA)
                                  .dtype(GetTorchTypeFromDataType(weights_map_[scales_name].dtype));
        torch::Tensor scales_gpu = torch::from_blob(
            weights_map_[scales_name].GetPtr<void>(),
            std::vector<int64_t>(weights_map_[scales_name].shape.begin(), weights_map_[scales_name].shape.end()),
            scales_options);

        auto qzeros_options = torch::TensorOptions()
                                  .device(torch::kCUDA)
                                  .dtype(GetTorchTypeFromDataType(weights_map_[qzeros_name].dtype));
        torch::Tensor qzeros_gpu = torch::from_blob(
            weights_map_[qzeros_name].GetPtr<void>(),
            std::vector<int64_t>(weights_map_[qzeros_name].shape.begin(), weights_map_[qzeros_name].shape.end()),
            qzeros_options);

        int8_t zero = std::pow(2, model_config_.quant_config.bits - 1);
        // In AWQ: weight@fp16 = scale@fp16 * (qweight@uint4 - zeros@uint4)
        // In kernel: weight@fp16 = scale@fp16 * qweight@int4 + zeros@fp16
        // So: weight = scale * (qweight - zeros)
        //            = scale * (qweight - 8 + 8 - zeros)
        //            = scale * (qweight - 8) + scale * (8 - zeros)
        torch::Tensor zeros_cpu = (scales_gpu * (zero - qzeros_gpu)).to(torch::kCPU).contiguous();

        AddWeightFromTorchTensor(zeros_name, zeros_cpu);

        GetBlockManager()->FreeContiguous(weights_map_[qzeros_name].GetBlockId());
        weights_map_.erase(qzeros_name);
      }
    }
  }

  // convert qweight layout and binding scales
  needed_slove_weights_name = {"self_attn.query_key_value", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                               "mlp.down_proj"};
  for (std::string& needed_slove_weight_name : needed_slove_weights_name) {
    for (size_t layer_idx = 0; layer_idx < (size_t)num_layer; ++layer_idx) {
      std::string qweight_name = fmt::format("model.layers.{}.{}.qweight", layer_idx, needed_slove_weight_name);
      std::string scales_name = fmt::format("model.layers.{}.{}.scales", layer_idx, needed_slove_weight_name);
      std::string weight_name = fmt::format("model.layers.{}.{}.weight", layer_idx, needed_slove_weight_name);

      auto options =
          torch::TensorOptions().device(torch::kCUDA).dtype(GetTorchTypeFromDataType(weights_map_[qweight_name].dtype));
      torch::Tensor qweight_int32_gpu = torch::from_blob(
          weights_map_[qweight_name].GetPtr<void>(),
          std::vector<int64_t>(weights_map_[qweight_name].shape.begin(), weights_map_[qweight_name].shape.end()),
          options);
      torch::Tensor qweight_int32 = qweight_int32_gpu.to(torch::kCPU);
      torch::Tensor processed_tensor =
          PreprocessWeightsForMixedGemmWarpper(qweight_int32, llm_kernels::nvidia::QuantType::PACKED_INT4_WEIGHT_ONLY);

      AddWeightFromTorchTensor(weight_name, processed_tensor);

      GetBlockManager()->FreeContiguous(weights_map_[qweight_name].GetBlockId());
      weights_map_.erase(qweight_name);

      weights_map_[weight_name].scales = &weights_map_[scales_name];

      if (model_config_.quant_config.method == QUANT_AWQ) {
        std::string zeros_name = fmt::format("model.layers.{}.{}.zeros", layer_idx, needed_slove_weight_name);
        weights_map_[weight_name].zeros = &weights_map_[zeros_name];
      }
    }
  }

  // permute lm_head: permute(1, 0)
  tensor_manager_->CreateTensorWithSameShape("lm_head.weight", "empty_lm_head_tensor");
  Tensor& lm_head_tensor = weights_map_["lm_head.weight"];
  Tensor& lm_head_transpose_tensor = weights_map_["empty_lm_head_tensor"];
  Permute(lm_head_tensor, lm_head_transpose_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
  Tensor t = lm_head_transpose_tensor;
  lm_head_transpose_tensor = lm_head_tensor;
  t.shape = {lm_head_tensor.shape[1], lm_head_tensor.shape[0]};
  weights_map_["lm_head.weight"] = t;
  GetBlockManager()->FreeContiguous(lm_head_transpose_tensor.GetBlockId());
  weights_map_.erase("empty_lm_head_tensor");

  return Status();
#endif
  return Status(RetCode::RET_RUNTIME, "Not supported Ascend.");
}

#ifdef ENABLE_FP8
template <typename T>
Status QuantWeight<T>::ConvertFp8E4m3(const int num_layer) {
  if (!context_->IsGemmFp8Supported()) {
    KLLM_THROW("Cublas is insufficient to support FP8.");
  }
  DataType quant_type = TYPE_FP8_E4M3;
  KLLM_LOG_INFO << "Converting weight to fp8_e4m3";
  GetBlockManager()->SetDeviceId(rank_);
  std::vector<std::string> names = {".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
                                    ".self_attn.o_proj.weight", ".self_attn.query_key_value.weight"};
  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    for (auto name : names) {
      std::string weight_name = "model.layers." + std::to_string(layer_idx) + name;
      STATUS_CHECK_RETURN(ConvertFp8E4m3Tensor(weight_name, quant_type));
    }
  }
  return Status();
}

template <typename T>
Status QuantWeight<T>::ConvertFp8E4m3Tensor(std::string& weight_name, DataType quant_type) {
  // replace weight tensor with quantized tensor in weights_map_
  // and add scale tensor to weights_map_
  std::string trans_name = weight_name + "_trans";
  std::string quant_name = weight_name + "_quant";
  std::string scale_name = weight_name + "_scale";

  Tensor& weight_tensor = weights_map_[weight_name];
  auto weight_shape = weight_tensor.shape;
  if (weight_shape.back() % 2 != 0) {
    KLLM_LOG_INFO << "The last dim of weight is " << weight_shape.back() << " % 2 != 0 "
                  << ", therefore the weight cannot be calculated after quantization. "
                  << "Tensor of weight will not be quantized.";
    return Status();
  }

  // transpose weight from [k, n] to [n, k]
  std::vector<size_t> trans_shape{weight_shape[1], weight_shape[0]};
  tensor_manager_->AddWeightTensor(trans_name, trans_shape, weight_tensor.dtype);
  Tensor& trans_tensor = weights_map_[trans_name];
  weight_tensor.shape.insert(weight_tensor.shape.begin(), 1);
  trans_tensor.shape.insert(trans_tensor.shape.begin(), 1);
  // Permute only support 3D trans
  Permute(weight_tensor, trans_tensor, {0, 2, 1}, context_->GetMemoryManageStreams()[rank_]);
  weight_tensor.shape.erase(weight_tensor.shape.begin());
  trans_tensor.shape.erase(trans_tensor.shape.begin());

  tensor_manager_->AddWeightTensor(quant_name, trans_tensor.shape, quant_type);
  tensor_manager_->AddWeightTensor(scale_name, {1}, TYPE_FP32);
  Tensor& quant_tensor = weights_map_[quant_name];
  Tensor& scale_tensor = weights_map_[scale_name];
  Fp8E4m3Quantize(1, trans_tensor.shape[0] * trans_tensor.shape[1], static_cast<const T*>(trans_tensor.GetPtr<void>()),
                  quant_tensor.GetPtr<void>(), static_cast<float*>(scale_tensor.GetPtr<void>()), false,
                  context_->GetMemoryManageStreams()[rank_].Get());
  quant_tensor.weight_scales = &scale_tensor;
  GetBlockManager()->FreeContiguous(weights_map_[weight_name].GetBlockId());
  GetBlockManager()->FreeContiguous(weights_map_[trans_name].GetBlockId());
  weights_map_[weight_name] = weights_map_[quant_name];
  weights_map_.erase(quant_name);
  weights_map_.erase(trans_name);
  return Status();
}

template <typename T>
bool QuantWeight<T>::LoadFp8E4m3Scale(std::string& tensor_name, std::vector<size_t>& weight_shape,
                                      DataType& weight_data_type, void* weight_ptr) {
  GetBlockManager()->SetDeviceId(rank_);
  if (model_config_.quant_config.method != QUANT_FP8_E4M3) {
    return false;
  }
  if (tensor_name.find(".weight_scale") == std::string::npos && tensor_name.find(".input_scale") == std::string::npos) {
    return false;
  }
  // scale is float scalar
  if (weight_data_type != TYPE_FP32) {
    KLLM_THROW("Not support data type of scale:" + tensor_name);
  }
  // shape is empty or [1]
  if (!weight_shape.empty() || (weight_shape.size() == 1 && weight_shape[0] == 1)) {
    KLLM_THROW("Not support shape of scale:" + tensor_name);
  }
  weight_shape = {static_cast<size_t>(1)};
  KLLM_LOG_DEBUG << "Start loading scale:" << tensor_name;
  std::string weight_name;
  if (tensor_name.find("self_attn.W_pack") != std::string::npos) {
    // .weight_scale or .input_scale
    std::string suffix = tensor_name.substr(tensor_name.find_last_of("."), tensor_name.length());
    weight_name = tensor_name.substr(0, tensor_name.rfind("W_pack")) + "query_key_value" + suffix;
    tensor_manager_->AddWeightTensor(weight_name, weight_shape, weight_data_type);
    MemcpyAsync(weights_map_[weight_name].GetPtr<void>(), weight_ptr, sizeof(float), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[rank_]);
  } else if (tensor_name.find("_proj")) {
    weight_name = tensor_name;
    tensor_manager_->AddWeightTensor(weight_name, weight_shape, weight_data_type);
    MemcpyAsync(weights_map_[weight_name].GetPtr<void>(), weight_ptr, sizeof(float), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[rank_]);
  } else {
    KLLM_THROW("Not support scale:" + tensor_name);
  }
  KLLM_LOG_DEBUG << "Success loading scale:" << weight_name;
  return true;
}

template <typename T>
Status QuantWeight<T>::BindFp8E4m3Scale(const int num_layer, const int num_heads, const int num_kv_heads) {
  KLLM_LOG_INFO << "Start binding scale";
  GetBlockManager()->SetDeviceId(rank_);
  std::vector<std::string> names = {".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.down_proj.", ".self_attn.o_proj."};
  for (auto name : names) {
    BindFp8E4m3ScaleOfProjWeight(name, num_layer);
  }

  std::string name = ".self_attn.query_key_value.";
  BindFp8E4m3ScaleOfQkvWeight(name, num_layer, num_heads, num_kv_heads);

  KLLM_LOG_INFO << "Success binding scale";
  return Status();
}

template <typename T>
Status QuantWeight<T>::BindFp8E4m3ScaleOfProjWeight(std::string name, const int num_layer) {
  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string weight_name = "model.layers." + std::to_string(layer_idx) + name + "weight";
    std::string weight_scale_name = "model.layers." + std::to_string(layer_idx) + name + "weight_scale";
    std::string input_scale_name = "model.layers." + std::to_string(layer_idx) + name + "input_scale";
    if (weights_map_.find(weight_scale_name) != weights_map_.end()) {
      KLLM_LOG_DEBUG << "Binding " << weight_scale_name << " to " << weight_name;
      weights_map_[weight_name].weight_scales = &(weights_map_[weight_scale_name]);
    }
    if (weights_map_.find(input_scale_name) != weights_map_.end()) {
      KLLM_LOG_DEBUG << "Binding " << input_scale_name << " to " << weight_name;
      weights_map_[weight_name].input_scales = &(weights_map_[input_scale_name]);
    }
    if (weights_map_.find(weight_scale_name) == weights_map_.end()) {
      std::string gate_name = ".mlp.gate_proj.";
      std::string gate_weight_scale_name = "model.layers." + std::to_string(layer_idx) + gate_name + "weight_scale";
      std::string gate_input_scale_name = "model.layers." + std::to_string(layer_idx) + gate_name + "input_scale";
      if (weights_map_.find(gate_weight_scale_name) != weights_map_.end()) {
        KLLM_LOG_DEBUG << "Binding " << gate_weight_scale_name << " to " << weight_name;
        weights_map_[weight_name].weight_scales = &(weights_map_[gate_weight_scale_name]);
      }
      if (weights_map_.find(gate_input_scale_name) != weights_map_.end()) {
        KLLM_LOG_DEBUG << "Binding " << gate_input_scale_name << " to " << weight_name;
        weights_map_[weight_name].input_scales = &(weights_map_[gate_input_scale_name]);
      }
    }
  }
  return Status();
}

template <typename T>
Status QuantWeight<T>::BindFp8E4m3ScaleOfQkvWeight(std::string name, const int num_layer, const int num_heads,
                                                   const int num_kv_heads) {
  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string weight_name = "model.layers." + std::to_string(layer_idx) + name + "weight";
    std::string weight_scale_name = "model.layers." + std::to_string(layer_idx) + name + "weight_scale";
    std::string input_scale_name = "model.layers." + std::to_string(layer_idx) + name + "input_scale";

    std::string q_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.";
    std::string k_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.";
    std::string v_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.";

    std::string q_input_scale_name = q_name + "input_scale";
    std::string k_input_scale_name = k_name + "input_scale";
    std::string v_input_scale_name = v_name + "input_scale";
    // If weights of q,k,v are saved independently,
    // input_scale of qkv is max of q/k/v's input_scale
    if (weights_map_.find(q_input_scale_name) != weights_map_.end() &&
        weights_map_.find(k_input_scale_name) != weights_map_.end() &&
        weights_map_.find(v_input_scale_name) != weights_map_.end() &&
        weights_map_.find(input_scale_name) == weights_map_.end()) {
      tensor_manager_->AddWeightTensor(input_scale_name, {1}, weights_map_[q_input_scale_name].dtype);
      float* q_scale = static_cast<float*>(weights_map_[q_input_scale_name].GetPtr<void>());
      float* k_scale = static_cast<float*>(weights_map_[k_input_scale_name].GetPtr<void>());
      float* v_scale = static_cast<float*>(weights_map_[v_input_scale_name].GetPtr<void>());
      float* qkv_scale = static_cast<float*>(weights_map_[input_scale_name].GetPtr<void>());
      GetMaxScaleOfQkv(q_scale, k_scale, v_scale, qkv_scale);
    }

    std::string q_weight_scale_name = q_name + "weight_scale";
    std::string k_weight_scale_name = k_name + "weight_scale";
    std::string v_weight_scale_name = v_name + "weight_scale";
    // If weights of q,k,v are saved independently,
    // weight_scale of qkv is max of q/k/v's weight_scale,
    // weight of qkv need to be Rescale.
    if (weights_map_.find(q_weight_scale_name) != weights_map_.end() &&
        weights_map_.find(k_weight_scale_name) != weights_map_.end() &&
        weights_map_.find(v_weight_scale_name) != weights_map_.end() &&
        weights_map_.find(weight_scale_name) == weights_map_.end()) {
      tensor_manager_->AddWeightTensor(weight_scale_name, {1}, weights_map_[q_weight_scale_name].dtype);
      float* q_scale = static_cast<float*>(weights_map_[q_weight_scale_name].GetPtr<void>());
      float* k_scale = static_cast<float*>(weights_map_[k_weight_scale_name].GetPtr<void>());
      float* v_scale = static_cast<float*>(weights_map_[v_weight_scale_name].GetPtr<void>());
      float* qkv_scale = static_cast<float*>(weights_map_[weight_scale_name].GetPtr<void>());
      GetMaxScaleOfQkv(q_scale, k_scale, v_scale, qkv_scale);

      // q,k,v_weight * (q,k,v_scale / qkv_scale)
      Tensor& weight = weights_map_[weight_name];
      size_t n = weight.GetElementNumber() / (num_heads / num_kv_heads + 2);
      size_t size = weight.GetTotalBytes() / (num_heads / num_kv_heads + 2);
      void* q_weight = weight.GetPtr<void>();
      void* k_weight = q_weight + size * num_heads / num_kv_heads;
      void* v_weight = k_weight + size;
      RescaleFp8E4m3(q_weight, q_weight, n * num_heads / num_kv_heads, q_scale, qkv_scale,
                     context_->GetMemoryManageStreams()[rank_].Get());
      RescaleFp8E4m3(k_weight, k_weight, n, k_scale, qkv_scale, context_->GetMemoryManageStreams()[rank_].Get());
      RescaleFp8E4m3(v_weight, v_weight, n, v_scale, qkv_scale, context_->GetMemoryManageStreams()[rank_].Get());
    }

    if (weights_map_.find(weight_scale_name) != weights_map_.end()) {
      KLLM_LOG_DEBUG << "Binding " << weight_scale_name << " to " << weight_name;
      weights_map_[weight_name].weight_scales = &(weights_map_[weight_scale_name]);
    }
    if (weights_map_.find(input_scale_name) != weights_map_.end()) {
      KLLM_LOG_DEBUG << "Binding " << input_scale_name << " to " << weight_name;
      weights_map_[weight_name].input_scales = &(weights_map_[input_scale_name]);
    }
  }
  return Status();
}

template <typename T>
Status QuantWeight<T>::GetMaxScaleOfQkv(float* q_scale, float* k_scale, float* v_scale, float* qkv_scale) {
  auto options = torch::TensorOptions().device(torch::kCUDA, rank_).dtype(torch::kFloat32);
  torch::Tensor q_scale_tensor = torch::from_blob(q_scale, {1}, options);
  torch::Tensor k_scale_tensor = torch::from_blob(k_scale, {1}, options);
  torch::Tensor v_scale_tensor = torch::from_blob(v_scale, {1}, options);
  torch::Tensor qkv_scale_tensor = torch::from_blob(qkv_scale, {1}, options);
  torch::max_out(qkv_scale_tensor, q_scale_tensor, k_scale_tensor);
  torch::max_out(qkv_scale_tensor, qkv_scale_tensor, v_scale_tensor);
  return Status();
}
#endif

#ifdef ENABLE_CUDA
template <typename T>
Status QuantWeight<T>::AddWeightFromTorchTensor(const std::string& name, torch::Tensor& tensor) {
  tensor_manager_->AddWeightTensor(name, std::vector<size_t>(tensor.sizes().begin(), tensor.sizes().end()),
                                   GetDataTypeFromTorchType(tensor.scalar_type()));
  MemcpyAsync(weights_map_[name].GetPtr<void>(), tensor.data_ptr(), weights_map_[name].GetTotalBytes(),
              MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
  return Status();
}

#endif

template class QuantWeight<float>;
template class QuantWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class QuantWeight<bfloat16>;
#endif

}  // namespace ksana_llm
