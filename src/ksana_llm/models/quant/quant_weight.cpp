/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

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
    : model_config_(model_config), rank_(rank), context_(context), weights_map_(weights_map) {
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
  if (tensor_name.find(".g_idx") != std::string::npos || tensor_name.find(".qzeros") != std::string::npos ||
      tensor_name.find(".o_proj.bias") != std::string::npos ||
      tensor_name.find(".gate_proj.bias") != std::string::npos ||
      tensor_name.find(".up_proj.bias") != std::string::npos ||
      tensor_name.find(".down_proj.bias") != std::string::npos) {
    return true;
  }
  return false;
}

template <typename T>
bool QuantWeight<T>::CheckQuantModel() {
  // TODO(jinxcwu): make a struct to store different quant type: gptq, awq, ...
  if (model_config_.is_quant) {
    if (model_config_.quant_config.method == "gptq") {
      return true;
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
  if (tensor_name.find(".qweight") != std::string::npos || tensor_name.find(".scales") != std::string::npos) {
    if (tensor_name.find("o_proj") != std::string::npos || tensor_name.find("down_proj") != std::string::npos) {
      if (weight_shape[0] % tensor_para_size_ != 0) {
        NLLM_LOG_ERROR << fmt::format("Model can't run with tensor_para_size == {}", tensor_para_size_);
        exit(-1);
      }

      weight_shape[0] /= tensor_para_size_;
      tensor_manager_->AddWeightTensor(tensor_name, weight_shape, weight_data_type);

      size_t single_proj_size = weights_map_[tensor_name].GetTotalBytes();
      MemcpyAsync(weights_map_[tensor_name].GetPtr<void>(), weight_ptr + rank_ * single_proj_size, single_proj_size,
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    } else {
      if (weight_shape[1] % tensor_para_size_ != 0) {
        NLLM_LOG_ERROR << fmt::format("Model can't run with tensor_para_size == {}", tensor_para_size_);
        exit(-1);
      }

      auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
      if (tensor_name.find(".scales") != std::string::npos) {
        options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kHalf);
      }
      torch::Tensor tensor = torch::from_blob(weight_ptr, {weight_shape[0], weight_shape[1]}, options);
      size_t single_size = weight_shape[1] / tensor_para_size_;
      tensor = tensor.slice(1, rank_ * single_size, (rank_ + 1) * single_size);
      tensor = tensor.contiguous();

      weight_shape[1] /= tensor_para_size_;
      tensor_manager_->AddWeightTensor(tensor_name, weight_shape, weight_data_type);

      MemcpyAsync(weights_map_[tensor_name].GetPtr<void>(), tensor.data_ptr(),
                  weights_map_[tensor_name].GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);
    }
    return true;
  }
#endif
  return false;
}

#ifdef ENABLE_CUDA
template <typename T>
torch::Tensor QuantWeight<T>::UnpackInt32IntoInt8(const torch::Tensor& w_packed) {
  auto w_packed_contiguous = w_packed.contiguous();
  auto w_packed_int4x2 = w_packed_contiguous.view(torch::kUInt8);
  auto w_unpacked = torch::zeros({w_packed_int4x2.size(0), w_packed_int4x2.size(1) * 2}, torch::kInt8);
  w_unpacked.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)},
                        w_packed_int4x2 % 16);
  w_unpacked.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)},
                        w_packed_int4x2 / 16);
  return w_unpacked.contiguous();
}

template <typename T>
torch::Tensor QuantWeight<T>::PackInt8TensorToPackedInt4(torch::Tensor weight) {
  std::vector<int64_t> packed_tensor_size(weight.dim());
  for (int i = 0; i < weight.dim(); ++i) {
    packed_tensor_size[i] = weight.size(i);
  }
  packed_tensor_size[weight.dim() - 1] = (packed_tensor_size[weight.dim() - 1] + 1) / 2;

  torch::Tensor packed_weight =
      torch::zeros(packed_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

  int8_t* unpacked_ptr = reinterpret_cast<int8_t*>(weight.data_ptr());
  int8_t* packed_ptr = reinterpret_cast<int8_t*>(packed_weight.data_ptr());

  for (size_t packed_idx = 0; packed_idx < packed_weight.numel(); ++packed_idx) {
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

template <typename T>
torch::Tensor QuantWeight<T>::ConvertGPTQLayout(torch::Tensor qweight_int32) {
  torch::Tensor qweight_unpacked_int8 = UnpackInt32IntoInt8(qweight_int32.clone().t()).t().contiguous() - 8;
  torch::Tensor qweight_packed_int4 = PackInt8TensorToPackedInt4(qweight_unpacked_int8);
  torch::Tensor processed_tensor = PreprocessWeightsForMixedGemmWarpper(
      qweight_packed_int4, llm_kernels::nvidia::QuantType::PACKED_INT4_WEIGHT_ONLY);
  return processed_tensor;
}
#endif

template <typename T>
Status QuantWeight<T>::ConvertGPTQTensor(int hidden_units, int inter_size, int num_layer) {
  if (!enable_) {
    return Status();
  }

#ifdef ENABLE_CUDA
  GetBlockManager()->SetDeviceId(rank_);

  // pack q, k, v to qkv
  std::vector<std::string> needed_slove_weights_name = {"qweight", "scales"};
  for (std::string& needed_slove_weight_name : needed_slove_weights_name) {
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    if (needed_slove_weight_name == "scales") {
      options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kHalf);
    }
    for (size_t layer_idx = 0; layer_idx < (size_t)num_layer; ++layer_idx) {
      std::string q_name = fmt::format("model.layers.{}.self_attn.q_proj.{}", layer_idx, needed_slove_weight_name);
      std::string k_name = fmt::format("model.layers.{}.self_attn.k_proj.{}", layer_idx, needed_slove_weight_name);
      std::string v_name = fmt::format("model.layers.{}.self_attn.v_proj.{}", layer_idx, needed_slove_weight_name);
      std::string qkv_name =
          fmt::format("model.layers.{}.self_attn.query_key_value.{}", layer_idx, needed_slove_weight_name);

      torch::Tensor q_tensor_gpu = torch::from_blob(
          weights_map_[q_name].GetPtr<void>(), {weights_map_[q_name].shape[0], weights_map_[q_name].shape[1]}, options);
      torch::Tensor k_tensor_gpu = torch::from_blob(
          weights_map_[k_name].GetPtr<void>(), {weights_map_[k_name].shape[0], weights_map_[k_name].shape[1]}, options);
      torch::Tensor v_tensor_gpu = torch::from_blob(
          weights_map_[v_name].GetPtr<void>(), {weights_map_[v_name].shape[0], weights_map_[v_name].shape[1]}, options);
      torch::Tensor qkv_tensor_gpu = torch::cat({q_tensor_gpu, k_tensor_gpu, v_tensor_gpu}, -1);

      tensor_manager_->AddWeightTensor(qkv_name, {qkv_tensor_gpu.size(0), qkv_tensor_gpu.size(1)},
                                       weights_map_[q_name].dtype);
      MemcpyAsync(weights_map_[qkv_name].GetPtr<void>(), qkv_tensor_gpu.data_ptr(),
                  weights_map_[qkv_name].GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);

      GetBlockManager()->FreeContiguous(weights_map_[q_name].GetBlockId());
      GetBlockManager()->FreeContiguous(weights_map_[k_name].GetBlockId());
      GetBlockManager()->FreeContiguous(weights_map_[v_name].GetBlockId());
      weights_map_.erase(q_name);
      weights_map_.erase(k_name);
      weights_map_.erase(v_name);
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

      auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
      torch::Tensor qweight_int32_gpu =
          torch::from_blob(weights_map_[qweight_name].GetPtr<void>(),
                           {weights_map_[qweight_name].shape[0], weights_map_[qweight_name].shape[1]}, options);
      torch::Tensor qweight_int32 = qweight_int32_gpu.to(torch::kCPU);
      torch::Tensor processed_tensor = ConvertGPTQLayout(qweight_int32);

      tensor_manager_->AddWeightTensor(weight_name, {processed_tensor.size(0), processed_tensor.size(1)}, TYPE_INT8);
      MemcpyAsync(weights_map_[weight_name].GetPtr<void>(), processed_tensor.data_ptr(),
                  weights_map_[weight_name].GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);

      GetBlockManager()->FreeContiguous(weights_map_[qweight_name].GetBlockId());
      weights_map_.erase(qweight_name);

      weights_map_[weight_name].scales = &weights_map_[scales_name];
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

template class QuantWeight<float>;
template class QuantWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class QuantWeight<bfloat16>;
#endif

}  // namespace ksana_llm
