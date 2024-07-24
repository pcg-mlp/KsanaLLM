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
    if (model_config_.quant_config.method == QUANT_GPTQ) {
      return true;
    }
    if (model_config_.quant_config.method == QUANT_FP8_E4M3) {
      if (model_config_.quant_config.is_checkpoint_fp8_serialized) {
        KLLM_LOG_ERROR << "Loading of fp8 weights from checkpoint is not supported.";
        throw std::runtime_error("Loading of fp8 weights from checkpoint is not supported.");
        // return true;
      } else {
        return false;
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
  if (tensor_name.find(".qweight") != std::string::npos || tensor_name.find(".scales") != std::string::npos) {
    if (tensor_name.find("W_pack") != std::string::npos) {
      size_t q_proj_size = model_config_.size_per_head * model_config_.head_num;
      size_t kv_proj_size = model_config_.size_per_head * model_config_.num_key_value_heads;

      if (q_proj_size % tensor_para_size_ != 0 || kv_proj_size % tensor_para_size_ != 0) {
        KLLM_LOG_ERROR << fmt::format("Model can't run with tensor_para_size == {}", tensor_para_size_);
        exit(-1);
      }

      const std::string prefix_name = tensor_name.substr(0, tensor_name.find("W_pack"));
      const std::string suffix_name =
          tensor_name.substr(tensor_name.find("W_pack") + sizeof("W_pack"), tensor_name.length() - 1);
      std::string q_proj_name = prefix_name + "q_proj." + suffix_name;
      std::string k_proj_name = prefix_name + "k_proj." + suffix_name;
      std::string v_proj_name = prefix_name + "v_proj." + suffix_name;

      std::vector<size_t> q_proj_shape = {weight_shape[0], q_proj_size / tensor_para_size_};
      std::vector<size_t> kv_proj_shape = {weight_shape[0], kv_proj_size / tensor_para_size_};

      tensor_manager_->AddWeightTensor(q_proj_name, q_proj_shape, weight_data_type);
      tensor_manager_->AddWeightTensor(k_proj_name, kv_proj_shape, weight_data_type);
      tensor_manager_->AddWeightTensor(v_proj_name, kv_proj_shape, weight_data_type);

      auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
      if (tensor_name.find(".scales") != std::string::npos) {
        options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kHalf);
      }
      torch::Tensor tensor = torch::from_blob(
          weight_ptr, {static_cast<int64_t>(weight_shape[0]), static_cast<int64_t>(weight_shape[1])}, options);
      // TODO(jinxcwu): use Memcpy2DAsync to the usage of torch
      auto tensors = torch::split(
          tensor,
          {static_cast<int64_t>(q_proj_size), static_cast<int64_t>(kv_proj_size), static_cast<int64_t>(kv_proj_size)},
          1);

      q_proj_size /= tensor_para_size_;
      kv_proj_size /= tensor_para_size_;

      tensors[0] = tensors[0].slice(1, rank_ * q_proj_size, (rank_ + 1) * q_proj_size).contiguous();
      tensors[1] = tensors[1].slice(1, rank_ * kv_proj_size, (rank_ + 1) * kv_proj_size).contiguous();
      tensors[2] = tensors[2].slice(1, rank_ * kv_proj_size, (rank_ + 1) * kv_proj_size).contiguous();

      MemcpyAsync(weights_map_[q_proj_name].GetPtr<void>(), tensors[0].data_ptr(),
                  weights_map_[q_proj_name].GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);
      MemcpyAsync(weights_map_[k_proj_name].GetPtr<void>(), tensors[1].data_ptr(),
                  weights_map_[k_proj_name].GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);
      MemcpyAsync(weights_map_[v_proj_name].GetPtr<void>(), tensors[2].data_ptr(),
                  weights_map_[v_proj_name].GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);
    } else if (tensor_name.find("o_proj") != std::string::npos || tensor_name.find("down_proj") != std::string::npos) {
      if (weight_shape[0] % tensor_para_size_ != 0) {
        KLLM_LOG_ERROR << fmt::format("Model can't run with tensor_para_size == {}", tensor_para_size_);
        exit(-1);
      }

      weight_shape[0] /= tensor_para_size_;
      tensor_manager_->AddWeightTensor(tensor_name, weight_shape, weight_data_type);

      size_t single_proj_size = weights_map_[tensor_name].GetTotalBytes();
      MemcpyAsync(weights_map_[tensor_name].GetPtr<void>(), weight_ptr + rank_ * single_proj_size, single_proj_size,
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    } else {
      if (weight_shape[1] % tensor_para_size_ != 0) {
        KLLM_LOG_ERROR << fmt::format("Model can't run with tensor_para_size == {}", tensor_para_size_);
        exit(-1);
      }

      auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
      if (tensor_name.find(".scales") != std::string::npos) {
        options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kHalf);
      }
      torch::Tensor tensor = torch::from_blob(
          weight_ptr, {static_cast<int64_t>(weight_shape[0]), static_cast<int64_t>(weight_shape[1])}, options);

      if (model_config_.type == "chatglm" && tensor_name.find("gate_proj") != std::string::npos) {
        const std::string gate_name = tensor_name;
        const std::string up_name = std::regex_replace(gate_name, std::regex("gate"), "up");

        auto tensors = torch::chunk(tensor, 2, -1);

        weight_shape[1] = weight_shape[1] / 2 / tensor_para_size_;
        tensor_manager_->AddWeightTensor(gate_name, weight_shape, weight_data_type);
        tensor_manager_->AddWeightTensor(up_name, weight_shape, weight_data_type);

        const size_t single_size = weight_shape[1];
        torch::Tensor gate_tensor = tensors[0].slice(1, rank_ * single_size, (rank_ + 1) * single_size).contiguous();
        torch::Tensor up_tensor = tensors[1].slice(1, rank_ * single_size, (rank_ + 1) * single_size).contiguous();

        MemcpyAsync(weights_map_[gate_name].GetPtr<void>(), gate_tensor.data_ptr(),
                    weights_map_[gate_name].GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
        MemcpyAsync(weights_map_[up_name].GetPtr<void>(), up_tensor.data_ptr(),
                    weights_map_[up_name].GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      } else {
        size_t single_size = weight_shape[1] / tensor_para_size_;
        tensor = tensor.slice(1, rank_ * single_size, (rank_ + 1) * single_size).contiguous();

        weight_shape[1] /= tensor_para_size_;
        tensor_manager_->AddWeightTensor(tensor_name, weight_shape, weight_data_type);

        MemcpyAsync(weights_map_[tensor_name].GetPtr<void>(), tensor.data_ptr(),
                    weights_map_[tensor_name].GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      }
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

template <typename T>
torch::Tensor QuantWeight<T>::ConvertGroupLayout(torch::Tensor qweight_int32) {
  torch::Tensor qweight_unpacked_int8 = UnpackInt32IntoInt8(qweight_int32.clone().t()).t().contiguous() - 8;
  torch::Tensor qweight_packed_int4 = PackInt8TensorToPackedInt4(qweight_unpacked_int8);
  torch::Tensor processed_tensor = PreprocessWeightsForMixedGemmWarpper(
      qweight_packed_int4, llm_kernels::nvidia::QuantType::PACKED_INT4_WEIGHT_ONLY);
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
          weights_map_[q_name].GetPtr<void>(),
          {static_cast<int64_t>(weights_map_[q_name].shape[0]), static_cast<int64_t>(weights_map_[q_name].shape[1])},
          options);
      torch::Tensor k_tensor_gpu = torch::from_blob(
          weights_map_[k_name].GetPtr<void>(),
          {static_cast<int64_t>(weights_map_[k_name].shape[0]), static_cast<int64_t>(weights_map_[k_name].shape[1])},
          options);
      torch::Tensor v_tensor_gpu = torch::from_blob(
          weights_map_[v_name].GetPtr<void>(),
          {static_cast<int64_t>(weights_map_[v_name].shape[0]), static_cast<int64_t>(weights_map_[v_name].shape[1])},
          options);
      torch::Tensor qkv_tensor_gpu = torch::cat({q_tensor_gpu, k_tensor_gpu, v_tensor_gpu}, -1);

      tensor_manager_->AddWeightTensor(
          qkv_name, {static_cast<size_t>(qkv_tensor_gpu.size(0)), static_cast<size_t>(qkv_tensor_gpu.size(1))},
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
      torch::Tensor qweight_int32_gpu = torch::from_blob(weights_map_[qweight_name].GetPtr<void>(),
                                                         {static_cast<int64_t>(weights_map_[qweight_name].shape[0]),
                                                          static_cast<int64_t>(weights_map_[qweight_name].shape[1])},
                                                         options);
      torch::Tensor qweight_int32 = qweight_int32_gpu.to(torch::kCPU);
      torch::Tensor processed_tensor = ConvertGroupLayout(qweight_int32);

      tensor_manager_->AddWeightTensor(
          weight_name, {static_cast<size_t>(processed_tensor.size(0)), static_cast<size_t>(processed_tensor.size(1))},
          TYPE_INT8);
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

#ifdef ENABLE_FP8
template <typename T>
Status QuantWeight<T>::ConvertFp8E4m3(const int num_layer) {
  if (!context_->IsGemmFp8Supported()) {
    KLLM_LOG_ERROR << "Cublas is insufficient to support FP8.";
    throw std::runtime_error("Cublas is insufficient to support FP8.");
  }
  DataType quant_type = TYPE_FP8_E4M3;
  KLLM_LOG_INFO << "Converting weight to fp8_e4m3";
  GetBlockManager()->SetDeviceId(rank_);
  std::string name;
  name = ".mlp.gate_proj.weight";
  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string weight_name = "model.layers." + std::to_string(layer_idx) + name;
    STATUS_CHECK_RETURN(ConvertFp8E4m3Tensor(weight_name, quant_type));
  }
  name = ".mlp.up_proj.weight";
  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string weight_name = "model.layers." + std::to_string(layer_idx) + name;
    STATUS_CHECK_RETURN(ConvertFp8E4m3Tensor(weight_name, quant_type));
  }
  name = ".mlp.down_proj.weight";
  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string weight_name = "model.layers." + std::to_string(layer_idx) + name;
    STATUS_CHECK_RETURN(ConvertFp8E4m3Tensor(weight_name, quant_type));
  }
  name = ".self_attn.o_proj.weight";
  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string weight_name = "model.layers." + std::to_string(layer_idx) + name;
    STATUS_CHECK_RETURN(ConvertFp8E4m3Tensor(weight_name, quant_type));
  }
  name = ".self_attn.query_key_value.weight";
  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string weight_name = "model.layers." + std::to_string(layer_idx) + name;
    STATUS_CHECK_RETURN(ConvertFp8E4m3Tensor(weight_name, quant_type));
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
  Fp8DynamicQuantize(1, trans_tensor.shape[0] * trans_tensor.shape[1],
                     static_cast<const T*>(trans_tensor.GetPtr<void>()), quant_tensor.GetPtr<void>(),
                     static_cast<float*>(scale_tensor.GetPtr<void>()), context_->GetMemoryManageStreams()[rank_].Get());
  quant_tensor.scales = &scale_tensor;
  GetBlockManager()->FreeContiguous(weights_map_[weight_name].GetBlockId());
  GetBlockManager()->FreeContiguous(weights_map_[trans_name].GetBlockId());
  weights_map_[weight_name] = weights_map_[quant_name];
  weights_map_.erase(quant_name);
  weights_map_.erase(trans_name);
  return Status();
}
#endif

template class QuantWeight<float>;
template class QuantWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class QuantWeight<bfloat16>;
#endif

}  // namespace ksana_llm
