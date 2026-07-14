/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/weight_method/gptq_method.h"

#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "model_loader/model_loader_utils.h"
#include "models/base/utils.h"
#include "models/quant/cutlass_utils.h"
#include "models/quant/marlin_utils.h"
#include "utils/logger.h"

namespace ksana_llm {

GptqMethod::GptqMethod(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp,
                       const QuantConfig& quant_config)
    : common_weight_loader_(common_weight_loader), tp_(tp), quant_config_(quant_config) {
#ifdef ENABLE_CUDA
  // Initialize helper objects for different quantization backends
  // Quantization differences between CUTLASS and Marlin:
  // 1. Weight Packing Format:
  //    - CUTLASS: packing via CutlassPreprocessWeightsForMixedGemm
  //    - Marlin: packing via PackGptqWeight
  //
  // 2. Scale Handling:
  //    - CUTLASS: Direct per-group scales without reordering
  //    - Marlin: Requires MarlinPermuteScales to reorder scales
  //
  // 3. Zero-point Handling:
  //    - CUTLASS: Standard zero-point storage
  //    - Marlin: Specialized conversion via MarlinGptqToMarlinZeroPoints
  //
  cutlass_helper_ = std::make_shared<CutlassUtils>(common_weight_loader_->GetContext(),
                                                   common_weight_loader_->GetContext()->device->GetDeviceId(),
                                                   static_cast<int>(quant_config_.bits));
  // TODO(winminkong): 支持 quant_config_.desc_act == true 时的GPTQ模型
  marlin_helper_ = std::make_shared<MarlinUtils>(
      common_weight_loader_->GetContext(), common_weight_loader_->GetContext()->device->GetDeviceId(),
      static_cast<int>(quant_config_.bits), static_cast<int>(quant_config_.group_size));
#endif
}

/*
 * Preprocess GPTQ weight (qweight/qzeros) for TP split in CUTLASS backend.
 * For qweight:
 *    - Unpack from packed int32 to unpacked int8
 *    - Apply zero offset (subtract 2^(bits-1))
 *    - Repack to int4 format (pack 2 int4 values per int8)
 */
Tensor GptqMethod::PreprocessGptqWeightForTpSplit(const std::string& weight_name, const Tensor& weight_tensor,
                                                  int dev_rank) {
#ifdef ENABLE_CUDA
  if (quant_config_.backend != CUTLASS_LINEAR_BACKEND) {
    return weight_tensor;
  }

  torch::Tensor tensor = common_weight_loader_->GetTorchTensorFromTensor(weight_tensor);

  if (tensor.device().type() == torch::kCPU) {
    tensor = tensor.to(torch::Device(torch::kCUDA, dev_rank));
  }

  // Process qweight: unpack -> apply zero offset -> repack
  if (CheckWeightNameEndMatched(weight_name, {".qweight"})) {
    // Step 1: Unpack GPTQ qweight (from packed int32 to unpacked int8)
    tensor = cutlass_helper_->CutlassUnpackGPTQ(tensor);

    // Step 2: Apply zero offset (subtract 2^(bits-1))
    int8_t zero = std::pow(2, quant_config_.bits - 1);
    tensor = (tensor - zero).contiguous();

    // Step 3: Repack to int4 format (pack 2 int4 values per int8)
    tensor = cutlass_helper_->CutlassPackInt8ToPackedInt4(tensor);
  }

  return common_weight_loader_->GetTensorFromTorchTensor(tensor);
#else
  KLLM_THROW("gptq quant method is only supported on CUDA devices!");
#endif
}

Status GptqMethod::load_attn_q_k_v_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                        const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".qweight", ".scales"})) {
    // GPTQ qweight/scales for q/k/v_proj: split on dim=1 (ColPara)
    // For qweight with CUTLASS backend: need preprocess before TP split
    Tensor tensor_to_split = weight_tensor;
    if (CheckWeightNameEndMatched(weight_name, {".qweight"})) {
      tensor_to_split = PreprocessGptqWeightForTpSplit(weight_name, weight_tensor, dev_rank);
    }

    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(tensor_to_split, dev_rank, tp_, TensorParallelMode::ColPara);
  } else if (CheckWeightNameEndMatched(weight_name, {".bias"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  return Status();
}

Status GptqMethod::load_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                    const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".qweight", ".scales"})) {
    // GPTQ qweight/scales for o_proj: split on dim=0 (RowPara)
    // For qweight with CUTLASS backend: need preprocess before TP split
    Tensor tensor_to_split = weight_tensor;
    if (CheckWeightNameEndMatched(weight_name, {".qweight"})) {
      tensor_to_split = PreprocessGptqWeightForTpSplit(weight_name, weight_tensor, dev_rank);
    }

    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(tensor_to_split, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  return Status();
}

Status GptqMethod::load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                         const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".qweight", ".scales"})) {
    // GPTQ qweight/scales for gate/up_proj: split on dim=1 (ColPara)
    // For qweight with CUTLASS backend: need preprocess before TP split
    Tensor tensor_to_split = weight_tensor;
    if (CheckWeightNameEndMatched(weight_name, {".qweight"})) {
      tensor_to_split = PreprocessGptqWeightForTpSplit(weight_name, weight_tensor, dev_rank);
    }

    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(tensor_to_split, dev_rank, tp_, TensorParallelMode::ColPara);
  }
  return Status();
}

Status GptqMethod::load_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                      const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".qweight", ".scales"})) {
    // GPTQ qweight/scales for down_proj: split on dim=0 (RowPara)
    // For qweight with CUTLASS backend: need preprocess before TP split
    Tensor tensor_to_split = weight_tensor;
    if (CheckWeightNameEndMatched(weight_name, {".qweight"})) {
      tensor_to_split = PreprocessGptqWeightForTpSplit(weight_name, weight_tensor, dev_rank);
    }

    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(tensor_to_split, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  return Status();
}

Status GptqMethod::process_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                         const std::string& weight_prefix_name, int dev_rank) {
  const std::string q_proj_name = weight_prefix_name;
  const std::string k_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "k_proj");
  const std::string v_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "v_proj");
  const std::string qkv_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "query_key_value");

  // Merge q/k/v qweight/scales on dim=1
  {
    const std::vector<std::string> weight_suffixs = {"qweight", "scales"};
    for (const std::string& weight_suffix : weight_suffixs) {
      common_weight_loader_->AutoMergeWeight(
          {q_proj_name + weight_suffix, k_proj_name + weight_suffix, v_proj_name + weight_suffix},
          qkv_proj_name + weight_suffix, device_model_weights, dev_rank, 1);
    }
  }

  return convert_weight_and_bind(device_model_weights, qkv_proj_name, dev_rank);
}

Status GptqMethod::process_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                            const std::string& weight_prefix_name, int dev_rank) {
  return convert_weight_and_bind(device_model_weights, weight_prefix_name, dev_rank);
}

Status GptqMethod::process_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                         const std::string& weight_prefix_name, int dev_rank) {
  return convert_weight_and_bind(device_model_weights, weight_prefix_name, dev_rank);
}

Status GptqMethod::process_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                       const std::string& weight_prefix_name, int dev_rank) {
  return convert_weight_and_bind(device_model_weights, weight_prefix_name, dev_rank);
}

/*
 * Process GPTQ weights according to the quant backend and bind scales to the weight tensor.
 * This function performs three main steps:
 * - Pack qweight into backend-specific runtime layout and rename to weight
 * - Bind scales tensors to weight tensor
 */
Status GptqMethod::convert_weight_and_bind(std::unordered_map<std::string, Tensor>& device_model_weights,
                                           const std::string& weight_prefix_name, int dev_rank) {
#ifdef ENABLE_CUDA
  const std::string qweight_name = weight_prefix_name + "qweight";
  const std::string scales_name = weight_prefix_name + "scales";
  const std::string weight_name = weight_prefix_name + "weight";

  if (device_model_weights.find(qweight_name) == device_model_weights.end()) return Status();
  if (device_model_weights.find(scales_name) == device_model_weights.end()) return Status();

  // 1) Pack qweight into runtime layout
  {
    torch::Tensor qweight_gpu = common_weight_loader_->GetTorchTensorFromTensor(device_model_weights.at(qweight_name));
    if (quant_config_.backend == CUTLASS_LINEAR_BACKEND) {
      torch::Tensor processed_tensor_gpu = cutlass_helper_->CutlassPreprocessWeightsForMixedGemmWarpper(
          qweight_gpu, llm_kernels::nvidia::QuantType::W4_A16);
      device_model_weights[weight_name] = common_weight_loader_->GetTensorFromTorchTensor(processed_tensor_gpu);
    } else if (quant_config_.backend == MARLIN_LINEAR_BACKEND) {
      std::optional<torch::Tensor> perm_gpu = std::nullopt;
      torch::Tensor processed_tensor_gpu = marlin_helper_->PackGptqWeight(qweight_gpu, perm_gpu);
      device_model_weights[weight_name] = common_weight_loader_->GetTensorFromTorchTensor(processed_tensor_gpu);

      torch::Tensor scales_gpu = common_weight_loader_->GetTorchTensorFromTensor(device_model_weights.at(scales_name));

      // Dynamically call MarlinPermuteScales based on scales dtype
      if (scales_gpu.scalar_type() == torch::kFloat) {
        scales_gpu = marlin_helper_->MarlinPermuteScales<float>(
            scales_gpu, quant_config_.group_size * scales_gpu.size(-2), scales_gpu.size(-1));
      } else if (scales_gpu.scalar_type() == torch::kHalf) {
        scales_gpu = marlin_helper_->MarlinPermuteScales<half>(
            scales_gpu, quant_config_.group_size * scales_gpu.size(-2), scales_gpu.size(-1));
      } else if (scales_gpu.scalar_type() == torch::kBFloat16) {
        scales_gpu = marlin_helper_->MarlinPermuteScales<bfloat16>(
            scales_gpu, quant_config_.group_size * scales_gpu.size(-2), scales_gpu.size(-1));
      } else {
        KLLM_THROW(fmt::format("Unsupported scales dtype for MarlinPermuteScales: {}",
                               torch::toString(scales_gpu.scalar_type())));
      }
      device_model_weights.erase(scales_name);
      device_model_weights[scales_name] = common_weight_loader_->GetTensorFromTorchTensor(scales_gpu);
    } else {
      KLLM_THROW("Unsupported backend for group quant, only support CUTLASS and MARLIN.");
    }
    device_model_weights.erase(qweight_name);
  }

  // 2) Bind
  device_model_weights[weight_name].scales = &device_model_weights[scales_name];

  KLLM_LOG_DEBUG << fmt::format(
      "convert_weight_and_bind: weight name: {} weight shape: {}, dtype: {}, scale shape: {}, dtype: {}", weight_name,
      Vector2Str(device_model_weights[weight_name].shape), device_model_weights[weight_name].dtype,
      Vector2Str(device_model_weights[scales_name].shape), device_model_weights[scales_name].dtype);

  KLLM_LOG_DEBUG << fmt::format("Successfully process quant weight name: {} on rank: {}", weight_name, dev_rank);
  return Status();
#else
  return Status(RetCode::RET_MODEL_QUANT_FAILED, "GPTQ quantization is only supported on CUDA devices");
#endif
}

}  // namespace ksana_llm
