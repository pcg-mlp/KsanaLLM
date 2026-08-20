/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "models/base/common_model_weight_loader.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

/**
 * W4A8KlxMethod: Weight loading method for Kunlun (XPU) W4A8 quantized models.
 *
 * This method handles the Kimi-K2.6-klx-w4a8-gptq format:
 *   - W8 layers (Attention/Dense MLP): INT8 per-channel, no packing
 *   - W4 layers (MoE experts): INT4 packed into INT8, pack_factor=2
 *   - No activation quantization scales in checkpoint (dynamic per-token at runtime)
 *   - weight_scale stored as BF16, needs cast to FP32 for kernel consumption
 *
 * Completely independent from W4A8AWQMethod (NVIDIA TRT-LLM format).
 */
class W4A8KlxMethod {
 public:
  W4A8KlxMethod(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp);

  // ============ Weight Loading ============

  // Load W8 layer weight/scale (Attention projections, Dense MLP, Shared Expert)
  // For weight: MoveToDevice or TP split
  // For scale: MoveToDevice or TP split (following weight's output dim)
  Status LoadW8Weight(std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_name,
                      const Tensor& weight_tensor, int dev_rank, TensorParallelMode tp_mode, size_t tp_size);

  // Load W4 expert weight/scale (MoE routed experts, already packed)
  // Just MoveToDevice (expert stacking handled by DeepSeekV3WeightLoader)
  Status LoadW4ExpertWeight(std::unordered_map<std::string, Tensor>& device_model_weights,
                            const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  // ============ Post-processing ============

  // Post-process W8 weight: permute {1,0} to [k,n], cast scale to fp32, bind
  Status ProcessW8Weight(std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_name,
                         int dev_rank);

  // Post-process W4 weight: permute {1,0} to [k/2,n], cast scale to fp32, scale *= 7, bind
  Status ProcessW4Weight(std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_name,
                         int dev_rank);

  // Post-process MoE expert stacked weights (3D: [num_experts, out, in])
  // For W4: cast scale to fp32, scale *= 7, bind
  Status ProcessW4ExpertStackedWeight(std::unordered_map<std::string, Tensor>& device_model_weights,
                                      const std::string& weight_name, int dev_rank);

  // ============ Helpers ============

  // Check if a weight name belongs to W4 layer (MoE routed experts)
  static bool IsW4ExpertWeight(const std::string& weight_name);

  // Check if a tensor name is a weight_scale
  static bool IsWeightScale(const std::string& weight_name);

  // Get the scale name for a given weight name
  static std::string GetScaleName(const std::string& weight_name);

  // Get the weight name for a given scale name
  static std::string GetWeightName(const std::string& scale_name);

 private:
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  int tp_;
};

}  // namespace ksana_llm
