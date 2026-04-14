/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "models/base/common_model_weight_loader.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

struct QuantConfig;

class Fp8E4m3Method {
 public:
  Fp8E4m3Method(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp, const QuantConfig& quant_config);

  Status load_attn_q_k_v_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                              const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  Status load_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_name,
                          const Tensor& weight_tensor, int dev_rank);

  Status load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  Status load_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                            const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  Status process_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_prefix_name, int dev_rank);

  Status process_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                             const std::string& weight_prefix_name, int dev_rank);

  Status process_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                  const std::string& weight_prefix_name, int dev_rank);

  Status process_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_prefix_name, int dev_rank);

  Status process_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                              const std::string& weight_prefix_name, int dev_rank, int num_experts);

  // Dynamic quantization: convert to FP8 and bind scale
  Status ConvertWeightToFp8(std::unordered_map<std::string, Tensor>& device_model_weights,
                            const std::string& weight_name, int dev_rank);

 private:
  // Check if weight is FP8 serialized checkpoint (static quantization)
  bool IsCheckpointFp8Serialized() const { return is_checkpoint_fp8_serialized_; }

  // Check if using block-wise FP8 quantization (per-block weight_scale, dynamic activation)
  bool IsFp8Blockwise() const { return is_fp8_blockwise_; }

  // Common load logic for all projection types: dispatches .weight_scale, .input_scale, .weight/.bias
  Status LoadProjTensor(std::unordered_map<std::string, Tensor>& device_model_weights,
                        const std::string& weight_name, const Tensor& weight_tensor, int dev_rank,
                        bool is_row_parallel);

  Status LoadWeightWithTPSplit(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_name, const Tensor& weight_tensor, int dev_rank,
                               bool is_row_parallel);

  Status LoadScaleTensor(std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_name,
                         const Tensor& weight_tensor, int dev_rank);

  // Load weight scale and split it across TP ranks to match the local weight's block layout.
  // For TP=1 or non-splittable shapes, falls back to full copy.
  Status LoadWeightScaleWithTPSplit(std::unordered_map<std::string, Tensor>& device_model_weights,
                                    const std::string& weight_name, const Tensor& weight_tensor, int dev_rank,
                                    bool is_row_parallel);

  // Process projection weights for FP8 quantization.
  Status ProcessProjWeight(std::unordered_map<std::string, Tensor>& device_model_weights,
                           const std::string& weight_prefix_name, int dev_rank);

  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  int tp_;
  bool is_checkpoint_fp8_serialized_;
  bool is_activation_scheme_static_;
  bool is_fp8_blockwise_;
};

}  // namespace ksana_llm
