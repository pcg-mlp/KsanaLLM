/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "configure/model_config.h"
#include "models/base/common_model_weight_loader.h"
#include "models/quant/cutlass_utils.h"
#include "models/quant/marlin_utils.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

class GptqMethod {
 public:
  // Constructor for GPTQ method.
  GptqMethod(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp, const QuantConfig& quant_config);

  // Load GPTQ q/k/v_proj qweights, zeros, scales, and bias using tensor-parallel split.
  Status load_attn_q_k_v_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                              const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  // Load GPTQ o_proj qweights, zeros and scales using tensor-parallel split.
  Status load_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_name,
                          const Tensor& weight_tensor, int dev_rank);

  // Load GPTQ gate/up_proj qweights, zeros and scales using tensor-parallel split.
  Status load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  // Load GPTQ down_proj qweights, zeros and scales using tensor-parallel split.
  Status load_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                            const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  // Process GPTQ q/k/v_proj qweights, zeros, scales, and bias.
  // Bind the scales and zeros to the weight tensor.
  Status process_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_prefix_name, int dev_rank);

  // Process GPTQ gate/up_proj qweights and zeros.
  // Bind the scales and zeros to the weight tensor.
  Status process_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                  const std::string& weight_prefix_name, int dev_rank);

  // Process GPTQ down_proj qweights and zeros.
  // Bind the scales and zeros to the weight tensor.
  Status process_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_prefix_name, int dev_rank);

  // Process GPTQ o_proj qweights and zeros.
  // Bind the scales and zeros to the weight tensor.
  Status process_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                             const std::string& weight_prefix_name, int dev_rank);

 private:
  // Process GPTQ weights according to the quant backend and bind scales/zeros to the weight tensor.
  Status convert_weight_and_bind(std::unordered_map<std::string, Tensor>& device_model_weights,
                                 const std::string& weight_prefix_name, int dev_rank);

  // Preprocess GPTQ weight (qweight/qzeros) for TP split in CUTLASS backend.
  Tensor PreprocessGptqWeightForTpSplit(const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  int tp_;
  QuantConfig quant_config_;

  // Helper objects for quantization operations.
  std::shared_ptr<CutlassUtils> cutlass_helper_{nullptr};
  std::shared_ptr<MarlinUtils> marlin_helper_{nullptr};
};

}  // namespace ksana_llm
