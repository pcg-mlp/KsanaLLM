/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_set>

#include "models/deepseek_v3/deepseek_v3_weight_loader.h"
#include "models/deepseek_v4/deepseek_v4_config.h"

namespace ksana_llm {
// The deepseek v4 weight loader.
// Inherits from DeepSeekV3WeightLoader for shared infrastructure (weight_impl_, pipeline/runtime config, etc.)
// but implements its own ProcessModelWeights and PostProcessModelWeights independently.
class DeepSeekV4WeightLoader : public DeepSeekV3WeightLoader {
 public:
  DeepSeekV4WeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                         std::shared_ptr<Context> context);
  virtual ~DeepSeekV4WeightLoader() override;

  // Do some filter on model weight names.
  virtual Status FilterWeightNames(std::vector<std::string>& weight_names) override;

  // Process weights independently (does NOT call V3 parent's ProcessModelWeights).
  virtual Status ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
                                     std::unordered_map<std::string, Tensor>& device_model_weights,
                                     std::unordered_map<std::string, Tensor>& left_host_weights) override;

  // Post-process weights independently.
  virtual Status PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                         int dev_rank) override;

  // Pre-process: convert E8M0 weight_scale_inv to FP32 before main loading.
  virtual Status PreProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights) override;

 private:
  // Fuse two weights along dim=0 into a single tensor.
  void CombineWeightByRow(const std::string& file_weight_name, const Tensor& host_weight_tensor,
                          const std::string& first_name, const std::string& second_name, const std::string& final_name,
                          std::unordered_map<std::string, Tensor>& device_model_weights, int dev_rank,
                          size_t fused_dim0 = 0);

  // Load FP4 MoE expert weights (weight and weight_scale_inv) for a single expert.
  // Returns true if the weight was handled, false otherwise.
  bool LoadFp4MoeWeight(const std::string& file_weight_name, const Tensor& host_weight_tensor, int dev_rank,
                        int layer_idx, int expert_idx_, std::unordered_map<std::string, Tensor>& device_model_weights);

  // Cached V4 config pointer, initialized once in constructor.
  std::shared_ptr<DeepSeekV4Config> deepseek_v4_config_;
};
}  // namespace ksana_llm
