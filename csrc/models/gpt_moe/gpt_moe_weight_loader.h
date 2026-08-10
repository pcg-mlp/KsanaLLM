/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "models/base/base_model_weight_loader.h"
#include "models/base/common_model_weight_loader.h"
#include "models/weight_method/weight_method.h"

namespace ksana_llm {

// gpt_moe weight loader
class GptMoeWeightLoader : public BaseModelWeightLoader {
 public:
  GptMoeWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                     std::shared_ptr<Context> context);

  // Do some filter on model weight names.
  Status FilterWeightNames(std::vector<std::string>& weight_names) override;

  // Invoked only once before ProcessModelWeights.
  Status PreProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights) override;

  // Process weights, such as rename, split, merge, type convert, quantization, etc.
  Status ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
                             std::unordered_map<std::string, Tensor>& device_model_weights,
                             std::unordered_map<std::string, Tensor>& left_host_weights) override;

  // Invoked only once after ProcessModelWeights.
  Status PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map, int dev_rank) override;

 public:
  Status load_attn_fused_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                  const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  Status process_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                              const std::string& weight_prefix_name, int dev_rank);

 private:
  PipelineConfig pipeline_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  std::shared_ptr<WeightMethod> weight_method_;

  size_t tp_;
};

}  // namespace ksana_llm
