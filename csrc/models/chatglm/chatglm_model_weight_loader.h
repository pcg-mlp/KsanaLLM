/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "models/base/base_model_weight_loader.h"
#include "models/base/common_model_weight_loader.h"
#include "models/weight_method/weight_method.h"

namespace ksana_llm {

class ChatglmModelWeightLoader : public BaseModelWeightLoader {
 public:
  ChatglmModelWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                           std::shared_ptr<Context> context);
  virtual ~ChatglmModelWeightLoader() override;

  // Do some filter on model weight names.
  virtual Status FilterWeightNames(std::vector<std::string>& weight_names) override;

  // Invoked only once before ProcessModelWeights.
  virtual Status PreProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights) override;

  // Process weights, such as rename, split, merge, type convert, quantization, etc.
  virtual Status ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
                                     std::unordered_map<std::string, Tensor>& device_model_weights,
                                     std::unordered_map<std::string, Tensor>& left_host_weights) override;

  // Invoked only once after ProcessModelWeights.
  virtual Status PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                         int dev_rank) override;

 public:
  // Custom methods for ChatGLM-specific weight processing
  // QKV weight: ChatGLM uses [Q_all, K_all, V_all] concatenated layout
  Status load_chatglm_qkv_weight(std::unordered_map<std::string, Tensor>& device_model_weights,
                                 const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  // QKV bias: Same concatenated layout as weight
  Status load_chatglm_qkv_bias(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  // Fused gate_up weight: Store as gate_up_proj for TwoLayeredFFN's fused mode
  Status load_chatglm_gate_up(std::unordered_map<std::string, Tensor>& device_model_weights,
                              const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  // Process QKV weight (transpose)
  Status process_chatglm_qkv(std::unordered_map<std::string, Tensor>& device_model_weights,
                             const std::string& weight_prefix_name, int dev_rank);

 private:
  PipelineConfig pipeline_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  std::shared_ptr<WeightMethod> weight_method_;

  size_t tp_;
};

}  // namespace ksana_llm
