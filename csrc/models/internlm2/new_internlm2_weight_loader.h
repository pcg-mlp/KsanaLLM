/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "models/base/base_model_weight_loader.h"
#include "models/base/common_model_weight_loader.h"
#include "models/internlm2/internlm2_config.h"
#include "models/weight_method/weight_method.h"

namespace ksana_llm {

class NewInternlm2WeightLoader : public BaseModelWeightLoader {
 public:
  NewInternlm2WeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                           std::shared_ptr<Context> context);
  virtual ~NewInternlm2WeightLoader() override;

  virtual Status FilterWeightNames(std::vector<std::string> &weight_names) override;

  virtual Status PreProcessModelWeights(const std::unordered_map<std::string, Tensor> &host_model_weights) override;

  virtual Status ProcessModelWeights(const std::unordered_map<std::string, Tensor> &host_model_weights, int dev_rank,
                                     std::unordered_map<std::string, Tensor> &device_model_weights,
                                     std::unordered_map<std::string, Tensor> &left_host_weights) override;

  virtual Status PostProcessModelWeights(std::unordered_map<std::string, Tensor> &dev_weights_map,
                                         int dev_rank) override;

 private:
  Status ProcessInterleavedQKV(const std::string &host_weight_name, const Tensor &host_weight_tensor,
                               std::unordered_map<std::string, Tensor> &device_model_weights, int dev_rank);

  PipelineConfig pipeline_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  std::shared_ptr<WeightMethod> weight_method_;

  size_t tp_;
};

}  // namespace ksana_llm

