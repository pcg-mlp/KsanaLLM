/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "models/base/base_model_weight_loader.h"
#include "models/base/common_model_weight_loader.h"
#include "models/qwen/qwen_config.h"
#include "models/weight_method/weight_method.h"

namespace ksana_llm {

// Dictionary mapping quantization methods to their weight suffixes that need to be filtered
inline const std::unordered_map<QuantMode, std::vector<std::string>> kQuantWeightSuffixFilters = {
    {QUANT_AWQ, {".scales"}},
    {QUANT_GPTQ, {".scales"}},
    {QUANT_FP8_E4M3, {".weight_scale", ".input_scale"}},
    {QUANT_BLOCK_FP8_E4M3, {".weight_scale", ".input_scale"}},
    {QUANT_W4A8_AWQ, {".input_scale", ".weight_scale", ".weight_scale_2", ".pre_quant_scale"}},
};

// Qwen weight loader
class QwenWeightLoader : public BaseModelWeightLoader {
 public:
  QwenWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                   std::shared_ptr<Context> context);
  virtual ~QwenWeightLoader() override;

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

 private:
  PipelineConfig pipeline_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  std::shared_ptr<WeightMethod> weight_method_;

  size_t tp_;
};

}  // namespace ksana_llm
