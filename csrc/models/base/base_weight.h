/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/context.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

class BaseWeight {
 public:
  BaseWeight() {}
  explicit BaseWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                      std::shared_ptr<Context> context);
  virtual ~BaseWeight() {}

  // 查表,返回 weights_map_[weight_name]
  virtual Tensor GetModelWeights(const std::string& weight_name) = 0;

  virtual void PrintDebugMessage() {}

 protected:
  int rank_ = 0;
  std::shared_ptr<Context> context_{nullptr};
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  PipelineConfig pipeline_config_;

  std::unordered_map<std::string, Tensor> weights_map_;
  std::unordered_map<std::string, DataType> weights_data_type_map_;

  struct {
    std::unordered_set<int16_t> all, moe, dense;
  } required_layer_idx_;
};

}  // namespace ksana_llm
