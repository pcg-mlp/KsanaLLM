/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/gguf_file_tensor_loader.h"
#include "ksana_llm/utils/pytorch_file_tensor_loader.h"
#include "ksana_llm/utils/safetensors_file_tensor_loader.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class BaseWeight {
 public:
  BaseWeight() {}
  explicit BaseWeight(const ModelConfig& model_config, int rank);
  virtual ~BaseWeight() {}

  // 查表,返回 weights_map_[weight_name]
  virtual Tensor GetModelWeights(const std::string& weight_name) = 0;
  // 加载权重
  virtual Status LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader,
                                     std::vector<std::string>& weight_name_list,
                                     std::vector<std::string>& custom_name_list) = 0;

  virtual void ProcessWeights() = 0;

  virtual void SetEmbeddingsConfig() = 0;
};

}  // namespace ksana_llm
