/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/models/base/base_weight.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class BaseModel {
 public:
  // Disable a default constructor
  BaseModel() {}

  virtual ~BaseModel() {}

  // The prefill stage.
  virtual Status ContextDecode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                               const std::vector<TensorMap*>& input_tensor_maps,
                               std::vector<TensorMap*>& output_tensor_maps) = 0;

  // The decode stage.
  virtual Status Decode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                        const std::vector<TensorMap*>& input_tensor_maps,
                        std::vector<TensorMap*>& output_tensor_maps) = 0;
};

}  // namespace numerous_llm