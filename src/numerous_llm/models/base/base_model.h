/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/models/base/base_weight.h"
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/runtime/forward_request.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class BaseModel {
 public:
  // Disable a default constructor
  BaseModel() {}

  virtual ~BaseModel() {}

  // The output logits pointer on device, used by sampler to avoid memory copy.
  virtual float* GetLogitsPtr() = 0;

  // The prefill stage.
  virtual Status ContextDecode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                               std::vector<ForwardRequest>& forward_reqs) = 0;

  // The decode stage.
  virtual Status Decode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                        std::vector<ForwardRequest>& forward_reqs) = 0;
};

}  // namespace numerous_llm
