/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/runtime/context.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class BaseModel {
 public:
  // Disable a default constructor
  BaseModel() {}

  virtual ~BaseModel() {}

  // The output logits pointer on device, used by sampler to avoid memory copy.
  virtual float* GetLogitsPtr() = 0;

  // The prefill stage.
  virtual Status ContextDecode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                               std::vector<ForwardRequest>& forward_reqs) = 0;

  // The decode stage.
  virtual Status Decode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                        std::vector<ForwardRequest>& forward_reqs) = 0;
};

}  // namespace ksana_llm
