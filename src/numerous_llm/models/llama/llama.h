/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/models/base/base_model.h"
#include "numerous_llm/models/llama/llama_weight.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/utils.h"

namespace numerous_llm {

class Llama : public BaseModel {
 public:
  Llama() {}
  ~Llama() {}

  float* GetLogitsPtr();

  // The prefill stage.
  Status ContextDecode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                       std::vector<ForwardRequest>& forward_reqs);

  // The decode stage.
  Status Decode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs);
};

}  // namespace numerous_llm
