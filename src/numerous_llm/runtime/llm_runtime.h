/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "numerous_llm/runtime/infer_request.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class LlmRuntime {
 public:
  // Execute one req in parallel.
  Status Step(std::vector<InferRequest> &reqs);
};

}  // namespace numerous_llm
