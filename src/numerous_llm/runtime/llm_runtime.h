/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "numerous_llm/runtime/infer_request.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class LlmRuntime {
 public:
  LlmRuntime() {}

  // Execute one req in parallel.
  Status Step(std::vector<std::shared_ptr<InferRequest>> &reqs);

 private:
  std::unordered_map<std::string, std::unordered_map<InferStage, std::pair<TensorMap, TensorMap>>> grouped_reqs_map_;
};

}  // namespace numerous_llm
