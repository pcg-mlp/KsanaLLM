/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/llm_runtime.h"
#include "numerous_llm/runtime/worker.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

Status LlmRuntime::Step(std::vector<InferRequest>& reqs) {
  NLLM_LOG_INFO << "llm runtime step invoked.";

  return Status();
}

}  // namespace numerous_llm
