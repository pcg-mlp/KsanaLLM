/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/llm_runtime.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

Status LlmRuntime::Step(std::vector<InferRequest> &reqs) {
  NLLM_LOG_INFO << "llm runtime step invoked." << std::endl;

  for (const InferRequest& req : reqs) {
    // this forward will execute on differ thread and its own GPU
    req.model_instance->Forward(req);
  }

  return Status();
}

} // namespace numerous_llm
