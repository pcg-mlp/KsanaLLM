/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/models/llama/llama.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

float* Llama::GetLogitsPtr() { return nullptr; }

Status Llama::ContextDecode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                            std::vector<ForwardRequest>& forward_reqs) {
  NLLM_LOG_INFO << "llama context decode stage inference";
  return Status();
}

Status Llama::Decode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                     std::vector<ForwardRequest>& forward_reqs) {
  NLLM_LOG_INFO << "llama decode stage inference";
  return Status();
}

}  // namespace numerous_llm
