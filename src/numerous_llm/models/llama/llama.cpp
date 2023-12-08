/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/models/llama/llama.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

Status Llama::ContextDecode() {
  NLLM_LOG_INFO << "llama context decode stage inference";
  return Status();
}

Status Llama::Decode() {
  NLLM_LOG_INFO << "llama decode stage inference";
  return Status();
}

}  // namespace numerous_llm
