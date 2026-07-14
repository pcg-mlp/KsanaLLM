/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "utils/grammar_matcher.h"
#include "device/device_types.h"

#include "utils/nvidia/grammar_matcher_nvidia.h"


namespace ksana_llm {

std::shared_ptr<GrammarMatcherWrapper> GrammarMatcherWrapper::Create(
    std::shared_ptr<CompiledGrammar> compiled_grammar) {
  if (ACTIVE_DEVICE_TYPE == DEVICE_TYPE_NVIDIA) {
    return std::make_shared<GrammarMatcherWrapperNvidia>(compiled_grammar);
  } else {
    // Ascend platform does not support grammar matcher yet
    KLLM_LOG_WARNING << "GrammarMatcherWrapper is not supported on Ascend platforms, returning nullptr";
    return nullptr;
  }
}

}  // namespace ksana_llm
