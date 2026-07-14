/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "utils/grammar_backend.h"
#include "configure/environment.h"  // For VocabInfo definition
#include "device/device_types.h"

#include "utils/nvidia/grammar_backend_nvidia.h"
// Note: Ascend header removed since we return nullptr for Ascend platform
// #include "utils/ascend/grammar_backend_ascend.h"

namespace ksana_llm {

std::unique_ptr<GrammarBackend> GrammarBackend::Create(const VocabInfo& vocab_info) {
  if (!vocab_info.initialized) {
    KLLM_LOG_ERROR << "VocabInfo not initialized";
    return nullptr;
  }

  if (ACTIVE_DEVICE_TYPE == DEVICE_TYPE_NVIDIA) {
    return std::make_unique<GrammarBackendNvidia>(vocab_info);
  } else {
    // Ascend platform does not support grammar backend yet
    KLLM_LOG_WARNING << "GrammarBackend is not supported on Ascend platforms, returning nullptr";
    return nullptr;
  }
}

}  // namespace ksana_llm
