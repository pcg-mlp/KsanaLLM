/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/split_layer.h"

namespace ksana_llm {

Status SplitLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                        std::shared_ptr<Context> context, int rank) {
  KLLM_THROW("SplitLayer not implement in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "SplitLayer not supported.");
}

Status SplitLayer::Forward(const Tensor* input, const std::vector<size_t>& split_sizes, std::vector<Tensor*> outputs,
                           const LayerExecContext& exec_ctx) {
  KLLM_THROW("SplitLayer not implement in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "SplitLayer not supported.");
}

}  // namespace ksana_llm
