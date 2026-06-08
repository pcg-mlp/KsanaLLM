/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/flash_attention_layer.h"
#include <cstdlib>

namespace ksana_llm {

Status FlashAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                 std::shared_ptr<Context> context, int rank) {
  AttentionLayer::Init(parameters, runtime_config, context, rank);
  return Status(RET_UNDEFINED_REFERENCE, "FlashAttentionLayer not supported.");
}

Status FlashAttentionLayer::Forward(const CudaForwardParams& params, Tensor* output, const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "FlashAttentionLayer not supported.");
}

}  // namespace ksana_llm
