/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/input_refit_layer.h"

namespace ksana_llm {

Status InputRefitLayer::Forward(const std::vector<const Tensor*>& emb_tensors, const Tensor* pos_tensor,
                                Tensor* output, const LayerExecContext& exec_ctx) {
  // Stage 2: text-only pass-through. Multimodal image token replacement deferred to Stage 3.
  // For text-only inference, this layer is a no-op: output is already set up by the caller.
  // Just propagate shape and dtype from output (which is the embedding buffer).
  // TECHDEBT: stage3-multimodal: implement image token replacement with XPU memcpy.
  return Status();
}

}  // namespace ksana_llm
