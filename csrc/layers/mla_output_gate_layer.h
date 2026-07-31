/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

// Fused Kimi-K3 MLA output gate.
class MlaOutputGateLayer : public BaseLayer {
 public:
  // x      [num_tokens, num_heads * v_head_dim]  attention output
  // gate   [num_tokens, num_heads * v_head_dim]  raw g_proj output, unactivated
  // output [num_tokens, num_heads * v_head_dim]  may alias x or gate
  //
  // The element count must be a multiple of 8 and all three buffers must be
  // 16-byte aligned; the kernel gates one 8-element (16 B) vector per thread.
  Status Forward(const Tensor* x, const Tensor* gate, Tensor* output, const LayerExecContext& exec_ctx = {});
};

}  // namespace ksana_llm
