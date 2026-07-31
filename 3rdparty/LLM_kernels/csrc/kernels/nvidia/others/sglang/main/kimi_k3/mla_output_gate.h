/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace llm_kernels {
namespace nvidia {

// Fused Kimi-K3 MLA output gate: out = bf16(bf16(x) * bf16(sigmoid(gate))).
//
// x:                    [num_elements] bf16, contiguous. Typically a flattened
//                        [num_tokens, num_heads * v_head_dim] attention output.
// gate:                 [num_elements] bf16, contiguous. Raw g_proj output; the
//                        sigmoid is applied here, so pass it unactivated.
// out:                  [num_elements] bf16, contiguous. May alias x or gate
//                        for an in-place update.
// num_elements:         total element count of each buffer. Must be a multiple
//                        of 8, and x/gate/out must be 16-byte aligned: one
//                        thread gates one 8-element (16 B) vector. Both are
//                        asserted. num_elements == 0 is a no-op.
void mla_output_gate(const void* x, const void* gate, void* out, size_t num_elements, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
