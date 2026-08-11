/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <algorithm>
#include <cstdint>

#include "csrc/utils/nvidia/cuda_utils.h"

// Runtime tile-size selection for the SM90 BF16 moe_fused_mul_sum triton kernel.
//
// Co-located with the kernel generator (generate_triton_moe_fused_mul_sum.py /
// triton_moe_fused_mul_sum_creator.py) so that any change to the BLOCK_M /
// BLOCK_K heuristic stays with the kernel it configures. The rules below must
// match the variant grid the creator pre-compiles, otherwise
// InvokeTritonKernel will throw at launch time on a missing cubin.
//
// Reference (Python):
//   compute_block_k(size) = min(triton.next_power_of_2(size), 512)
//   BLOCK_MS = (1, 2, 4)   # from triton_moe_fused_mul_sum_creator.py
//
// Note: there is intentionally no lower bound (floor) on BLOCK_K. The earlier
// max(., 256) was a C++-only addition with no counterpart on the generator
// side and no matching cubin for BLOCK_K=256; for the supported sizes
// (4096 / 7168) it was dead code, and for any smaller size it would have
// selected a non-existent cubin.

namespace llm_kernels {
namespace nvidia {

inline int SelectBlockMSM90(int num_tokens) {
  if (num_tokens <= 4) return 1;
  if (num_tokens <= 128) return 2;
  return 4;
}

inline int SelectBlockKSM90(int hidden_size) {
  return std::min(static_cast<int>(llm_kernels::utils::NextPow2(static_cast<uint32_t>(hidden_size))), 512);
}

}  // namespace nvidia
}  // namespace llm_kernels
