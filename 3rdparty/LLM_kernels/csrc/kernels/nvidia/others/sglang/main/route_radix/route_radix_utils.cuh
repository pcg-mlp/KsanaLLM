/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Utility headers for the route_radix kernel, adapted from
 *   python/sglang/kernels/jit/csrc/moe/route_radix.cuh (SGLang PR #32541)
 *
 * Replaces the SGLang `sgl_kernel/` device-side dependencies by reusing the
 * already-migrated headers under
 *   csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/
 * and inlining the route_radix-specific device helpers.
 *
 * Copyright 2023-2024 SGLang Team
 * Licensed under the Apache License, Version 2.0
 */
#pragma once

#include "sgl_kernel/utils.cuh"
#include "sgl_kernel/vec.cuh"
#include "sgl_kernel/type.cuh"
#include "sgl_kernel/warp.cuh"

#include <cstdint>

namespace llm_kernels {
namespace nvidia {
namespace sglang_gate {
namespace route_radix_detail {

inline constexpr float kNanFloor = -1e30f;

// Monotone unsigned key: larger biased -> larger key. Caller must have floored
// biased-NaN. Canonicalizes -0.0 -> +0.0 so equal values get equal keys.
SGL_DEVICE uint32_t biased_to_key(float biased) {
  if (biased == 0.0f) biased = 0.0f;
  uint32_t u = __float_as_uint(biased);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// tl.sigmoid(x) = 1/(1+exp(-x)). Must stay instruction-identical to v1's
// sigmoid_match so both kernels rank (and weight) identically.
SGL_DEVICE float sigmoid_match(float x) {
  return __fdividef(1.0f, 1.0f + __expf(-x));
}

SGL_DEVICE float nan_floor(float x) {
  return (x == x) ? x : kNanFloor;
}

SGL_DEVICE void bar_sync(uint32_t id, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
}

SGL_DEVICE uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

// Exclusive prefix (block-wide, thread-rank order) of `cnt`. Uses
// smem_warp_sum[kNumWarps]; syncs on entry (so the workspace can be reused
// across calls) and before the cross-warp read.
SGL_DEVICE uint32_t block_exclusive_sum(uint32_t cnt, uint32_t lane_id, uint32_t warp_id, uint32_t* smem_warp_sum) {
  const uint32_t inc = warp_inclusive_sum(lane_id, cnt);
  if (lane_id == 31) smem_warp_sum[warp_id] = inc;
  __syncthreads();
  const auto base = __reduce_add_sync(0xFFFFFFFF, lane_id < warp_id ? smem_warp_sum[lane_id] : 0u);
  return base + inc - cnt;
}

}  // namespace route_radix_detail
}  // namespace sglang_gate
}  // namespace nvidia
}  // namespace llm_kernels
