/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/include/sgl_kernel/warp.cuh
 *
 * Copyright 2023-2024 SGLang Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "sgl_kernel/math.cuh"
#include "sgl_kernel/utils.cuh"

// Some warp primitives
namespace device::warp {

static constexpr uint32_t kFullMask = 0xffffffffu;

template <uint32_t kThreads = kWarpThreads, typename T>
SGL_DEVICE T reduce_sum(T value, uint32_t active_mask = kFullMask) {
#pragma unroll
  for (auto mask = kThreads >> 1; mask > 0; mask >>= 1) value = value + __shfl_xor_sync(active_mask, value, mask, 32);
  return value;
}

template <uint32_t kThreads = kWarpThreads, typename T>
SGL_DEVICE T reduce_max(T value, uint32_t active_mask = kFullMask) {
#pragma unroll
  for (auto mask = kThreads >> 1; mask > 0; mask >>= 1)
    value = math::max(value, __shfl_xor_sync(active_mask, value, mask, 32));
  return value;
}

}  // namespace device::warp
