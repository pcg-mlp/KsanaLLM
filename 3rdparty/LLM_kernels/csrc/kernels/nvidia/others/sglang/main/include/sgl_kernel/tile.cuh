/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/include/sgl_kernel/tile.cuh
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
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/utils.cuh"

#include <cstdint>

namespace device::tile {

template <typename T>
struct Memory {
 public:
  SGL_DEVICE constexpr Memory(uint32_t tid, uint32_t tsize) : tid(tid), tsize(tsize) {}
  SGL_DEVICE static constexpr Memory thread() { return Memory{0, 1}; }
  SGL_DEVICE static Memory warp(int warp_threads = kWarpThreads) {
    return Memory{threadIdx.x % warp_threads, warp_threads};
  }
  SGL_DEVICE static Memory cta(int cta_threads = blockDim.x) { return Memory{threadIdx.x, cta_threads}; }
  SGL_DEVICE T load(const void* ptr, int64_t offset = 0) const {
    return static_cast<const T*>(ptr)[tid + offset * tsize];
  }
  SGL_DEVICE void store(void* ptr, T val, int64_t offset = 0) const {
    static_cast<T*>(ptr)[tid + offset * tsize] = val;
  }
  SGL_DEVICE bool in_bound(int64_t element_count, int64_t offset = 0) const {
    return tid + offset * tsize < element_count;
  }

 private:
  uint32_t tid;
  uint32_t tsize;
};

}  // namespace device::tile
