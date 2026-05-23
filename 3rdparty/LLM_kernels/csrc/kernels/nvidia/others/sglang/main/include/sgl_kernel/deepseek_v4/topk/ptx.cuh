/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/4807f6c5220c8ac53e395d6d2ed493cda241bc8e/python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk/ptx.cuh
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

// Ksana-local copy of <sgl_kernel/deepseek_v4/topk/ptx.cuh>.
//
// Modifications vs. the original:
//   * Replaced `#include <sgl_kernel/utils.cuh>` with the local compat header.
//   * No other changes; all device::top512::ptx wrappers are kept verbatim.
//   * Requires CUDA >= 12 for `<cuda/ptx>`.

#pragma once

#include "sgl_kernel/deepseek_v4/topk/utils_compat.cuh"

#include <cstdint>

namespace device::top512 {

namespace ptx {

SGL_DEVICE void mbarrier_wait(uint64_t* addr, uint32_t phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  uint32_t waitComplete;
  do {
    asm volatile(
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64  P_OUT, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(waitComplete)
        : "r"(smem), "r"(phase)
        : "memory");
  } while (!waitComplete);
#endif
}

SGL_DEVICE void mbarrier_init(uint64_t* addr, uint32_t arrives) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(smem), "r"(arrives) : "memory");
#endif
}

SGL_DEVICE void mbarrier_arrive_expect_tx(uint64_t* addr, uint32_t tx) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(smem), "r"(tx) : "memory");
#endif
}

SGL_DEVICE void mbarrier_arrive(uint64_t* addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem) : "memory");
#endif
}

SGL_DEVICE void tma_load(void* dst, const void* src, uint32_t num_bytes, uint64_t* mbar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t dst_smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint32_t mbar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst_smem),
               "l"(src), "r"(num_bytes), "r"(mbar_smem)
               : "memory");
#endif
}

SGL_DEVICE uint32_t elect_sync() {
  uint32_t pred = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
#endif
  return pred;
}

SGL_DEVICE bool elect_sync_cta(uint32_t tx) {
  const auto warp_id = tx / 32;
  const auto uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
  return (uniform_warp_id == 0 && elect_sync());
}

}  // namespace ptx

}  // namespace device::top512
