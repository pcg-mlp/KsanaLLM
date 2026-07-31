/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/jit/csrc/kimi_k3/mla_output_gate.cuh
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

#include "csrc/kernels/nvidia/others/sglang/main/kimi_k3/mla_output_gate.h"

#include <cuda_bf16.h>
#include <cstdint>

#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/type.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/utils.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/vec.cuh"
#include "csrc/utils/nvidia/assert.h"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

namespace {

// =============================================================================
// Constants
// =============================================================================
constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kVecN = 8;
constexpr uint32_t kVecBytes = kVecN * sizeof(bf16_t);

// =============================================================================
// Kernel parameters struct
// =============================================================================
struct MlaOutputGateParams {
  const bf16_t* __restrict__ x;     // [N] contiguous (flattened [T, H])
  const bf16_t* __restrict__ gate;  // [N] contiguous
  bf16_t* __restrict__ out;         // [N] contiguous
  uint32_t n_vecs;
};

// =============================================================================
// Kernel
// =============================================================================
// One thread handles one 8-element vector, so N must be a multiple of 8 and the
// three pointers must be 16 B aligned.
// Template:  kThreads - block size, must match the launch config
//            kUsePDL  - enable programmatic dependent launch
template <uint32_t kThreads, bool kUsePDL>
__global__ void mla_output_gate_kernel(const MlaOutputGateParams __grid_constant__ params) {
  using namespace device;

  using vec_bf16_t = AlignedVector<bf16_t, kVecN>;

  const uint32_t v = blockIdx.x * kThreads + threadIdx.x;
  if (v >= params.n_vecs) return;

  PDLWaitPrimary<kUsePDL>();

  vec_bf16_t xv, gv, ov;
  xv.load(params.x, v);
  gv.load(params.gate, v);
#pragma unroll
  for (uint32_t i = 0; i < kVecN; ++i) {
    // Match torch.sigmoid(bf16): fp32 sigmoid, round to bf16, then the bf16
    // multiply upcasts both operands to fp32 and rounds once more.
    const float g = cast<fp32_t>(gv[i]);
    const bf16_t s = cast<bf16_t>(1.0f / (1.0f + expf(-g)));
    ov[i] = cast<bf16_t>(cast<fp32_t>(xv[i]) * cast<fp32_t>(s));
  }
  ov.store(params.out, v);

  PDLTriggerSecondary<kUsePDL>();
}

inline bool is_vec_aligned(const void* p) { return reinterpret_cast<uintptr_t>(p) % kVecBytes == 0; }

}  // namespace

// =============================================================================
// Host launcher
// =============================================================================
void mla_output_gate(const void* x, const void* gate, void* out, size_t num_elements, cudaStream_t stream) {
  KLLM_KERNEL_CHECK_WITH_INFO(num_elements % kVecN == 0, "mla_output_gate: num_elements (%zu) must be divisible by %u",
                              num_elements, kVecN);
  KLLM_KERNEL_CHECK_WITH_INFO(is_vec_aligned(x) && is_vec_aligned(gate) && is_vec_aligned(out),
                              "mla_output_gate: x/gate/out must be %u-byte aligned", kVecBytes);
  if (num_elements == 0) return;

  const MlaOutputGateParams params{
      static_cast<const bf16_t*>(x),
      static_cast<const bf16_t*>(gate),
      static_cast<bf16_t*>(out),
      static_cast<uint32_t>(num_elements / kVecN),
  };

  const uint32_t num_blocks = (params.n_vecs + kBlockSize - 1) / kBlockSize;
  constexpr auto kernel = mla_output_gate_kernel<kBlockSize, /*kUsePDL=*/true>;
  LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kBlockSize, 0, stream, params);
}

}  // namespace nvidia
}  // namespace llm_kernels
