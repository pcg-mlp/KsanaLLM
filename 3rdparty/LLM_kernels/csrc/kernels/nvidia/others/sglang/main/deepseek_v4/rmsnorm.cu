/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/rmsnorm.cuh
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

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/math.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/tile.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/type.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/utils.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/vec.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/warp.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {


constexpr uint32_t kRmsNormBlockSize = 128;
constexpr uint32_t kRmsNormNumWarps = kRmsNormBlockSize / device::kWarpThreads;

struct RMSNormSelfParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  int64_t stride_batch_bytes;
  int64_t stride_head_bytes;
  uint32_t batch_size;
  uint32_t num_head;
  float eps;
};

// Per-head RMSNorm operating on a strided (non-contiguous) input tensor.
// No learnable weight is applied (identity weight).
// Each warp handles one (batch_id, head_id) pair.
template <typename DType, int64_t kHeadDim, bool kUsePDL>
__global__ __launch_bounds__(kRmsNormBlockSize, 20)  //
    void rmsnorm_self_kernel(const __grid_constant__ RMSNormSelfParams params) {
  using namespace device;
  // kVecSize: prefer 128-bit loads (16/sizeof), but cap at kHeadDim/kWarpThreads
  // so that kNumLoop >= 1 for all supported head_dim/dtype combinations.
  // Example: bf16 (2B) + head_dim=128 → kVecSize=4 instead of 8, kNumLoop=1.
  static constexpr int64_t kMaxVec = 16 / sizeof(DType);
  static constexpr int64_t kMaxByDim = kHeadDim / kWarpThreads;
  constexpr int64_t kVecSize = kMaxVec < kMaxByDim ? kMaxVec : kMaxByDim;
  constexpr uint32_t kNumLoop = kHeadDim / (kVecSize * kWarpThreads);
  static_assert(kHeadDim % (kWarpThreads * kVecSize) == 0);
  using DType2 = packed_t<DType>;
  using Vec = AlignedVector<DType2, kVecSize / 2>;

  const auto warp_id = blockIdx.x * kRmsNormNumWarps + threadIdx.x / kWarpThreads;
  const auto batch_id = warp_id / params.num_head;
  const auto head_id = warp_id % params.num_head;
  const auto gmem = tile::Memory<Vec>::warp();
  if (batch_id >= params.batch_size) return;
  const auto input_ptr = pointer::offset(  //
      params.input, batch_id * params.stride_batch_bytes, head_id * params.stride_head_bytes);
  // use contiguous layout
  const auto output_ptr = pointer::offset(  //
      params.output, warp_id * kHeadDim * sizeof(DType));
  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  Vec inputs[kNumLoop];
#pragma unroll
  for (uint32_t i = 0; i < kNumLoop; ++i) {
    inputs[i] = gmem.load(input_ptr, i);
  }

  // compute sum of squares
  float local_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < kNumLoop; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      const auto [x, y] = cast<fp32x2_t>(inputs[i][j]);
      local_sum += x * x + y * y;
    }
  }

  const auto sum_of_squares = warp::reduce_sum(local_sum);
  const auto factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

  // weight must be identity (null, not used)
#pragma unroll
  for (uint32_t i = 0; i < kNumLoop; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      const auto [x, y] = cast<fp32x2_t>(inputs[i][j]);
      inputs[i][j] = cast<DType2>(fp32x2_t{x * factor, y * factor});
    }
    gmem.store(output_ptr, inputs[i], i);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
// Input  layout: [batch_size, num_head, head_dim] with arbitrary strides
//                (stride_batch_bytes and stride_head_bytes in bytes).
// Output layout: contiguous [batch_size * num_head, head_dim].
//
// Template parameters:
//   DType    - element type: __nv_bfloat16 or float
//   kHeadDim - compile-time head dimension (must satisfy kHeadDim % 64 == 0)
template <typename DType, int64_t kHeadDim>
inline void rmsnorm_self(const void* input, void* output, int64_t stride_batch_bytes, int64_t stride_head_bytes,
                         uint32_t batch_size, uint32_t num_head, float eps, cudaStream_t stream) {
  if (batch_size == 0 || num_head == 0) return;

  const uint32_t needed_warps = batch_size * num_head;
  const uint32_t num_blocks = device::div_ceil(needed_warps, kRmsNormNumWarps);

  const RMSNormSelfParams params{input, output, stride_batch_bytes, stride_head_bytes, batch_size, num_head, eps};

  constexpr auto kernel = rmsnorm_self_kernel<DType, kHeadDim, /*kUsePDL=*/true>;
  LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kRmsNormBlockSize, 0, stream, params);
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
#define INSTANTIATE_RMSNORM(DType, kHeadDim)                                                                   \
  template void rmsnorm_self<DType, kHeadDim>(const void*, void*, int64_t, int64_t, uint32_t, uint32_t, float, \
                                              cudaStream_t);

INSTANTIATE_RMSNORM(__nv_bfloat16, 128)
INSTANTIATE_RMSNORM(__nv_bfloat16, 256)
INSTANTIATE_RMSNORM(__nv_bfloat16, 512)
INSTANTIATE_RMSNORM(half, 128)
INSTANTIATE_RMSNORM(half, 256)
INSTANTIATE_RMSNORM(half, 512)
INSTANTIATE_RMSNORM(float, 128)
INSTANTIATE_RMSNORM(float, 256)
INSTANTIATE_RMSNORM(float, 512)

#undef INSTANTIATE_RMSNORM

}  // namespace nvidia
}  // namespace llm_kernels
