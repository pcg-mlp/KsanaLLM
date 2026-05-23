/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/rope.cuh
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

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/rope.h"

#include <cuda_bf16.h>

#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/type.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/utils.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/vec.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {


// =============================================================================
// Constants
// =============================================================================
using RopeDType = bf16_t;
static constexpr int64_t kRopeDim = 64;
static constexpr uint32_t kRopeBlockSize = 128;
static constexpr uint32_t kRopeNumWarps = kRopeBlockSize / device::kWarpThreads;

// =============================================================================
// Kernel parameters struct
// =============================================================================
struct FusedQKRopeParams {
  void* __restrict__ q;
  void* __restrict__ k;
  const void* __restrict__ freqs_cis;
  const void* __restrict__ positions;
  int64_t q_stride_batch;
  int64_t k_stride_batch;
  int64_t q_stride_head;
  int64_t k_stride_head;
  uint32_t num_q_heads;
  uint32_t num_k_heads;
  uint32_t batch_size;
};

// =============================================================================
// Kernel
// =============================================================================
// Each warp handles one (batch_id, head_id) pair.
// Lane i handles complex pair i in the rope region.
// Template:  kUsePDL   - enable programmatic dependent launch
//            kInverse  - reverse rotation direction
//            kIsNeox   - true: NeoX style (split-half layout: first half = real, second half = imag)
//                        false: GPT-J interleaved style (adjacent pairs: [real, imag, real, imag, ...])
template <bool kUsePDL, bool kInverse, bool kIsNeox>
__global__ __launch_bounds__(kRopeBlockSize, 16)  //
    void deepseek_rope_kernel(const __grid_constant__ FusedQKRopeParams param) {
  using namespace device;
  using DType = RopeDType;
  using DType2 = packed_t<DType>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x * kRopeNumWarps + warp_id;

  const auto& [q, k, freqs_cis, positions,                                    //
               q_stride_batch, k_stride_batch, q_stride_head, k_stride_head,  //
               num_q_heads, num_k_heads, batch_size] = param;

  const auto num_total_heads = num_q_heads + num_k_heads;
  const auto head_id = global_warp_id % num_total_heads;
  const auto batch_id = global_warp_id / num_total_heads;
  if (batch_id >= batch_size) return;

  const auto position = static_cast<const int64_t*>(positions)[batch_id];
  const auto is_q = head_id < num_q_heads;
  // k may be null (num_k_heads == 0 or unused path).
  if (!is_q && k == nullptr) return;
  const auto local_head = is_q ? head_id : (head_id - num_q_heads);
  const auto stride_batch = is_q ? q_stride_batch : k_stride_batch;
  const auto stride_head = is_q ? q_stride_head : k_stride_head;
  const auto base_ptr = is_q ? q : k;

  // Load freqs (split layout: [cos_0..cos_{N/2-1}, sin_0..sin_{N/2-1}], stored as DType)
  const auto* freq_base = static_cast<const DType*>(freqs_cis) + position * kRopeDim;
  const float f_cos = cast<float>(freq_base[lane_id]);
  const float f_sin = cast<float>(freq_base[kRopeDim / 2 + lane_id]);
  PDLWaitPrimary<kUsePDL>();

  if constexpr (kIsNeox) {
    // NeoX style: first half = real parts [0..31], second half = imag parts [32..63]
    // Lane i handles pair (element[i], element[i + kRopeDim/2])
    auto* input_dtype =
        static_cast<DType*>(pointer::offset(base_ptr, batch_id * stride_batch, local_head * stride_head));
    const float x_real = cast<float>(input_dtype[lane_id]);
    const float x_imag = cast<float>(input_dtype[kRopeDim / 2 + lane_id]);
    float out_real, out_imag;
    if constexpr (kInverse) {
      out_real = x_real * f_cos + x_imag * f_sin;
      out_imag = x_imag * f_cos - x_real * f_sin;
    } else {
      out_real = x_real * f_cos - x_imag * f_sin;
      out_imag = x_real * f_sin + x_imag * f_cos;
    }
    input_dtype[lane_id] = cast<DType>(out_real);
    input_dtype[kRopeDim / 2 + lane_id] = cast<DType>(out_imag);
  } else {
    // GPT-J interleaved style: adjacent pairs [real0, imag0, real1, imag1, ...]
    const auto input =
        static_cast<DType2*>(pointer::offset(base_ptr, batch_id * stride_batch, local_head * stride_head));
    const auto data = input[lane_id];
    const auto [x_real, x_imag] = cast<fp32x2_t>(data);
    fp32x2_t output;
    if constexpr (kInverse) {
      // (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
      output = {
          x_real * f_cos + x_imag * f_sin,
          x_imag * f_cos - x_real * f_sin,
      };
    } else {
      // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
      output = {
          x_real * f_cos - x_imag * f_sin,
          x_real * f_sin + x_imag * f_cos,
      };
    }
    input[lane_id] = cast<DType2>(output);
  }

  PDLTriggerSecondary<kUsePDL>();
}

// =============================================================================
// Host launcher
// =============================================================================
void fused_qk_rope(void* q, void* k, const void* freqs_cis, const void* positions, int64_t q_stride_batch,
                   int64_t k_stride_batch, int64_t q_stride_head, int64_t k_stride_head, uint32_t num_q_heads,
                   uint32_t num_k_heads, uint32_t batch_size, cudaStream_t stream, bool inverse, bool is_neox) {
  if (batch_size == 0) return;

  const uint32_t num_total_heads = num_q_heads + num_k_heads;
  const uint32_t total_warps = batch_size * num_total_heads;
  const uint32_t num_blocks = device::div_ceil(total_warps, kRopeNumWarps);

  const FusedQKRopeParams params{
      q,           k,           freqs_cis, positions, q_stride_batch, k_stride_batch, q_stride_head, k_stride_head,
      num_q_heads, num_k_heads, batch_size};

  if (is_neox) {
    if (!inverse) {
      constexpr auto kernel = deepseek_rope_kernel</*kUsePDL=*/true, /*kInverse=*/false, /*kIsNeox=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kRopeBlockSize, 0, stream, params);
    } else {
      constexpr auto kernel = deepseek_rope_kernel</*kUsePDL=*/true, /*kInverse=*/true, /*kIsNeox=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kRopeBlockSize, 0, stream, params);
    }
  } else {
    if (!inverse) {
      constexpr auto kernel = deepseek_rope_kernel</*kUsePDL=*/true, /*kInverse=*/false, /*kIsNeox=*/false>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kRopeBlockSize, 0, stream, params);
    } else {
      constexpr auto kernel = deepseek_rope_kernel</*kUsePDL=*/true, /*kInverse=*/true, /*kIsNeox=*/false>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kRopeBlockSize, 0, stream, params);
    }
  }
}

}  // namespace nvidia
}  // namespace llm_kernels
