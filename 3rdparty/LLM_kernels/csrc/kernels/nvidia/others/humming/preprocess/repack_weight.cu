/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Repack FP4 weights to WGMMA memory layout for humming kernels.
 * Instantiates weight_repack_nk from humming/kernel/process.cuh.
 *
 * Template parameters for W4A16 path:
 *   kNumBitsB=4, kNumBitsA=16, kPackedInput=1,
 *   kShouldPreprocessForINT2FP=0, kShouldPreprocessWithZP=0,
 *   kShouldTransposeMiniBlock=1, kGroupSizeZP=0
 *
 * Template parameters for W4A8 path:
 *   kNumBitsB=4, kNumBitsA=8, kPackedInput=1,
 *   kShouldPreprocessForINT2FP=0, kShouldPreprocessWithZP=0,
 *   kShouldTransposeMiniBlock=0, kGroupSizeZP=0
 *
 * NOTE: kShouldTransposeMiniBlock differs between W4A16 (=1) and W4A8 (=0).
 * Matches humming reference: should_transpose_mini_block = use_wgmma && !use_fused_e8m0_scale.
 * For W4A8 with fused E8M0 scale, the mini-block transpose must be disabled.
 */

#include <cassert>
#include <humming/kernel/process.cuh>

#include "csrc/kernels/nvidia/others/humming/launcher/kernel_meta.h"
#include "humming_preprocess.h"

namespace llm_kernels {
namespace nvidia {
namespace humming {

void repack_weight_wgmma(cudaStream_t stream, const void* input, void* output, int64_t num_experts, int64_t size_n,
                         int64_t size_k, ComputeDtype compute_dtype) {
  // Input: [E, N, K*4/32] uint32 (FP4 packed, kPackedInput=1)
  //   gmem_stride (uint32 columns) = CEIL_DIV(size_k * 4, 32) = size_k / 8
  // Grid: each block processes a 64-row tile in N dimension
  //   gridDimX = ceil(size_n / 64)
  //   gridDimY = ceil(gmem_stride / smem_stride)
  //     where smem_stride = 64 * kNumBitsInputB / 32 = 64 * 4 / 32 = 8
  //   gridDimZ = num_experts

  // For packed FP4: kNumBitsInputB = 4, smem_stride = 64*4/32 = 8
  int64_t gmem_stride = (size_k * 4 + 31) / 32;  // K/8 for FP4 packed
  constexpr int64_t smem_stride = 8;             // 64 * 4 / 32

  dim3 grid(static_cast<unsigned>((size_n + 63) / 64),
            static_cast<unsigned>((gmem_stride + smem_stride - 1) / smem_stride), static_cast<unsigned>(num_experts));
  dim3 block(32);

  // interleave_mode matches humming layer.py:
  //   W4A16 (BF16): 3 -> interleaved index for both low bits and sign bit
  //   W4A8 (FP8 fused E8M0): 2 -> sequential index for low bits, interleaved for sign bit
  if (compute_dtype == ComputeDtype::BF16) {
    weight_repack_nk<4, 16, 1, 0, 0, 1, 0>
        <<<grid, block, 0, stream>>>(reinterpret_cast<const uint32_t*>(input), reinterpret_cast<uint32_t*>(output),
                                     nullptr, static_cast<uint32_t>(size_n), static_cast<uint32_t>(size_k),
                                     static_cast<uint32_t>(size_n), static_cast<uint32_t>(size_k),
                                     /*interleave_mode=*/3u);
  } else {
    // W4A8 path: kNumBitsA=8, kShouldTransposeMiniBlock=0
    // (use_fused_e8m0_scale disables the mini-block transpose; matches humming reference)
    weight_repack_nk<4, 8, 1, 0, 0, 0, 0>
        <<<grid, block, 0, stream>>>(reinterpret_cast<const uint32_t*>(input), reinterpret_cast<uint32_t*>(output),
                                     nullptr, static_cast<uint32_t>(size_n), static_cast<uint32_t>(size_k),
                                     static_cast<uint32_t>(size_n), static_cast<uint32_t>(size_k),
                                     /*interleave_mode=*/2u);
  }
}

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
