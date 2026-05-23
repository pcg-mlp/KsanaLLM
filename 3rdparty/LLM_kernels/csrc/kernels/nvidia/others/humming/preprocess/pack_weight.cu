/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * FP4 weight packing: 8 x 4-bit values -> 1 x int32.
 * Instantiates pack_weight_kernel<4> from humming/kernel/pack_weight.cuh.
 */

#include <humming/kernel/pack_weight.cuh>

#include "humming_preprocess.h"

namespace llm_kernels {
namespace nvidia {
namespace humming {

void pack_weight_fp4(cudaStream_t stream, const void* input, void* output,
                     int64_t num_experts, int64_t size_n, int64_t size_k) {
  // Input: [num_experts * size_n * size_k] int32 values (low 4 bits each)
  // Output: [num_experts * size_n * size_k / 8] int32 packed values
  // Kernel: each block processes 32 threads x 32 values = 1024 values
  //         output: 32 threads x 4 ints (kNumBits=4) = 128 ints
  int64_t total_elements = num_experts * size_n * size_k;
  int64_t num_blocks = total_elements / (32 * 32);

  pack_weight_kernel<4><<<num_blocks, 32, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(input),
      reinterpret_cast<uint32_t*>(output));
}

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
