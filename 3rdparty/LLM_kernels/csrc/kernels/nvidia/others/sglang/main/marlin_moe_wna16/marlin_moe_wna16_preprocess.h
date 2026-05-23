/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Weight preprocessing utilities for Marlin MoE WNA16.
 * Transforms raw FP4 weights into Marlin tile layout.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

namespace llm_kernels {
namespace nvidia {
namespace marlin_moe_wna16 {

/// Repack quantized weights from standard GPTQ layout into Marlin tile layout.
/// Operates on ALL experts in a single kernel launch (expert dim is gridDim.y).
void marlin_moe_repack(const uint32_t* b_q_weight_ptr, uint32_t* out_ptr,
                       int64_t num_experts, int64_t size_k, int64_t size_n,
                       int64_t num_bits, int rank, cudaStream_t stream);

/// Calculate the output shape after repack: {size_k / 16, size_n * 16 / pack_factor}
std::vector<int64_t> marlin_moe_repack_meta(int64_t size_k, int64_t size_n, int64_t num_bits);

/// Permute quantization scales to match Marlin's expected layout.
template <typename T>
void marlin_moe_permute_scales(cudaStream_t stream, const T* input, T* output,
                               int64_t num_experts, size_t k, size_t n, int64_t groupsize);

/// Process MXFP4 scales for Marlin: [0,2,1,3] swizzle + add 6 to E8M0 exponent.
/// Operates in-place on uint8 E8M0 scale data.
void mxfp4_process_scales(cudaStream_t stream, uint8_t* data, int64_t num_elements);

/// Batched 2D transpose: [num_batches, rows, cols] → [num_batches, cols, rows]
/// element_size is bytes per element (1 for uint8/E8M0, 4 for uint32/int32).
void transpose_2d_batched(cudaStream_t stream, const void* input, void* output,
                          int64_t num_batches, int64_t rows, int64_t cols,
                          int element_size);

}  // namespace marlin_moe_wna16
}  // namespace nvidia
}  // namespace llm_kernels
