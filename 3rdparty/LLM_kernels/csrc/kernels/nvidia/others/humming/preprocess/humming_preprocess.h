/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Weight preprocessing for humming MoE GEMM operator.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {
namespace humming {

enum class ComputeDtype : int;

/// Pack FP4 weights into int32 (8 x 4-bit -> 1 x int32).
void pack_weight_fp4(cudaStream_t stream,
                     const void* input, void* output,
                     int64_t num_experts, int64_t size_n, int64_t size_k);

/// Repack weights to WGMMA memory layout.
void repack_weight_wgmma(cudaStream_t stream,
                         const void* input, void* output,
                         int64_t num_experts, int64_t size_n, int64_t size_k,
                         ComputeDtype compute_dtype);

/// Process MXFP4 weights for W4A8 path:
/// 1. Computes per-expert E8M0 scale range, clamps to max_range=12
/// 2. Adjusts FP4 weight values via process_mxfp4_w4a8 kernel
/// 3. Normalizes weight_scale in-place
/// 4. Outputs per-expert global_scale = 2^(scale_min_new - 127)
///
/// @param weight      [E * N * K/8] int32, packed FP4. Modified in-place.
/// @param weight_scale [E * N * K/group_size] uint8, E8M0. Modified in-place (normalized).
/// @param global_scale_output [E] float. Output per-expert global scale.
void process_mxfp4_w4a8_weight(cudaStream_t stream,
                               void* weight,
                               void* weight_scale,
                               float* global_scale_output,
                               int64_t num_experts, int64_t size_n, int64_t size_k,
                               int64_t group_size);

/// Process E8M0 scales.
void process_scales(cudaStream_t stream,
                    const void* input, void* output,
                    int64_t num_experts, int64_t size_n, int64_t size_k,
                    int64_t group_size, ComputeDtype compute_dtype);

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
