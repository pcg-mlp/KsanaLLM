/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 * Kernel launch function for humming MoE GEMM.
 */
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/kernels/nvidia/others/humming/launcher/kernel_meta.h"

namespace llm_kernels {
namespace nvidia {
namespace humming {

/// Build TMA descriptors and launch the kernel via cuLaunchKernelEx.
///
/// This function:
/// 1. Creates TMA descriptors for operands that use TMA (B, BS always; A, C conditional)
/// 2. Packs 17 kernel arguments matching the humming kernel interface
/// 3. Configures grid (stream-K: num_ctas_per_sm * SM_count) and shared memory
/// 4. Launches via cuLaunchKernelEx
///
/// @param meta             Kernel metadata (function, tile sizes, flags, etc.)
/// @param a                Activation tensor [M, K] (FP8 or BF16)
/// @param b                Weight tensor [num_experts, N, K_packed] (FP4 packed as int32)
/// @param c                Output tensor [M, N] (BF16)
/// @param a_scale          Activation scale (per-token)
/// @param b_scale          Weight scale [num_experts, num_groups, N] (E8M0/uint8)
/// @param global_scale     Global FP4 scale (scalar on device)
/// @param b_zero_point     Weight zero-point (optional, can be nullptr)
/// @param bias             Bias tensor (optional, can be nullptr)
/// @param sorted_ids       MoE routing: sorted token IDs
/// @param expert_ids       MoE routing: expert IDs per CTA
/// @param num_tokens_padded MoE routing: padded token counts per expert
/// @param expert_layout    MoE routing: expert layout (int32 or int64)
/// @param locks            Stream-K synchronization locks
/// @param workspace        Workspace for tensor_map_buffer (num_ctas * 128 bytes)
/// @param size_m           Problem M dimension
/// @param size_n           Problem N dimension
/// @param size_k           Problem K dimension
/// @param num_experts      Number of experts
/// @param top_k            MoE top-K routing
/// @param stream           CUDA stream
void launch_kernel(const KernelMeta* meta,
                   const void* a, const void* b, void* c,
                   const void* a_scale, const void* b_scale,
                   const void* global_scale, const void* b_zero_point,
                   const void* bias,
                   const int32_t* sorted_ids, const int32_t* expert_ids,
                   const int32_t* num_tokens_padded, const void* expert_layout,
                   int32_t* locks,
                   void* workspace,
                   int64_t size_m, int64_t size_n, int64_t size_k,
                   int64_t num_experts, int64_t top_k,
                   bool use_int64_expert_layout,
                   cudaStream_t stream);

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
