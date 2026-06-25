/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Humming MoE GEMM operator — public dispatch interface.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "csrc/kernels/nvidia/others/humming/launcher/kernel_meta.h"

namespace llm_kernels {
namespace nvidia {
namespace humming {

class HummingJitCompiler;

/// Set the global JIT compiler instance used by humming_moe_gemm.
/// Must be called before any Forward call.
void SetJitCompiler(HummingJitCompiler* compiler);

/// Get the global JIT compiler instance.
HummingJitCompiler* GetJitCompiler();

/// Query the BLOCK_SHAPE_M of the kernel that would be selected for a given problem size.
/// Returns 8 if no JIT compiler is set or no kernel is found.
/// If out_meta != nullptr, writes the selected KernelMeta* to *out_meta (or nullptr if none).
int32_t get_kernel_block_m(int64_t size_m, int64_t size_n, int64_t size_k, GemmType gemm_type,
                           ComputeDtype compute_dtype, const KernelMeta** out_meta = nullptr);

/// Core MoE Humming GEMM dispatch function.
/// All buffers must be pre-allocated on GPU by the caller.
/// If forced_meta != nullptr, uses it directly (profiled selection).
/// If required_block_m > 0, forces selection of a kernel with that specific BLOCK_SHAPE_M.
void humming_moe_gemm(const void* a, const void* b, void* c, const void* a_scale, const void* b_scale,
                      const void* global_scale, const void* b_zero_point, const void* bias, const int32_t* sorted_ids,
                      const int32_t* expert_ids, const int32_t* num_tokens_padded, const int32_t* expert_layout,
                      // Currently unused by launch_kernel — weight application is done by InvokeMoeFusedMulSum.
                      const float* topk_weights, int32_t* locks, void* workspace, GemmType gemm_type,
                      ComputeDtype compute_dtype, int64_t size_m, int64_t size_n, int64_t size_k, int64_t num_experts,
                      int64_t top_k, int64_t group_size, cudaStream_t stream, int32_t required_block_m = 0,
                      const KernelMeta* forced_meta = nullptr);

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
