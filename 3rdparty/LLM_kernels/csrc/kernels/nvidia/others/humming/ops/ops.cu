/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/others/humming/jit/humming_jit_compiler.h"
#include "csrc/kernels/nvidia/others/humming/launcher/kernel_meta.h"
#include "csrc/kernels/nvidia/others/humming/launcher/launcher.h"
#include "csrc/kernels/nvidia/others/humming/ops/ops.h"
#include "csrc/utils/nvidia/assert.h"

namespace llm_kernels {
namespace nvidia {
namespace humming {

// Global JIT compiler instance — set by HummingFp4MoeLayer::Init
static HummingJitCompiler* g_jit_compiler = nullptr;

void SetJitCompiler(HummingJitCompiler* compiler) { g_jit_compiler = compiler; }

HummingJitCompiler* GetJitCompiler() { return g_jit_compiler; }

void humming_moe_gemm(const void* a, const void* b, void* c, const void* a_scale, const void* b_scale,
                      const void* global_scale, const void* b_zero_point, const void* bias, const int32_t* sorted_ids,
                      const int32_t* expert_ids, const int32_t* num_tokens_padded, const int32_t* expert_layout,
                      const float* topk_weights, int32_t* locks, void* workspace, GemmType gemm_type,
                      ComputeDtype compute_dtype, int64_t size_m, int64_t size_n, int64_t size_k, int64_t num_experts,
                      int64_t top_k, int64_t group_size, cudaStream_t stream, int32_t required_block_m,
                      const KernelMeta* forced_meta) {
  KLLM_KERNEL_CHECK_WITH_INFO(g_jit_compiler != nullptr,
                              "HummingJitCompiler not initialized. Call SetJitCompiler() first.");
  KLLM_KERNEL_CHECK_WITH_INFO(size_m > 0 && size_n > 0 && size_k > 0, "Invalid dimensions M=%ld N=%ld K=%ld",
                              (long)size_m, (long)size_n, (long)size_k);

  const KernelMeta* meta = nullptr;
  if (forced_meta != nullptr) {
    // Profiled selection: use directly
    meta = forced_meta;
  } else if (required_block_m > 0) {
    // Force selection of a kernel with the specified BLOCK_SHAPE_M.
    // This is used when GEMM2 must use the same moe_block_size as GEMM1.
    for (const auto& km : g_jit_compiler->GetKernelTable()) {
      if (km.size_n != size_n || km.size_k != size_k) continue;
      if (km.gemm_type != gemm_type) continue;
      if (km.compute_dtype != compute_dtype) continue;
      if (km.block_m == required_block_m) {
        meta = &km;
        break;
      }
    }
    KLLM_KERNEL_CHECK_WITH_INFO(meta != nullptr,
                                "No humming kernel with block_m=%d for N=%ld K=%ld",
                                required_block_m, (long)size_n, (long)size_k);
  } else {
    meta = g_jit_compiler->SelectKernel(size_m, size_n, size_k, gemm_type, compute_dtype);
    KLLM_KERNEL_CHECK_WITH_INFO(meta != nullptr,
                                "No compiled humming kernel found for N=%ld K=%ld gemm_type=%d compute_dtype=%d",
                                (long)size_n, (long)size_k, (int)gemm_type, (int)compute_dtype);
  }

  launch_kernel(meta, a, b, c, a_scale, b_scale, global_scale, b_zero_point, bias, sorted_ids, expert_ids,
                num_tokens_padded, static_cast<const void*>(expert_layout), locks, workspace,
                size_m, size_n, size_k, num_experts, top_k, false, stream);
}

int32_t get_kernel_block_m(int64_t size_m, int64_t size_n, int64_t size_k,
                           GemmType gemm_type, ComputeDtype compute_dtype,
                           const KernelMeta** out_meta) {
  if (g_jit_compiler == nullptr) {
    if (out_meta) *out_meta = nullptr;
    return 8;
  }
  const KernelMeta* meta = g_jit_compiler->SelectKernel(size_m, size_n, size_k, gemm_type, compute_dtype);
  if (meta == nullptr) {
    if (out_meta) *out_meta = nullptr;
    return 8;
  }
  if (out_meta) *out_meta = meta;
  return static_cast<int32_t>(meta->block_m);
}

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
