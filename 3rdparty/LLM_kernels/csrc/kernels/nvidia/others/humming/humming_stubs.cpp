/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Stub implementations of humming symbols for non-SM90 platforms.
 * Provides link-time placeholders with neutral no-op/zero-return semantics
 * for harmless queries; any invocation of a real humming entry point is a
 * programming error on non-SM90 platforms and fails fast by throwing
 * KllmException (with file:line).
 */

#include "csrc/kernels/nvidia/others/humming/jit/humming_jit_compiler.h"
#include "csrc/kernels/nvidia/others/humming/launcher/launcher.h"
#include "csrc/kernels/nvidia/others/humming/launcher/tma.h"
#include "csrc/kernels/nvidia/others/humming/ops/ops.h"
#include "csrc/kernels/nvidia/others/humming/ops/wrapper.h"
#include "csrc/kernels/nvidia/others/humming/preprocess/humming_preprocess.h"
#include "csrc/utils/nvidia/assert.h"

namespace llm_kernels {
namespace nvidia {
namespace humming {

// ============================================================================
// Shared cache-root resolution stub
// ============================================================================

std::string GetKernelCacheRootDir() {
  KLLM_KERNEL_THROW("GetKernelCacheRootDir called on non-SM90 platform — not supported.");
}

// ============================================================================
// HummingJitCompiler stubs
// ============================================================================

HummingJitCompiler::HummingJitCompiler() {}
HummingJitCompiler::~HummingJitCompiler() {}

void HummingJitCompiler::CompileKernels(int64_t, int64_t, int64_t, ComputeDtype, GemmType, bool) {}
void HummingJitCompiler::CompileKernels(const std::vector<HummingJitCompiler::KernelConfig>&, bool) {}

const KernelMeta* HummingJitCompiler::SelectKernel(int64_t, int64_t, int64_t, GemmType, ComputeDtype) const {
  return nullptr;
}

std::vector<HummingJitCompiler::KernelConfig> HummingJitCompiler::QueryKernelConfigs(int64_t, int64_t, int64_t,
                                                                                     ComputeDtype, GemmType) {
  return {};
}

// ============================================================================
// Ops stubs
// ============================================================================

static HummingJitCompiler* g_jit_compiler = nullptr;

void SetJitCompiler(HummingJitCompiler* compiler) { g_jit_compiler = compiler; }
HummingJitCompiler* GetJitCompiler() { return g_jit_compiler; }

int32_t get_kernel_block_m(int64_t, int64_t, int64_t, GemmType, ComputeDtype, const KernelMeta**) { return 8; }

void humming_moe_gemm(const void*, const void*, void*, const void*, const void*, const void*, const void*, const void*,
                      const int32_t*, const int32_t*, const int32_t*, const int32_t*, const float*, int32_t*, void*,
                      GemmType, ComputeDtype, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t,
                      int32_t, const KernelMeta*) {
  KLLM_KERNEL_THROW("humming_moe_gemm called on non-SM90 platform — not supported.");
}

// ============================================================================
// Wrapper (workspace / locks sizing) stubs
// ============================================================================

int64_t get_locks_size(int64_t, int64_t, int64_t, ComputeDtype) { return 0; }
int64_t get_workspace_size(int64_t, int64_t, int64_t, int64_t, int64_t, ComputeDtype) { return 0; }

// ============================================================================
// Preprocess stubs
// ============================================================================

void pack_weight_fp4(cudaStream_t, const void*, void*, int64_t, int64_t, int64_t) {
  KLLM_KERNEL_THROW("pack_weight_fp4 called on non-SM90 platform — not supported.");
}

void repack_weight_wgmma(cudaStream_t, const void*, void*, int64_t, int64_t, int64_t, ComputeDtype) {
  KLLM_KERNEL_THROW("repack_weight_wgmma called on non-SM90 platform — not supported.");
}

void process_mxfp4_w4a8_weight(cudaStream_t, void*, void*, float*, int64_t, int64_t, int64_t, int64_t) {
  KLLM_KERNEL_THROW("process_mxfp4_w4a8_weight called on non-SM90 platform — not supported.");
}

void process_scales(cudaStream_t, const void*, void*, int64_t, int64_t, int64_t, int64_t, ComputeDtype) {
  KLLM_KERNEL_THROW("process_scales called on non-SM90 platform — not supported.");
}

// ============================================================================
// Launcher stubs
// ============================================================================

void launch_kernel(const KernelMeta*, const void*, const void*, void*, const void*, const void*, const void*,
                   const void*, const void*, const int32_t*, const int32_t*, const int32_t*, const void*, int32_t*,
                   void*, int64_t, int64_t, int64_t, int64_t, int64_t, bool, cudaStream_t) {
  KLLM_KERNEL_THROW("launch_kernel called on non-SM90 platform — not supported.");
}

// ============================================================================
// TMA stubs
// ============================================================================

CUtensorMapSwizzle GetSwizzleEnum(uint32_t) { return CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE; }
CUtensorMapDataType GetTmaDataType(uint32_t) { return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8; }
void BuildTmaDescriptor(CUtensorMap*, const void*, CUtensorMapDataType, uint32_t, const uint64_t*, const uint64_t*,
                        const uint32_t*, uint32_t) {}

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels