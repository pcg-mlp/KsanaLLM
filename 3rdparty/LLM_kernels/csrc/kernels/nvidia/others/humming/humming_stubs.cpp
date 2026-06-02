/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Stub implementations of humming symbols for non-SM90 platforms.
 * Provides no-op/zero-return implementations so that callers link successfully,
 * but any actual invocation at runtime will log an error and return safely.
 */

#include "csrc/kernels/nvidia/others/humming/jit/humming_jit_compiler.h"
#include "csrc/kernels/nvidia/others/humming/launcher/launcher.h"
#include "csrc/kernels/nvidia/others/humming/launcher/tma.h"
#include "csrc/kernels/nvidia/others/humming/ops/ops.h"
#include "csrc/kernels/nvidia/others/humming/ops/wrapper.h"
#include "csrc/kernels/nvidia/others/humming/preprocess/humming_preprocess.h"

#include <cstdio>

namespace llm_kernels {
namespace nvidia {
namespace humming {

// ============================================================================
// HummingJitCompiler stubs
// ============================================================================

std::string HummingJitCompiler::KernelConfig::CacheKey() const { return "stub"; }

HummingJitCompiler::HummingJitCompiler() {}
HummingJitCompiler::~HummingJitCompiler() {}

void HummingJitCompiler::CompileKernels(int64_t, int64_t, int64_t, ComputeDtype, GemmType) {}
void HummingJitCompiler::CompileKernels(const std::vector<HummingJitCompiler::KernelConfig>&) {}

const KernelMeta* HummingJitCompiler::SelectKernel(int64_t, int64_t, int64_t, GemmType, ComputeDtype) const {
  return nullptr;
}

std::vector<HummingJitCompiler::KernelConfig> HummingJitCompiler::QueryH20Configs(int64_t, int64_t, int64_t,
                                                                                  ComputeDtype, GemmType) {
  return {};
}

// ============================================================================
// Ops stubs
// ============================================================================

static HummingJitCompiler* g_jit_compiler = nullptr;

void SetJitCompiler(HummingJitCompiler* compiler) { g_jit_compiler = compiler; }
HummingJitCompiler* GetJitCompiler() { return g_jit_compiler; }

int32_t get_kernel_block_m(int64_t, int64_t, int64_t, GemmType, ComputeDtype) { return 8; }

void humming_moe_gemm(const void*, const void*, void*, const void*, const void*, const void*, const void*, const void*,
                      const int32_t*, const int32_t*, const int32_t*, const int32_t*, const float*, int32_t*, void*,
                      GemmType, ComputeDtype, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t,
                      int32_t) {
  fprintf(stderr, "ERROR: humming_moe_gemm called on non-SM90 platform — not supported.\n");
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
  fprintf(stderr, "ERROR: pack_weight_fp4 called on non-SM90 platform — not supported.\n");
}

void repack_weight_wgmma(cudaStream_t, const void*, void*, int64_t, int64_t, int64_t, ComputeDtype) {
  fprintf(stderr, "ERROR: repack_weight_wgmma called on non-SM90 platform — not supported.\n");
}

void process_mxfp4_w4a8_weight(cudaStream_t, void*, void*, float*, int64_t, int64_t, int64_t, int64_t) {
  fprintf(stderr, "ERROR: process_mxfp4_w4a8_weight called on non-SM90 platform — not supported.\n");
}

void process_scales(cudaStream_t, const void*, void*, int64_t, int64_t, int64_t, int64_t, ComputeDtype) {
  fprintf(stderr, "ERROR: process_scales called on non-SM90 platform — not supported.\n");
}

// ============================================================================
// Launcher stubs
// ============================================================================

void launch_kernel(const KernelMeta*, const void*, const void*, void*, const void*, const void*, const void*,
                   const void*, const void*, const int32_t*, const int32_t*, const int32_t*, const void*, int32_t*,
                   void*, int64_t, int64_t, int64_t, int64_t, int64_t, bool, cudaStream_t) {
  fprintf(stderr, "ERROR: launch_kernel called on non-SM90 platform — not supported.\n");
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