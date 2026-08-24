/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// Opt-in end-to-end test: exercises the real cold-start path
// (query-configs python -> ONE batch-generate python -> parallel nvcc ->
// cuModuleLoad).  Requires python3 + humming package + nvcc, takes ~1-2 min.
//
//   HUMMING_JIT_E2E=1 ctest -R llm_kernels_nvidia_kernel_humming_jit_test
#include <cuda.h>
#include <dirent.h>
#include <unistd.h>

#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/others/humming/jit/humming_jit_compiler.h"

using llm_kernels::nvidia::humming::ComputeDtype;
using llm_kernels::nvidia::humming::GemmType;
using llm_kernels::nvidia::humming::HummingJitCompiler;

class HummingJitCompilerE2eTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (std::getenv("HUMMING_JIT_E2E") == nullptr) {
      GTEST_SKIP() << "set HUMMING_JIT_E2E=1 to run the cold-start E2E test";
    }
    ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
    CUdevice dev;
    ASSERT_EQ(cuDeviceGet(&dev, 0), CUDA_SUCCESS);
    CUcontext ctx;
    ASSERT_EQ(cuDevicePrimaryCtxRetain(&ctx, dev), CUDA_SUCCESS);
    ASSERT_EQ(cuCtxSetCurrent(ctx), CUDA_SUCCESS);
  }
};

TEST_F(HummingJitCompilerE2eTest, ColdCompileHeuristicOnly) {
  // Fresh cache dir -> full cold path.  compile_extend_configs=false keeps the
  // run at heuristic-only (~6 configs, ~1 min total).
  std::string cache_root = "/tmp/humming_jit_e2e_cache_" + std::to_string(getpid());
  setenv("KSANA_HUMMING_KERNEL_PATH", cache_root.c_str(), 1 /* before construction */);
  setenv("HUMMING_JIT_LOG", "1", 1);

  HummingJitCompiler compiler;
  compiler.CompileKernels(1024, 4096, 256, ComputeDtype::FP8, GemmType::INDEXED,
                          /*compile_extend_configs=*/false);

  const auto& table = compiler.GetKernelTable();
  ASSERT_GE(table.size(), 1u);
  for (const auto& meta : table) {
    EXPECT_EQ(meta.size_n, 1024);
    EXPECT_EQ(meta.size_k, 4096);
    EXPECT_NE(meta.func, nullptr);
  }
  EXPECT_NE(compiler.SelectKernel(64, 1024, 4096, GemmType::INDEXED, ComputeDtype::FP8), nullptr);

  // No leftover batch manifests in the cache root.
  DIR* dir = opendir(cache_root.c_str());
  ASSERT_NE(dir, nullptr);
  struct dirent* ent;
  bool leftover = false;
  while ((ent = readdir(dir)) != nullptr) {
    if (std::string(ent->d_name).find(".batch_manifest_") != std::string::npos) leftover = true;
  }
  closedir(dir);
  EXPECT_FALSE(leftover);
}
