/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Smoke test for humming JIT compiler infrastructure.
 * Verifies: generate_kernel.py runs, nvcc compiles, cubin loads.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "csrc/kernels/nvidia/others/humming/jit/humming_jit_compiler.h"
#include "csrc/kernels/nvidia/others/humming/launcher/kernel_meta.h"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {
namespace humming {
namespace test {

class HummingJitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    skip_test_ = llm_kernels::utils::GetSMVersion() < 90;
    if (skip_test_) {
      return;
    }

    cudaStreamCreate(&stream_);

    // Determine paths relative to project source
    const char* src_dir = std::getenv("HUMMING_TEST_SOURCE_DIR");
    if (src_dir) {
      base_dir_ = std::string(src_dir);
    } else {
      // Derive humming root from this file's compile-time path: tests/../
      std::string this_file = __FILE__;
      auto pos = this_file.rfind("/tests/");
      if (pos != std::string::npos) {
        base_dir_ = this_file.substr(0, pos);
      } else {
        base_dir_ = "..";
      }
    }
    script_dir_ = base_dir_ + "/jit";
    cache_dir_ = "/tmp/humming_test_cache_" + std::to_string(getpid());

    // Configure the JIT compiler via environment variables
    setenv("HUMMING_BASE_DIR", base_dir_.c_str(), 1);
    setenv("KSANA_HUMMING_KERNEL_PATH", cache_dir_.c_str(), 1);
  }

  void TearDown() override {
    if (skip_test_) {
      return;
    }

    cudaStreamDestroy(stream_);
    unsetenv("HUMMING_BASE_DIR");
    unsetenv("KSANA_HUMMING_KERNEL_PATH");
    std::string cmd = "rm -rf " + cache_dir_;
    system(cmd.c_str());
  }

  cudaStream_t stream_;
  std::string base_dir_;
  std::string script_dir_;
  std::string cache_dir_;
  bool skip_test_ = false;
};

// Test that generate_kernel.py can produce source code
TEST_F(HummingJitTest, GenerateSourceSmoke) {
  if (skip_test_) {
    GTEST_SKIP() << "Skipping: requires SM90 or higher";
  }
  std::string output_cu = cache_dir_ + "/test_gen.cu";
  std::string output_meta = cache_dir_ + "/test_gen_meta.json";

  // Create cache dir
  system(("mkdir -p " + cache_dir_).c_str());

  // Call generate_kernel.py directly
  std::string cmd = "python3 " + script_dir_ +
                    "/generate_kernel.py"
                    " --size_n 128 --size_k 256 --num_experts 4"
                    " --block_m 64 --gemm_type 1 --compute_dtype 1"
                    " --output " +
                    output_cu + " --meta_output " + output_meta;

  int ret = system(cmd.c_str());
  ASSERT_EQ(ret, 0) << "generate_kernel.py failed";

  // Verify output file exists and has content
  FILE* f = fopen(output_cu.c_str(), "r");
  ASSERT_NE(f, nullptr) << "Generated .cu file not found";
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fclose(f);
  EXPECT_GT(size, 1000) << "Generated .cu file too small";
}

// Test full JIT pipeline: generate + compile + load
TEST_F(HummingJitTest, CompileAndLoadSmoke) {
  if (skip_test_) {
    GTEST_SKIP() << "Skipping: requires SM90 or higher";
  }
  HummingJitCompiler compiler;

  // Small test config
  std::vector<HummingJitCompiler::KernelConfig> configs = {
      {128, 256, 4, 64, GemmType::INDEXED, ComputeDtype::FP8},
  };

  // This will call Python and nvcc — may take 30-60s
  ASSERT_NO_THROW(compiler.CompileKernels(configs));

  // Verify kernel is selectable
  const KernelMeta* meta = compiler.SelectKernel(16, 128, 256, GemmType::INDEXED, ComputeDtype::FP8);
  ASSERT_NE(meta, nullptr) << "SelectKernel returned nullptr";
  EXPECT_EQ(meta->size_n, 128);
  EXPECT_EQ(meta->size_k, 256);
  EXPECT_NE(meta->func, nullptr);
  EXPECT_GT(meta->smem_size, 0);
  EXPECT_GT(meta->num_threads, 0);
}

// Test cache hit (second compile should be fast)
TEST_F(HummingJitTest, CacheHit) {
  if (skip_test_) {
    GTEST_SKIP() << "Skipping: requires SM90 or higher";
  }
  std::vector<HummingJitCompiler::KernelConfig> configs = {
      {128, 256, 4, 64, GemmType::INDEXED, ComputeDtype::FP8},
  };

  // First compile (slow)
  HummingJitCompiler compiler1;
  compiler1.CompileKernels(configs);

  // Second compile (should hit cache, fast)
  HummingJitCompiler compiler2;
  compiler2.CompileKernels(configs);

  const KernelMeta* meta = compiler2.SelectKernel(16, 128, 256, GemmType::INDEXED, ComputeDtype::FP8);
  ASSERT_NE(meta, nullptr);
}

}  // namespace test
}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
