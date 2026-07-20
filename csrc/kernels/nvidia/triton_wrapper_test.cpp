/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "kernels/nvidia/triton_wrapper.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

namespace ksana_llm {

namespace {

// RAII helper: snapshot an env var on construction and restore it on
// destruction, so mutating it inside a test never leaks into other tests.
class ScopedEnv {
 public:
  explicit ScopedEnv(const char* name) : name_(name) {
    const char* value = std::getenv(name);
    if (value != nullptr) {
      had_value_ = true;
      old_value_ = value;
    }
  }

  ~ScopedEnv() {
    if (had_value_) {
      setenv(name_.c_str(), old_value_.c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  void Set(const std::string& value) { setenv(name_.c_str(), value.c_str(), /*overwrite=*/1); }
  void Unset() { unsetenv(name_.c_str()); }

 private:
  std::string name_;
  bool had_value_ = false;
  std::string old_value_;
};

}  // namespace

// Covers TritonWrapper::InitKernelsDir()'s single-directory selection by
// priority: KSANA_TRITON_KERNEL_PATH > ${HOME}/.cache > relative fallback.
class TritonWrapperInitKernelsDirTest : public testing::Test {
 protected:
  ScopedEnv triton_path_env_{"KSANA_TRITON_KERNEL_PATH"};
  ScopedEnv home_env_{"HOME"};
};

// Priority 1: an explicit KSANA_TRITON_KERNEL_PATH override always wins, and
// the "/triton_kernel_files" suffix is appended (no trailing slash).
TEST_F(TritonWrapperInitKernelsDirTest, PrefersExplicitEnvOverride) {
  triton_path_env_.Set("/opt/custom/ksana");
  home_env_.Set("/home/tester");

  TritonWrapper wrapper;
  EXPECT_EQ(wrapper.GetKernelsDir(), "/opt/custom/ksana/triton_kernel_files");
}

// The env override wins even when HOME is unset (it never reaches the cache).
TEST_F(TritonWrapperInitKernelsDirTest, EnvOverrideWinsWithoutHome) {
  triton_path_env_.Set("/opt/custom/ksana");
  home_env_.Unset();

  TritonWrapper wrapper;
  EXPECT_EQ(wrapper.GetKernelsDir(), "/opt/custom/ksana/triton_kernel_files");
}

// Priority 2: with no explicit override, fall back to ${HOME}/.cache.
TEST_F(TritonWrapperInitKernelsDirTest, FallsBackToHomeCache) {
  triton_path_env_.Unset();
  home_env_.Set("/home/tester");

  TritonWrapper wrapper;
  EXPECT_EQ(wrapper.GetKernelsDir(), "/home/tester/.cache/KsanaLLM/triton_kernel_files");
}

// Priority 3: with neither env var available, use the relative fallback.
TEST_F(TritonWrapperInitKernelsDirTest, FallsBackToRelativePathWithoutHome) {
  triton_path_env_.Unset();
  home_env_.Unset();

  TritonWrapper wrapper;
  EXPECT_EQ(wrapper.GetKernelsDir(), "./triton_kernel_files");
}

// InitKernelsDir() is idempotent w.r.t. the current environment: calling it
// again re-resolves against whatever the env currently says.
TEST_F(TritonWrapperInitKernelsDirTest, ReResolvesOnSubsequentCall) {
  triton_path_env_.Unset();
  home_env_.Set("/home/tester");

  TritonWrapper wrapper;
  EXPECT_EQ(wrapper.GetKernelsDir(), "/home/tester/.cache/KsanaLLM/triton_kernel_files");

  triton_path_env_.Set("/opt/custom/ksana");
  wrapper.InitKernelsDir();
  EXPECT_EQ(wrapper.GetKernelsDir(), "/opt/custom/ksana/triton_kernel_files");
}

// SetKernelsDir() overrides whatever InitKernelsDir() picked.
TEST_F(TritonWrapperInitKernelsDirTest, SetKernelsDirOverrides) {
  triton_path_env_.Unset();
  home_env_.Set("/home/tester");

  TritonWrapper wrapper;
  wrapper.SetKernelsDir("/explicit/dir");
  EXPECT_EQ(wrapper.GetKernelsDir(), "/explicit/dir");
}

}  // namespace ksana_llm
