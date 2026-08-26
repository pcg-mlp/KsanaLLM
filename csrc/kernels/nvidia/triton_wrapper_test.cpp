/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "kernels/nvidia/triton_wrapper.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ksana_llm {

namespace {

namespace fs = std::filesystem;

void WriteEmptyFile(const fs::path& path) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path);
  out << "{}";
}

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

class ScopedCwd {
 public:
  explicit ScopedCwd(const fs::path& path) : old_(fs::current_path()) { fs::current_path(path); }
  ~ScopedCwd() { fs::current_path(old_); }

 private:
  fs::path old_;
};

}  // namespace

// Covers TritonWrapper::InitKernelsDir()'s single-directory selection by
// priority: KSANA_TRITON_KERNEL_PATH > cmake-baked absolute TRITON_KERNEL_PATH.
class TritonWrapperInitKernelsDirTest : public testing::Test {
 protected:
  ScopedEnv triton_path_env_{"KSANA_TRITON_KERNEL_PATH"};

  fs::path MakeIsolatedCwd() {
    fs::path dir = fs::path(testing::TempDir()) / "ksana_triton_cwd" /
                   ::testing::UnitTest::GetInstance()->current_test_info()->name() / "cwd";
    fs::remove_all(dir.parent_path());
    fs::create_directories(dir);
    return dir;
  }
};

TEST_F(TritonWrapperInitKernelsDirTest, PrefersExplicitEnvOverride) {
  const fs::path cwd = MakeIsolatedCwd();
  fs::create_directories(cwd.parent_path() / "triton_kernel_files");
  ScopedCwd in_cwd(cwd);
  triton_path_env_.Set("/opt/custom/ksana");

  TritonWrapper wrapper;
  EXPECT_EQ(wrapper.GetKernelsDir(), "/opt/custom/ksana/triton_kernel_files");
}

TEST_F(TritonWrapperInitKernelsDirTest, FallsBackToCompiledDefaultWhenEnvUnset) {
  ScopedCwd in_cwd(MakeIsolatedCwd());
  triton_path_env_.Unset();

  TritonWrapper wrapper;
  EXPECT_EQ(wrapper.GetKernelsDir(), TRITON_KERNEL_PATH);
}

TEST_F(TritonWrapperInitKernelsDirTest, EmptyEnvFallsBackToCompiledDefault) {
  ScopedCwd in_cwd(MakeIsolatedCwd());
  triton_path_env_.Set("");

  TritonWrapper wrapper;
  EXPECT_EQ(wrapper.GetKernelsDir(), TRITON_KERNEL_PATH);
}

TEST_F(TritonWrapperInitKernelsDirTest, CompiledDefaultIsIndependentOfCwd) {
  ScopedCwd in_cwd(MakeIsolatedCwd());
  triton_path_env_.Unset();

  TritonWrapper wrapper;
  EXPECT_EQ(wrapper.GetKernelsDir(), TRITON_KERNEL_PATH);
  EXPECT_NE(wrapper.GetKernelsDir(), "../triton_kernel_files");
}

TEST_F(TritonWrapperInitKernelsDirTest, ReResolvesOnSubsequentCall) {
  ScopedCwd in_cwd(MakeIsolatedCwd());
  triton_path_env_.Unset();

  TritonWrapper wrapper;
  EXPECT_EQ(wrapper.GetKernelsDir(), TRITON_KERNEL_PATH);

  triton_path_env_.Set("/opt/custom/ksana");
  wrapper.InitKernelsDir();
  EXPECT_EQ(wrapper.GetKernelsDir(), "/opt/custom/ksana/triton_kernel_files");
}

TEST_F(TritonWrapperInitKernelsDirTest, SetKernelsDirOverrides) {
  ScopedCwd in_cwd(MakeIsolatedCwd());
  triton_path_env_.Unset();

  TritonWrapper wrapper;
  wrapper.SetKernelsDir("/explicit/dir");
  EXPECT_EQ(wrapper.GetKernelsDir(), "/explicit/dir");
}

class TritonWrapperResolveArtifactTest : public TritonWrapperInitKernelsDirTest {
 protected:
  fs::path MakeRoot() {
    fs::path root = fs::path(testing::TempDir()) / "ksana_triton_resolve" /
                    ::testing::UnitTest::GetInstance()->current_test_info()->name();
    fs::remove_all(root);
    fs::create_directories(root);
    return root;
  }
};

TEST_F(TritonWrapperResolveArtifactTest, HitsJobSubdirectory) {
  triton_path_env_.Unset();
  TritonWrapper wrapper;
  const fs::path root = MakeRoot();
  const fs::path expected = root / "kda" / "stem.json";
  WriteEmptyFile(expected);
  wrapper.SetKernelsDir(root.string());
  EXPECT_EQ(wrapper.ResolveKernelArtifact("stem", ".json"), expected.string());
}

TEST_F(TritonWrapperResolveArtifactTest, IgnoresFlatRootFile) {
  triton_path_env_.Unset();
  TritonWrapper wrapper;
  const fs::path root = MakeRoot();
  WriteEmptyFile(root / "stem.json");
  wrapper.SetKernelsDir(root.string());
  EXPECT_THROW(wrapper.ResolveKernelArtifact("stem", ".json"), std::runtime_error);
}

TEST_F(TritonWrapperResolveArtifactTest, PrefersSubdirWhenFlatAlsoPresent) {
  triton_path_env_.Unset();
  TritonWrapper wrapper;
  const fs::path root = MakeRoot();
  WriteEmptyFile(root / "stem.json");
  const fs::path expected = root / "kda" / "stem.json";
  WriteEmptyFile(expected);
  wrapper.SetKernelsDir(root.string());
  EXPECT_EQ(wrapper.ResolveKernelArtifact("stem", ".json"), expected.string());
}

TEST_F(TritonWrapperResolveArtifactTest, ThrowsWhenTwoJobSubdirsMatch) {
  triton_path_env_.Unset();
  TritonWrapper wrapper;
  const fs::path root = MakeRoot();
  WriteEmptyFile(root / "fused_moe" / "stem.json");
  WriteEmptyFile(root / "kda" / "stem.json");
  wrapper.SetKernelsDir(root.string());
  EXPECT_THROW(wrapper.ResolveKernelArtifact("stem", ".json"), std::runtime_error);
}

TEST_F(TritonWrapperResolveArtifactTest, SkipsMd5SumDirectory) {
  triton_path_env_.Unset();
  TritonWrapper wrapper;
  const fs::path root = MakeRoot();
  WriteEmptyFile(root / "md5_sum" / "stem.json");
  wrapper.SetKernelsDir(root.string());
  EXPECT_THROW(wrapper.ResolveKernelArtifact("stem", ".json"), std::runtime_error);
}

TEST_F(TritonWrapperResolveArtifactTest, SkipsMd5SumInsideJobSubdir) {
  triton_path_env_.Unset();
  TritonWrapper wrapper;
  const fs::path root = MakeRoot();
  WriteEmptyFile(root / "kda" / "md5_sum" / "stem.json");
  wrapper.SetKernelsDir(root.string());
  EXPECT_THROW(wrapper.ResolveKernelArtifact("stem", ".json"), std::runtime_error);
}

}  // namespace ksana_llm
