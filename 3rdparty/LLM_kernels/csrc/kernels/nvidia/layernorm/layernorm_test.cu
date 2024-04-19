/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/layernorm/layernorm.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
class LlamaNvidiaLayernormTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const std::vector<std::pair<int, int>> m_n_pairs = {{2, 4096}};

  template <typename T>
  void RunRMSNormRef(const float variance_epsilon = 1e-6f) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }

    std::stringstream ss;
    ss << "python layernorm_test.py --type=" << type_str << " --variance_epsilon=" << std::to_string(variance_epsilon);
    system(ss.str().c_str());
  }

  template <typename T>
  void TestRMSNorm(const size_t m, const size_t n) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }

    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                            /*is_random_init*/ true);
    BufferMeta weight_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {n},
                                             /*is_random_init*/ true);
    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                             /*is_random_init*/ false);
    BufferMeta output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                                 /*is_random_init*/ false);
    float layernorm_eps = 1e-6;

    input_meta.SaveNpy<T>("rmsnorm_test_input.npy");
    weight_meta.SaveNpy<T>("rmsnorm_test_weight.npy");
    RunRMSNormRef<T>(layernorm_eps);
    LoadNpy<T>("rmsnorm_test_output.npy", MemoryType::MEMORY_GPU, output_ref_meta);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    T* beta = nullptr;
    InvokeLayerNorm<T>(reinterpret_cast<T*>(output_meta.data_ptr), reinterpret_cast<const T*>(input_meta.data_ptr),
                       reinterpret_cast<const T*>(weight_meta.data_ptr), beta, layernorm_eps, m, n, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_TRUE(CheckResult<T>("layernorm_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               output_ref_meta, output_meta, 1e-3f, 1e-3f));

    DeleteBuffer(output_ref_meta);
    DeleteBuffer(output_meta);
    DeleteBuffer(weight_meta);
    DeleteBuffer(input_meta);
  }
};

TEST_F(LlamaNvidiaLayernormTestSuit, HalfLayernormCommonTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestRMSNorm<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(LlamaNvidiaLayernormTestSuit, FloatLayernormCommonTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestRMSNorm<float>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels