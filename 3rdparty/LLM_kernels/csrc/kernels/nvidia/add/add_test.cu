/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/add/add.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaAddTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const std::vector<std::pair<int, int>> m_n_pairs = {{2, 4096}};

 protected:
  template <typename T>
  void AddBiasResidualRef(const size_t m, const size_t n, const T* input_a, const T* input_b, T* output) {
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
      for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        if (std::is_same<T, half>::value) {
          output[m_idx * n + n_idx] =
              (half)(((half_float::half)input_a[m_idx * n + n_idx]) + ((half_float::half)input_b[m_idx * n + n_idx]));
        } else if (std::is_same<T, float>::value) {
          output[m_idx * n + n_idx] = input_a[m_idx * n + n_idx] + input_b[m_idx * n + n_idx];
        }
      }
    }
  }

  template <typename T>
  void TestAddBiasResidual(const size_t m, const size_t n) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }

    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                             /*is_random_init*/ false);
    BufferMeta input_a_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                              /*is_random_init*/ true);
    BufferMeta input_b_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                              /*is_random_init*/ true);
    BufferMeta output_cpu_meta = CopyToHost<T>(output_meta);
    BufferMeta input_a_cpu_meta = CopyToHost<T>(input_a_meta);
    BufferMeta input_b_cpu_meta = CopyToHost<T>(input_b_meta);

    AddBiasResidualRef<T>(m, n, reinterpret_cast<const T*>(input_a_cpu_meta.data_ptr),
                          reinterpret_cast<const T*>(input_b_cpu_meta.data_ptr),
                          reinterpret_cast<T*>(output_cpu_meta.data_ptr));
    BufferMeta output_ref_meta = CopyToDevice<T>(output_cpu_meta);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    InvokeAddBiasResidual<T>(
        reinterpret_cast<T*>(output_meta.data_ptr), reinterpret_cast<const T*>(input_a_meta.data_ptr),
        reinterpret_cast<const T*>(input_b_meta.data_ptr), nullptr, nullptr, nullptr, nullptr, m, n, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_TRUE(CheckResult<T>("add_bias_residual_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               output_ref_meta, output_meta, 1e-5f, 1e-5f));

    DeleteBuffer(output_ref_meta);
    DeleteBuffer(input_b_cpu_meta);
    DeleteBuffer(input_a_cpu_meta);
    DeleteBuffer(output_cpu_meta);
    DeleteBuffer(output_meta);
    DeleteBuffer(input_a_meta);
    DeleteBuffer(input_b_meta);
  }
};

TEST_F(LlamaNvidiaAddTestSuit, HalfAddBiasResidualTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestAddBiasResidual<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(LlamaNvidiaAddTestSuit, FloatAddBiasResidualTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestAddBiasResidual<float>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels