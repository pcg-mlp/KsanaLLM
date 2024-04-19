/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/activation/activation.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaActivationTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const std::vector<std::pair<int, int>> m_n_pairs = {{2, 4096}};

 protected:
  template <typename T>
  void RunSiluRef() {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }

    std::stringstream ss;
    ss << "python activation_silu_test.py --type=" << type_str;
    system(ss.str().c_str());
  }

  template <typename T>
  void TestSilu(const size_t m, const size_t n, cudaStream_t stream) {
    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                            /*is_random_init*/ true);
    BufferMeta gated_weight_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                                   /*is_random_init*/ true);
    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                             /*is_random_init*/ false);
    BufferMeta output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                                 /*is_random_init*/ false);

    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }

    input_meta.SaveNpy<T>("silu_test_input.npy");
    gated_weight_meta.SaveNpy<T>("silu_test_gated_weight.npy");

    RunSiluRef<T>();

    LoadNpy<T>("silu_test_output.npy", MemoryType::MEMORY_GPU, output_ref_meta);

    const int* ia3_tasks = nullptr;
    const T* bias = nullptr;
    const T* ia3_weights = nullptr;
    const T* gated_bias = nullptr;
    const int int8_mode = 0;
    const int* padding_offset = nullptr;
    const int seq_len = 0;
    const float* activation_in = nullptr;
    const float* activation_out = nullptr;
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(output_meta.data_ptr, input_meta.data_ptr, sizeof(T) * m * n,
                                            cudaMemcpyDeviceToDevice, stream));
    InvokeGenericActivation<SiluActivation, T, T>(reinterpret_cast<T*>(output_meta.data_ptr), bias,
                                                  reinterpret_cast<const T*>(gated_weight_meta.data_ptr), gated_bias,
                                                  ia3_tasks, ia3_weights, m, n, int8_mode, activation_in,
                                                  activation_out, padding_offset, seq_len, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_TRUE(CheckResult<T>("activation_silu_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               output_ref_meta, output_meta, 1e-5f, 1e-5f));

    DeleteBuffer(output_ref_meta);
    DeleteBuffer(output_meta);
    DeleteBuffer(gated_weight_meta);
    DeleteBuffer(input_meta);
  }
};

TEST_F(LlamaNvidiaActivationTestSuit, HalfSiluCommonTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestSilu<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second), stream);
  }
}

TEST_F(LlamaNvidiaActivationTestSuit, FloatSiluCommonTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestSilu<float>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second), stream);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels