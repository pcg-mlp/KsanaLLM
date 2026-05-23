/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/others/sglang/main/quantization/fp8/per_token_group_quant.h"

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/activation/activation.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaPerTokenGroupQuantFp8TestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const int group_size_ = 128;
  const std::vector<std::pair<int, int>> m_n_pairs_ = {{1, 128}, {1, 256}};

 protected:
  template <typename T>
  void TestPerTokenGroupQuantFp8(const size_t m, const size_t n, cudaStream_t stream, bool is_column_major) {
    size_t data_size = m * n * sizeof(T);
    size_t q_size = m * n;
    size_t s_size = m * n / group_size_;

    // Allocate device memory
    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                            /*is_random_init*/ true);
    BufferMeta output_q_meta = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {q_size},
                                                           /*is_random_init*/ false);
    BufferMeta output_s_meta = CreateBuffer<float>(MemoryType::MEMORY_GPU, {s_size},
                                                   /*is_random_init*/ false);
    // Set input data to device
    std::vector<float> h_data_float(m * n);
    for (size_t i = 0; i < m * n; ++i) {
      h_data_float[i] = i / 100.0f;
    }
    if (std::is_same<T, float>::value) {
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(input_meta.data_ptr, h_data_float.data(), data_size, cudaMemcpyHostToDevice));
    } else if (std::is_same<T, half>::value) {
      std::vector<half> h_data_half(m * n);
      for (size_t i = 0; i < m * n; ++i) {
        h_data_half[i] = __float2half(h_data_float[i]);
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(input_meta.data_ptr, h_data_half.data(), data_size, cudaMemcpyHostToDevice));
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      std::vector<__nv_bfloat16> h_data_bf16(m * n);
      for (size_t i = 0; i < m * n; ++i) {
        h_data_bf16[i] = __float2bfloat16(h_data_float[i]);
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(input_meta.data_ptr, h_data_bf16.data(), data_size, cudaMemcpyHostToDevice));
    }

    // Call the kernel
    per_token_group_quant_fp8<T>(input_meta.data_ptr, output_q_meta.data_ptr, output_s_meta.data_ptr, m, n, group_size_,
                                 is_column_major, /*fuse_silu_mul*/ false, /*scale_ue8m0*/ false,
                                 /*swiglu_limit*/ 0.0f, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy output data back to host
    std::vector<uint8_t> h_q(q_size);
    std::vector<float> h_s(s_size);
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(h_q.data(), output_q_meta.data_ptr, q_size * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(h_s.data(), output_s_meta.data_ptr, s_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Reference data
    std::vector<float> target_scales = {0.0028, 0.0057};
    std::vector<int> target_q = {
        0,   70,  78,  83,  86,  89,  91,  92,  94,  96,  97,  98,  99,  99,  100, 101, 102, 103, 104, 104, 105, 105,
        106, 106, 107, 107, 107, 108, 108, 109, 109, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 113, 113,
        114, 114, 114, 114, 115, 115, 115, 115, 115, 116, 116, 116, 116, 117, 117, 117, 117, 117, 118, 118, 118, 118,
        119, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120, 120, 121, 121, 121, 121, 121, 121, 121, 121, 121, 122,
        122, 122, 122, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124,
        124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 118, 118, 118, 118,
        118, 119, 119, 119, 119, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
        120, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 122, 122,
        122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123,
        123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 124, 124, 124, 124,
        124, 124, 124, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
        125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126};

    // Check the results
    for (size_t i = 0; i < (m * n) / 128; ++i) {
      EXPECT_NEAR(h_s[i], target_scales[i], 1e-4);
    }
    for (size_t i = 0; i < m * n; ++i) {
      EXPECT_NEAR(static_cast<int>(h_q[i]), target_q[i], 1);
    }

    DeleteBuffer(input_meta);
    DeleteBuffer(output_q_meta);
    DeleteBuffer(output_s_meta);
  }

  template <typename T>
  void TestSiluMulPerTokenGroupQuantFp8(const size_t m, const size_t n, cudaStream_t stream, bool is_column_major,
                                        bool scale_ue8m0) {
    size_t q_size = m * n;
    size_t s_size = m * n / group_size_;

    // Allocate device memory
    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n * 2},
                                            /*is_random_init*/ true);
    BufferMeta output_q_meta = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {q_size},
                                                           /*is_random_init*/ false);
    BufferMeta output_s_meta = CreateBuffer<float>(MemoryType::MEMORY_GPU, {s_size},
                                                   /*is_random_init*/ false);
    BufferMeta fused_output_q_meta = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {q_size},
                                                                 /*is_random_init*/ false);
    BufferMeta fused_output_s_meta = CreateBuffer<float>(MemoryType::MEMORY_GPU, {s_size},
                                                         /*is_random_init*/ false);

    // Invoke the fused version
    per_token_group_quant_fp8<T>(input_meta.data_ptr, fused_output_q_meta.data_ptr, fused_output_s_meta.data_ptr, m, n,
                                 group_size_, is_column_major, /*fuse_silu_mul*/ true, scale_ue8m0,
                                 /*swiglu_limit*/ 0.0f, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    // Invoke the unfused version
    InvokeRowBasedActivation<SiluActivation, T, /*HAS_SWIGLU_LIMIT*/ false>(
        reinterpret_cast<T*>(input_meta.data_ptr), reinterpret_cast<const T*>(input_meta.data_ptr), m, n * 2,
        /*swiglu_limit=*/0.0f, stream);
    per_token_group_quant_fp8<T>(input_meta.data_ptr, output_q_meta.data_ptr, output_s_meta.data_ptr, m, n, group_size_,
                                 is_column_major, /*fuse_silu_mul*/ false, scale_ue8m0,
                                 /*swiglu_limit*/ 0.0f, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Verify accuracy
    EXPECT_TRUE(CheckResult<__nv_fp8_e4m3>(
        "silu_mul_per_token_group_quant_fp8_m_" + std::to_string(m) + "_n_" + std::to_string(n) + "_output_q",
        fused_output_q_meta, output_q_meta, 1e-4, 1e-4));
    EXPECT_TRUE(CheckResult<T>(
        "silu_mul_per_token_group_quant_fp8_m_" + std::to_string(m) + "_n_" + std::to_string(n) + "_output_s",
        fused_output_s_meta, output_s_meta, 1e-4, 1e-4));

    // Free data
    DeleteBuffer(input_meta);
    DeleteBuffer(output_q_meta);
    DeleteBuffer(output_s_meta);
    DeleteBuffer(fused_output_q_meta);
    DeleteBuffer(fused_output_s_meta);
  }
};

TEST_F(LlamaNvidiaPerTokenGroupQuantFp8TestSuit, HalfPerTokenGroupQuantFp8Test) {
  for (const auto& m_n_pair : m_n_pairs_) {
    TestPerTokenGroupQuantFp8<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second), stream,
                                    true);
    TestPerTokenGroupQuantFp8<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second), stream,
                                    false);
  }
}

TEST_F(LlamaNvidiaPerTokenGroupQuantFp8TestSuit, BFloat16PerTokenGroupQuantFp8Test) {
  for (const auto& m_n_pair : m_n_pairs_) {
    TestPerTokenGroupQuantFp8<__nv_bfloat16>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second),
                                             stream, true);
    TestPerTokenGroupQuantFp8<__nv_bfloat16>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second),
                                             stream, false);
  }
}

// Performance test is disabled by default
TEST_F(LlamaNvidiaPerTokenGroupQuantFp8TestSuit, DISABLED_PerTokenGroupQuantFp8PerfTest) {
  const std::vector<size_t> token_nums = {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  // Config of DeepSeek-V3
  const size_t hidden_size = 7168;

  for (const size_t token_num : token_nums) {
    // Prepare device data
    BufferMeta d_input = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {token_num, hidden_size},
                                                     /*is_random_init*/ true);
    BufferMeta d_output_q = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {token_num, hidden_size},
                                                        /*is_random_init*/ false);
    BufferMeta d_output_s = CreateBuffer<float>(MemoryType::MEMORY_GPU, {token_num, hidden_size / group_size_},
                                                /*is_random_init*/ false);

    const int warmups = 5;
    const int iterations = 10;
    auto cuda_run = [&]() {
      per_token_group_quant_fp8<__nv_bfloat16>(
          d_input.data_ptr, d_output_q.data_ptr, d_output_s.data_ptr, token_num, hidden_size, group_size_,
          /*is_column_major*/ false, /*fuse_silu_mul*/ false, /*scale_ue8m0*/ false,
          /*swiglu_limit*/ 0.0f, stream);
    };
    const float elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, warmups, iterations);
    std::cout << "Token num: " << token_num << ", Execution time: " << elapsed_ms << " ms" << std::endl;

    // Free data
    DeleteBuffer(d_input);
    DeleteBuffer(d_output_q);
    DeleteBuffer(d_output_s);
  }
}

TEST_F(LlamaNvidiaPerTokenGroupQuantFp8TestSuit, HalfSiluMulPerTokenGroupQuantFp8Test) {
  for (const auto& m_n_pair : m_n_pairs_) {
    TestSiluMulPerTokenGroupQuantFp8<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second),
                                           stream, true, false);
    TestSiluMulPerTokenGroupQuantFp8<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second),
                                           stream, false, false);
    TestSiluMulPerTokenGroupQuantFp8<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second),
                                           stream, false, true);
  }
}

TEST_F(LlamaNvidiaPerTokenGroupQuantFp8TestSuit, BFloat16SiluMulPerTokenGroupQuantFp8Test) {
  for (const auto& m_n_pair : m_n_pairs_) {
    TestSiluMulPerTokenGroupQuantFp8<__nv_bfloat16>(static_cast<size_t>(m_n_pair.first),
                                                    static_cast<size_t>(m_n_pair.second), stream, true, false);
    TestSiluMulPerTokenGroupQuantFp8<__nv_bfloat16>(static_cast<size_t>(m_n_pair.first),
                                                    static_cast<size_t>(m_n_pair.second), stream, false, false);
    TestSiluMulPerTokenGroupQuantFp8<__nv_bfloat16>(static_cast<size_t>(m_n_pair.first),
                                                    static_cast<size_t>(m_n_pair.second), stream, false, true);
  }
}

TEST_F(LlamaNvidiaPerTokenGroupQuantFp8TestSuit, SwigluLimitSiluMulPerTokenGroupQuantFp8Test) {
  const size_t m = 8, n = 128;
  const float swiglu_limit = 3.0f;
  using T = __nv_bfloat16;

  // Input shape: [m, 2*n] (gate in [0..n-1], up in [n..2n-1])
  // Output shape: [m, n] fp8 + [m, n/group_size_] float scale
  const size_t q_size = m * n;
  const size_t s_size = m * n / group_size_;

  BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n * 2}, /*is_random_init*/ false);
  BufferMeta output_q_clamp = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {q_size}, /*is_random_init*/ false);
  BufferMeta output_s_clamp = CreateBuffer<float>(MemoryType::MEMORY_GPU, {s_size}, /*is_random_init*/ false);
  BufferMeta output_q_noclamp = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {q_size}, /*is_random_init*/ false);
  BufferMeta output_s_noclamp = CreateBuffer<float>(MemoryType::MEMORY_GPU, {s_size}, /*is_random_init*/ false);

  // Initialize host input:
  // All positions: gate=0.5, up=0.5 (small baseline values producing silu(0.5)*0.5 ≈ 0.188)
  // Token 0, ALL gate positions: set to +8.0 (well above limit=3)
  // Token 0, ALL up positions: set to +8.0 (well above limit=3)
  // With clamp: gate=clamp(8, max=3)=3, up=clamp(8, [-3,3])=3
  //   silu(3)*3 ≈ 2.884 * 3 ≈ 8.65 (group max ≈ 8.65)
  // Without clamp: silu(8)*8 ≈ 8.0 * 8 = 64.0 (group max ≈ 64.0)
  // The scales will differ by ~7x, making this a robust test.
  std::vector<T> h_input(m * n * 2);
  for (size_t i = 0; i < m * n * 2; ++i) {
    h_input[i] = __float2bfloat16(0.5f);
  }
  // Token 0: ALL gate positions [0..127] = 8.0
  for (size_t j = 0; j < n; ++j) {
    h_input[j] = __float2bfloat16(8.0f);
  }
  // Token 0: ALL up positions [128..255] = 8.0
  for (size_t j = 0; j < n; ++j) {
    h_input[n + j] = __float2bfloat16(8.0f);
  }
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(input_meta.data_ptr, h_input.data(), m * n * 2 * sizeof(T), cudaMemcpyHostToDevice));

  // Call with HAS_SWIGLU_LIMIT=true
  per_token_group_quant_fp8<T>(input_meta.data_ptr, output_q_clamp.data_ptr, output_s_clamp.data_ptr, m, n, group_size_,
                               /*is_column_major*/ false, /*fuse_silu_mul*/ true, /*scale_ue8m0*/ false,
                               /*swiglu_limit*/ swiglu_limit, stream);

  // Call with HAS_SWIGLU_LIMIT=false
  per_token_group_quant_fp8<T>(input_meta.data_ptr, output_q_noclamp.data_ptr, output_s_noclamp.data_ptr, m, n,
                               group_size_, /*is_column_major*/ false, /*fuse_silu_mul*/ true, /*scale_ue8m0*/ false,
                               /*swiglu_limit*/ 0.0f, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Copy outputs back to host
  std::vector<float> h_s_clamp(s_size), h_s_noclamp(s_size);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(h_s_clamp.data(), output_s_clamp.data_ptr, s_size * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(h_s_noclamp.data(), output_s_noclamp.data_ptr, s_size * sizeof(float), cudaMemcpyDeviceToHost));

  // Validation: scale for token 0 group 0 should be much smaller with clamp than without.
  // With clamp: silu(clamp(8,3)) * clamp(8,[-3,3]) = silu(3)*3 ≈ 2.884*3 ≈ 8.65
  //   scale_clamp = 8.65 / 448 ≈ 0.0193
  // Without clamp: silu(8)*8 ≈ 8.0*8 = 64.0
  //   scale_noclamp = 64.0 / 448 ≈ 0.1429
  // Expect: scale_clamp < scale_noclamp (by at least 3x)
  const float scale_clamp = h_s_clamp[0];
  const float scale_noclamp = h_s_noclamp[0];
  EXPECT_GT(scale_noclamp, scale_clamp * 3.0f) << "noclamp scale (" << scale_noclamp << ") should be > 3x clamp scale ("
                                               << scale_clamp << "), indicating clamp reduced input magnitudes";

  // Also verify absolute bounds on clamp scale:
  // silu(3) = 3 / (1 + e^{-3}) ≈ 2.8761, silu(3)*3 ≈ 8.628
  // expected scale_clamp ≈ 8.628 / 448 ≈ 0.01926
  const float silu3 = 3.0f / (1.0f + expf(-3.0f));
  const float expected_group_max_clamp = silu3 * 3.0f;
  const float expected_scale_clamp = expected_group_max_clamp / 448.0f;
  EXPECT_NEAR(scale_clamp, expected_scale_clamp, expected_scale_clamp * 0.1f)
      << "clamp scale should be approximately " << expected_scale_clamp;

  DeleteBuffer(input_meta);
  DeleteBuffer(output_q_clamp);
  DeleteBuffer(output_s_clamp);
  DeleteBuffer(output_q_noclamp);
  DeleteBuffer(output_s_noclamp);
}

// Performance test is disabled by default
TEST_F(LlamaNvidiaPerTokenGroupQuantFp8TestSuit, DISABLED_SiluMulPerTokenGroupQuantFp8PerfTest) {
  const std::vector<size_t> token_nums = {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  // Config of DeepSeek-R1
  const size_t hidden_size = 7168;

  for (const size_t token_num : token_nums) {
    // Prepare device data
    BufferMeta d_input = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {token_num, hidden_size * 2},
                                                     /*is_random_init*/ true);
    BufferMeta d_output_q = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {token_num, hidden_size},
                                                        /*is_random_init*/ false);
    BufferMeta d_output_s = CreateBuffer<float>(MemoryType::MEMORY_GPU, {token_num, hidden_size / group_size_},
                                                /*is_random_init*/ false);

    const int warmups = 5;
    const int iterations = 10;
    // Run the fused version
    auto cuda_run_fused = [&]() {
      per_token_group_quant_fp8<__nv_bfloat16>(d_input.data_ptr, d_output_q.data_ptr, d_output_s.data_ptr, token_num,
                                               hidden_size, group_size_,
                                               /*is_column_major*/ false, /*fuse_silu_mul*/ true, /*scale_ue8m0*/ false,
                                               /*swiglu_limit*/ 0.0f, stream);
    };
    // Run the unfused version
    auto cuda_run_unfused = [&]() {
      InvokeRowBasedActivation<SiluActivation, __nv_bfloat16, /*HAS_SWIGLU_LIMIT*/ false>(
          reinterpret_cast<__nv_bfloat16*>(d_input.data_ptr), reinterpret_cast<const __nv_bfloat16*>(d_input.data_ptr),
          token_num, hidden_size * 2, /*swiglu_limit=*/0.0f, stream);
      per_token_group_quant_fp8<__nv_bfloat16>(
          d_input.data_ptr, d_output_q.data_ptr, d_output_s.data_ptr, token_num, hidden_size, group_size_,
          /*is_column_major*/ false, /*fuse_silu_mul*/ false, /*scale_ue8m0*/ false,
          /*swiglu_limit*/ 0.0f, stream);
    };
    const float elapsed_ms_fused = MeasureCudaExecutionTime(cuda_run_fused, stream, warmups, iterations);
    const float elapsed_ms_unfused = MeasureCudaExecutionTime(cuda_run_unfused, stream, warmups, iterations);
    std::cout << "Token num: " << token_num << ", Execution time of fused version: " << elapsed_ms_fused << " ms"
              << ", Execution time of unfused version: " << elapsed_ms_unfused << " ms" << std::endl;

    // Free data
    DeleteBuffer(d_input);
    DeleteBuffer(d_output_q);
    DeleteBuffer(d_output_s);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
