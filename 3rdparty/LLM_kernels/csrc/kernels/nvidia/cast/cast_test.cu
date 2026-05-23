/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <random>
#include <sstream>

#include <gtest/gtest.h>

#include "3rdparty/half/include/half.hpp"
#include "cast.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
class LlamaNvidiaCastTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const std::vector<size_t> test_data_size = {1ul, 4ul, 33ul, 1024ul, 32768ul};
  const std::vector<size_t> perf_data_size = {1048576ul, 4194304ul, 16777216ul};
  size_t warmup_times = 5;
  size_t test_times = 20;

  cudaEvent_t start;
  cudaEvent_t stop;

  void PrintPerfResult(const char* name, size_t data_size, float time_ms, size_t src_bytes, size_t dst_bytes) {
    float avg_ms = time_ms / test_times;
    // Total bytes = read src + write dst
    double total_bytes = static_cast<double>(data_size) * (src_bytes + dst_bytes);
    double throughput_gbps = (total_bytes / (1.0e9)) / (avg_ms / 1000.0);
    std::cout << name << " size: " << data_size << " avg_time: " << avg_ms << " ms"
              << " throughput: " << throughput_gbps << " GB/s" << std::endl;
  }
};

// ============================================================================
// Correctness Tests
// ============================================================================

TEST_F(LlamaNvidiaCastTestSuit, TestCastFloatToHalf) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<float>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<float>(input);
    BufferMeta output_device = CreateBuffer<half>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    FloatToHalf(reinterpret_cast<float*>(input.data_ptr), input_data_size,
                reinterpret_cast<half*>(output_device.data_ptr), stream);
    BufferMeta output_host = CopyToHost<half>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_float_to_half_run = [&]() {
      FloatToHalf(reinterpret_cast<float*>(input.data_ptr), input_data_size,
                  reinterpret_cast<half*>(output_device.data_ptr), stream);
    };
    float time_elapsed_ms = MeasureCudaExecutionTime(test_cast_float_to_half_run, stream, warmup_times, test_times);

    std::cout << "FloatToHalf tensor shape: [" << input_data_size << "] time elapsed: " << time_elapsed_ms / test_times
              << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      half* output_ptr = reinterpret_cast<half*>(output_host.data_ptr);
      float* output_ref_ptr = reinterpret_cast<float*>(input_host_reference.data_ptr);
      EXPECT_TRUE((half_float::half)(output_ptr[idx]) == (half_float::half)(output_ref_ptr[idx]));
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, TestCastFloatToBFloat16) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<float>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<float>(input);
    BufferMeta output_device =
        CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    FloatToBFloat16(reinterpret_cast<float*>(input.data_ptr), input_data_size,
                    reinterpret_cast<__nv_bfloat16*>(output_device.data_ptr), stream);
    BufferMeta output_host = CopyToHost<__nv_bfloat16>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_float_to_bfloat16_run = [&]() {
      FloatToBFloat16(reinterpret_cast<float*>(input.data_ptr), input_data_size,
                      reinterpret_cast<__nv_bfloat16*>(output_device.data_ptr), stream);
    };
    float time_elapsed_ms = MeasureCudaExecutionTime(test_cast_float_to_bfloat16_run, stream, warmup_times, test_times);

    std::cout << "FloatToBFloat16 tensor shape: [" << input_data_size
              << "] time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      __nv_bfloat16* output_ptr = reinterpret_cast<__nv_bfloat16*>(output_host.data_ptr);
      float* output_ref_ptr = reinterpret_cast<float*>(input_host_reference.data_ptr);
      // Due to the lower precision of BFloat16, we need to use approximate comparison
      float output_val = __bfloat162float(output_ptr[idx]);
      float ref_val = output_ref_ptr[idx];
      // For BFloat16, we allow a certain margin of error
      float abs_diff = std::abs(output_val - ref_val);
      float rel_diff = abs_diff / (std::abs(ref_val) + 1e-5f);
      EXPECT_TRUE(rel_diff < 1e-2f || abs_diff < 1e-2f);
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, TestCastHalfToFloat) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<half>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<half>(input);
    BufferMeta output_device = CreateBuffer<float>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    HalfToFloat(reinterpret_cast<half*>(input.data_ptr), input_data_size,
                reinterpret_cast<float*>(output_device.data_ptr), stream, 4ul, 2ul);
    BufferMeta output_host = CopyToHost<float>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    for (size_t run_it = 0; run_it < warmup_times; ++run_it) {
      HalfToFloat(reinterpret_cast<half*>(input.data_ptr), input_data_size,
                  reinterpret_cast<float*>(output_device.data_ptr), stream, 4ul, 2ul);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_half_to_float_run = [&]() {
      HalfToFloat(reinterpret_cast<half*>(input.data_ptr), input_data_size,
                  reinterpret_cast<float*>(output_device.data_ptr), stream, 4ul, 2ul);
    };
    float time_elapsed_ms = MeasureCudaExecutionTime(test_cast_half_to_float_run, stream, warmup_times, test_times);

    std::cout << "HalfToFloat tensor shape: [" << input_data_size << "] time elapsed: " << time_elapsed_ms / test_times
              << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size / 2; ++idx) {
      float* output_ptr = reinterpret_cast<float*>(output_host.data_ptr);
      half* output_ref_ptr = reinterpret_cast<half*>(input_host_reference.data_ptr);
      float output_val = output_ptr[idx];
      float ref_val = static_cast<float>(static_cast<half_float::half>(output_ref_ptr[idx / 2 * 4 + idx % 2]));
      // For Half to Float conversion, precision should be exact, but for code consistency, we also use relative and
      // absolute error metrics
      float abs_diff = std::abs(output_val - ref_val);
      float rel_diff = abs_diff / (std::abs(ref_val) + 1e-5f);
      EXPECT_TRUE(rel_diff < 1e-5f || abs_diff < 1e-5f);
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, TestCastHalfToFloatWithStrides) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<half>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<half>(input);
    BufferMeta output_device =
        CreateBuffer<float>(MemoryType::MEMORY_GPU, {input_data_size * 2}, /*is_random_init*/ false);
    HalfToFloat(reinterpret_cast<half*>(input.data_ptr), input_data_size,
                reinterpret_cast<float*>(output_device.data_ptr), stream, 2ul, 4ul);
    BufferMeta output_host = CopyToHost<float>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_half_to_float_with_stride_run = [&]() {
      HalfToFloat(reinterpret_cast<half*>(input.data_ptr), input_data_size,
                  reinterpret_cast<float*>(output_device.data_ptr), stream, 2ul, 4ul);
    };
    float time_elapsed_ms =
        MeasureCudaExecutionTime(test_cast_half_to_float_with_stride_run, stream, warmup_times, test_times);

    std::cout << "HalfToFloatWithStrides tensor shape: [" << input_data_size
              << "] time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      float* output_ptr = reinterpret_cast<float*>(output_host.data_ptr);
      half* output_ref_ptr = reinterpret_cast<half*>(input_host_reference.data_ptr);
      float output_val = output_ptr[idx / 2 * 4 + idx % 2];
      float ref_val = static_cast<float>(static_cast<half_float::half>((output_ref_ptr[idx])));
      // For Half to Float conversion, precision should be exact, but for code consistency, we also use relative and
      // absolute error metrics
      float abs_diff = std::abs(output_val - ref_val);
      float rel_diff = abs_diff / (std::abs(ref_val) + 1e-5f);
      EXPECT_TRUE(rel_diff < 1e-5f || abs_diff < 1e-5f);
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, TestCastHalfToBFloat16) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<half>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host = CopyToHost<half>(input);
    HalfToBFloat16(input.data_ptr, input_data_size, stream);
    BufferMeta output_host = CopyToHost<__nv_bfloat16>(input);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_half_to_bfloat16_run = [&]() { HalfToBFloat16(input.data_ptr, input_data_size, stream); };
    float time_elapsed_ms = MeasureCudaExecutionTime(test_cast_half_to_bfloat16_run, stream, warmup_times, test_times);

    std::cout << "HalfToBFloat16 tensor shape: [" << input_data_size
              << "] time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      half* input_ptr = reinterpret_cast<half*>(input_host.data_ptr);
      __nv_bfloat16* output_ptr = reinterpret_cast<__nv_bfloat16*>(output_host.data_ptr);
      EXPECT_TRUE(fabs((float)input_ptr[idx] - (float)(output_ptr[idx])) < 1e-3);
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, TestCastBFloat16ToFloat) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<__nv_bfloat16>(input);
    BufferMeta output_device = CreateBuffer<float>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    BFloat16ToFloat(reinterpret_cast<__nv_bfloat16*>(input.data_ptr), input_data_size,
                    reinterpret_cast<float*>(output_device.data_ptr), stream, 4ul, 2ul);
    BufferMeta output_host = CopyToHost<float>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_bfloat16_to_float_run = [&]() {
      BFloat16ToFloat(reinterpret_cast<__nv_bfloat16*>(input.data_ptr), input_data_size,
                      reinterpret_cast<float*>(output_device.data_ptr), stream, 4ul, 2ul);
    };
    float time_elapsed_ms = MeasureCudaExecutionTime(test_cast_bfloat16_to_float_run, stream, warmup_times, test_times);

    std::cout << "BFloat16ToFloat tensor shape: [" << input_data_size
              << "] time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size / 2; ++idx) {
      float* output_ptr = reinterpret_cast<float*>(output_host.data_ptr);
      __nv_bfloat16* output_ref_ptr = reinterpret_cast<__nv_bfloat16*>(input_host_reference.data_ptr);
      EXPECT_TRUE(output_ptr[idx] == static_cast<float>(output_ref_ptr[idx / 2 * 4 + idx % 2]));
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, TestCastBFloat16ToFloatWithStrides) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<__nv_bfloat16>(input);
    BufferMeta output_device =
        CreateBuffer<float>(MemoryType::MEMORY_GPU, {input_data_size * 2}, /*is_random_init*/ false);
    BFloat16ToFloat(reinterpret_cast<__nv_bfloat16*>(input.data_ptr), input_data_size,
                    reinterpret_cast<float*>(output_device.data_ptr), stream, 2ul, 4ul);
    BufferMeta output_host = CopyToHost<float>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_bfloat16_to_float_with_stride_run = [&]() {
      BFloat16ToFloat(reinterpret_cast<__nv_bfloat16*>(input.data_ptr), input_data_size,
                      reinterpret_cast<float*>(output_device.data_ptr), stream, 2ul, 4ul);
    };
    float time_elapsed_ms =
        MeasureCudaExecutionTime(test_cast_bfloat16_to_float_with_stride_run, stream, warmup_times, test_times);

    std::cout << "BFloat16ToFloatWithStrides tensor shape: [" << input_data_size
              << "] time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      float* output_ptr = reinterpret_cast<float*>(output_host.data_ptr);
      __nv_bfloat16* output_ref_ptr = reinterpret_cast<__nv_bfloat16*>(input_host_reference.data_ptr);
      float output_val = output_ptr[idx / 2 * 4 + idx % 2];
      float ref_val = __bfloat162float(output_ref_ptr[idx]);
      // For BFloat16 to Float conversion, precision should be exact, but for code consistency, we also use relative and
      // absolute error metrics
      float abs_diff = std::abs(output_val - ref_val);
      float rel_diff = abs_diff / (std::abs(ref_val) + 1e-5f);
      EXPECT_TRUE(rel_diff < 1e-5f || abs_diff < 1e-5f);
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, TestCastBFloat16ToHalf) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host = CopyToHost<__nv_bfloat16>(input);
    BFloat16ToHalf(input.data_ptr, input_data_size, stream);
    BufferMeta output_host = CopyToHost<half>(input);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_bfloat16_to_half_run = [&]() { BFloat16ToHalf(input.data_ptr, input_data_size, stream); };
    float time_elapsed_ms = MeasureCudaExecutionTime(test_cast_bfloat16_to_half_run, stream, warmup_times, test_times);

    std::cout << "BFloat16ToHalf tensor shape: [" << input_data_size
              << "] time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      __nv_bfloat16* input_ptr = reinterpret_cast<__nv_bfloat16*>(input_host.data_ptr);
      half* output_ptr = reinterpret_cast<half*>(output_host.data_ptr);
      EXPECT_TRUE((float)input_ptr[idx] == (float)(output_ptr[idx]));
    }
  }
}

#if defined(ENABLE_FP8)
TEST_F(LlamaNvidiaCastTestSuit, TestCastFloatToFp8E4M3) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<float>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<float>(input);
    BufferMeta output_device =
        CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    FloatToFp8E4M3(reinterpret_cast<float*>(input.data_ptr), input_data_size,
                   reinterpret_cast<__nv_fp8_e4m3*>(output_device.data_ptr), stream);
    BufferMeta output_host = CopyToHost<__nv_fp8_e4m3>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_float_to_fp8e4m3_run = [&]() {
      FloatToFp8E4M3(reinterpret_cast<float*>(input.data_ptr), input_data_size,
                     reinterpret_cast<__nv_fp8_e4m3*>(output_device.data_ptr), stream);
    };
    float time_elapsed_ms = MeasureCudaExecutionTime(test_cast_float_to_fp8e4m3_run, stream, warmup_times, test_times);

    std::cout << "FloatToFp8E4M3 tensor shape: [" << input_data_size
              << "] time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      __nv_fp8_e4m3* output_ptr = reinterpret_cast<__nv_fp8_e4m3*>(output_host.data_ptr);
      float* input_ref_ptr = reinterpret_cast<float*>(input_host_reference.data_ptr);
      // Due to the lower precision of FP8E4M3, we need to use approximate comparison
      float output_val = static_cast<float>(output_ptr[idx]);
      float ref_val = input_ref_ptr[idx];
      // For FP8E4M3, we allow a larger margin of error due to very low precision
      float abs_diff = std::abs(output_val - ref_val);
      float rel_diff = abs_diff / (std::abs(ref_val) + 1e-3f);
      EXPECT_TRUE(rel_diff < 0.1f || abs_diff < 0.1f);
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, TestCastFp8E4M3toFloat) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<__nv_fp8_e4m3>(input);
    BufferMeta output_device = CreateBuffer<float>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    Fp8E4M3ToFloat(reinterpret_cast<__nv_fp8_e4m3*>(input.data_ptr), input_data_size,
                   reinterpret_cast<float*>(output_device.data_ptr), stream);
    BufferMeta output_host = CopyToHost<float>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_fp8_e4m3_to_float_run = [&]() {
      Fp8E4M3ToFloat(reinterpret_cast<__nv_fp8_e4m3*>(input.data_ptr), input_data_size,
                     reinterpret_cast<float*>(output_device.data_ptr), stream);
    };
    float time_elapsed_ms = MeasureCudaExecutionTime(test_cast_fp8_e4m3_to_float_run, stream, warmup_times, test_times);

    std::cout << "Fp8E4M3ToFloat tensor shape: [" << input_data_size
              << "] time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      float* output_ptr = reinterpret_cast<float*>(output_host.data_ptr);
      __nv_fp8_e4m3* input_ref_ptr = reinterpret_cast<__nv_fp8_e4m3*>(input_host_reference.data_ptr);
      float output_val = output_ptr[idx];
      float ref_val = static_cast<float>(input_ref_ptr[idx]);
      // For FP8E4M3 to Float conversion, should be exact
      EXPECT_EQ(output_val, ref_val);
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, TestCastHalfToFp8E4M3) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<half>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<half>(input);
    BufferMeta output_device =
        CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    HalfToFp8E4M3(reinterpret_cast<half*>(input.data_ptr), input_data_size,
                  reinterpret_cast<__nv_fp8_e4m3*>(output_device.data_ptr), stream);
    BufferMeta output_host = CopyToHost<__nv_fp8_e4m3>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_half_to_fp8_e4m3_run = [&]() {
      HalfToFp8E4M3(reinterpret_cast<half*>(input.data_ptr), input_data_size,
                    reinterpret_cast<__nv_fp8_e4m3*>(output_device.data_ptr), stream);
    };
    float time_elapsed_ms = MeasureCudaExecutionTime(test_cast_half_to_fp8_e4m3_run, stream, warmup_times, test_times);

    std::cout << "HalfToFp8E4M3 tensor shape: [" << input_data_size
              << "] time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      __nv_fp8_e4m3* output_ptr = reinterpret_cast<__nv_fp8_e4m3*>(output_host.data_ptr);
      half* input_ref_ptr = reinterpret_cast<half*>(input_host_reference.data_ptr);
      // Due to the lower precision of FP8E4M3, we need to use approximate comparison
      float output_val = static_cast<float>(output_ptr[idx]);
      float ref_val = static_cast<float>(static_cast<half_float::half>(input_ref_ptr[idx]));
      // For FP8E4M3, we allow a larger margin of error due to very low precision
      float abs_diff = std::abs(output_val - ref_val);
      float rel_diff = abs_diff / (std::abs(ref_val) + 1e-3f);
      EXPECT_TRUE(rel_diff < 0.1f || abs_diff < 0.1f);
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, TestCastFp8E4M3toHalf) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<__nv_fp8_e4m3>(input);
    BufferMeta output_device = CreateBuffer<half>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    Fp8E4M3ToHalf(reinterpret_cast<__nv_fp8_e4m3*>(input.data_ptr), input_data_size,
                  reinterpret_cast<half*>(output_device.data_ptr), stream);
    BufferMeta output_host = CopyToHost<half>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_fp8_e4m3_to_half_run = [&]() {
      Fp8E4M3ToHalf(reinterpret_cast<__nv_fp8_e4m3*>(input.data_ptr), input_data_size,
                    reinterpret_cast<half*>(output_device.data_ptr), stream);
    };
    float time_elapsed_ms = MeasureCudaExecutionTime(test_cast_fp8_e4m3_to_half_run, stream, warmup_times, test_times);

    std::cout << "Fp8E4M3ToHalf tensor shape: [" << input_data_size
              << "] time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      half* output_ptr = reinterpret_cast<half*>(output_host.data_ptr);
      __nv_fp8_e4m3* input_ref_ptr = reinterpret_cast<__nv_fp8_e4m3*>(input_host_reference.data_ptr);
      float output_val = static_cast<float>(static_cast<half_float::half>(output_ptr[idx]));
      float ref_val = static_cast<float>(input_ref_ptr[idx]);
      // For FP8E4M3 to Half conversion, should be exact
      EXPECT_EQ(output_val, ref_val);
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, TestCastBFloat16ToFp8E4M3) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<__nv_bfloat16>(input);
    BufferMeta output_device =
        CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    BFloat16ToFp8E4M3(reinterpret_cast<__nv_bfloat16*>(input.data_ptr), input_data_size,
                      reinterpret_cast<__nv_fp8_e4m3*>(output_device.data_ptr), stream);
    BufferMeta output_host = CopyToHost<__nv_fp8_e4m3>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_bfloat16_to_fp8_e4m3_run = [&]() {
      BFloat16ToFp8E4M3(reinterpret_cast<__nv_bfloat16*>(input.data_ptr), input_data_size,
                        reinterpret_cast<__nv_fp8_e4m3*>(output_device.data_ptr), stream);
    };
    float time_elapsed_ms =
        MeasureCudaExecutionTime(test_cast_bfloat16_to_fp8_e4m3_run, stream, warmup_times, test_times);

    std::cout << "BFloat16ToFp8E4M3 tensor shape: [" << input_data_size
              << "] time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      __nv_fp8_e4m3* output_ptr = reinterpret_cast<__nv_fp8_e4m3*>(output_host.data_ptr);
      __nv_bfloat16* input_ref_ptr = reinterpret_cast<__nv_bfloat16*>(input_host_reference.data_ptr);
      // Due to the lower precision of FP8E4M3, we need to use approximate comparison
      float output_val = static_cast<float>(output_ptr[idx]);
      float ref_val = __bfloat162float(input_ref_ptr[idx]);
      // For FP8E4M3, we allow a larger margin of error due to very low precision
      float abs_diff = std::abs(output_val - ref_val);
      float rel_diff = abs_diff / (std::abs(ref_val) + 1e-3f);
      EXPECT_TRUE(rel_diff < 0.1f || abs_diff < 0.1f);
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, TestCastFp8E4M3toBFloat16) {
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<__nv_fp8_e4m3>(input);
    BufferMeta output_device =
        CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    Fp8E4M3ToBFloat16(reinterpret_cast<__nv_fp8_e4m3*>(input.data_ptr), input_data_size,
                      reinterpret_cast<__nv_bfloat16*>(output_device.data_ptr), stream);
    BufferMeta output_host = CopyToHost<__nv_bfloat16>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto test_cast_fp8_e4m3_to_bfloat16_run = [&]() {
      Fp8E4M3ToBFloat16(reinterpret_cast<__nv_fp8_e4m3*>(input.data_ptr), input_data_size,
                        reinterpret_cast<__nv_bfloat16*>(output_device.data_ptr), stream);
    };
    float time_elapsed_ms =
        MeasureCudaExecutionTime(test_cast_fp8_e4m3_to_bfloat16_run, stream, warmup_times, test_times);

    std::cout << "Fp8E4M3ToBFloat16 tensor shape: [" << input_data_size
              << "] time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      __nv_bfloat16* output_ptr = reinterpret_cast<__nv_bfloat16*>(output_host.data_ptr);
      __nv_fp8_e4m3* input_ref_ptr = reinterpret_cast<__nv_fp8_e4m3*>(input_host_reference.data_ptr);
      float output_val = __bfloat162float(output_ptr[idx]);
      float ref_val = static_cast<float>(input_ref_ptr[idx]);
      // For FP8E4M3 to BFloat16 conversion, should be exact
      EXPECT_EQ(output_val, ref_val);
    }
  }
}
#endif

// ============================================================================
// Performance Benchmarks (large data sizes)
// Prefixed with DISABLED_ to skip by default. Run with --gtest_also_run_disabled_tests.
// ============================================================================

TEST_F(LlamaNvidiaCastTestSuit, DISABLED_PerfBenchmarkFloatToHalf) {
  std::cout << "\n=== Performance Benchmark: FloatToHalf ===" << std::endl;
  for (size_t data_size : perf_data_size) {
    BufferMeta input = CreateBuffer<float>(MemoryType::MEMORY_GPU, {data_size}, /*is_random_init*/ true);
    BufferMeta output = CreateBuffer<half>(MemoryType::MEMORY_GPU, {data_size}, /*is_random_init*/ false);

    auto run = [&]() {
      FloatToHalf(reinterpret_cast<float*>(input.data_ptr), data_size, reinterpret_cast<half*>(output.data_ptr),
                  stream);
    };
    float time_ms = MeasureCudaExecutionTime(run, stream, warmup_times, test_times);
    PrintPerfResult("FloatToHalf", data_size, time_ms, sizeof(float), sizeof(half));

    // Quick correctness check on first 1000 elements
    BufferMeta input_host = CopyToHost<float>(input);
    BufferMeta output_host = CopyToHost<half>(output);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < std::min(data_size, (size_t)1000); ++i) {
      half* out_ptr = reinterpret_cast<half*>(output_host.data_ptr);
      float* in_ptr = reinterpret_cast<float*>(input_host.data_ptr);
      EXPECT_TRUE((half_float::half)(out_ptr[i]) == (half_float::half)(in_ptr[i]));
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, DISABLED_PerfBenchmarkFloatToBFloat16) {
  std::cout << "\n=== Performance Benchmark: FloatToBFloat16 ===" << std::endl;
  for (size_t data_size : perf_data_size) {
    BufferMeta input = CreateBuffer<float>(MemoryType::MEMORY_GPU, {data_size}, /*is_random_init*/ true);
    BufferMeta output = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {data_size}, /*is_random_init*/ false);

    auto run = [&]() {
      FloatToBFloat16(reinterpret_cast<float*>(input.data_ptr), data_size,
                      reinterpret_cast<__nv_bfloat16*>(output.data_ptr), stream);
    };
    float time_ms = MeasureCudaExecutionTime(run, stream, warmup_times, test_times);
    PrintPerfResult("FloatToBFloat16", data_size, time_ms, sizeof(float), sizeof(__nv_bfloat16));
  }
}

TEST_F(LlamaNvidiaCastTestSuit, DISABLED_PerfBenchmarkHalfToFloat) {
  std::cout << "\n=== Performance Benchmark: HalfToFloat (no stride) ===" << std::endl;
  for (size_t data_size : perf_data_size) {
    BufferMeta input = CreateBuffer<half>(MemoryType::MEMORY_GPU, {data_size}, /*is_random_init*/ true);
    BufferMeta output = CreateBuffer<float>(MemoryType::MEMORY_GPU, {data_size}, /*is_random_init*/ false);

    auto run = [&]() {
      HalfToFloat(reinterpret_cast<half*>(input.data_ptr), data_size, reinterpret_cast<float*>(output.data_ptr),
                  stream);
    };
    float time_ms = MeasureCudaExecutionTime(run, stream, warmup_times, test_times);
    PrintPerfResult("HalfToFloat", data_size, time_ms, sizeof(half), sizeof(float));
  }
}

TEST_F(LlamaNvidiaCastTestSuit, DISABLED_PerfBenchmarkBFloat16ToFloat) {
  std::cout << "\n=== Performance Benchmark: BFloat16ToFloat (no stride) ===" << std::endl;
  for (size_t data_size : perf_data_size) {
    BufferMeta input = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {data_size}, /*is_random_init*/ true);
    BufferMeta output = CreateBuffer<float>(MemoryType::MEMORY_GPU, {data_size}, /*is_random_init*/ false);

    auto run = [&]() {
      BFloat16ToFloat(reinterpret_cast<__nv_bfloat16*>(input.data_ptr), data_size,
                      reinterpret_cast<float*>(output.data_ptr), stream);
    };
    float time_ms = MeasureCudaExecutionTime(run, stream, warmup_times, test_times);
    PrintPerfResult("BFloat16ToFloat", data_size, time_ms, sizeof(__nv_bfloat16), sizeof(float));
  }
}

TEST_F(LlamaNvidiaCastTestSuit, DISABLED_PerfBenchmarkHalfToBFloat16) {
  std::cout << "\n=== Performance Benchmark: HalfToBFloat16 (in-place) ===" << std::endl;
  for (size_t data_size : perf_data_size) {
    BufferMeta buffer = CreateBuffer<half>(MemoryType::MEMORY_GPU, {data_size}, /*is_random_init*/ true);

    auto run = [&]() { HalfToBFloat16(buffer.data_ptr, data_size, stream); };
    float time_ms = MeasureCudaExecutionTime(run, stream, warmup_times, test_times);
    PrintPerfResult("HalfToBFloat16", data_size, time_ms, sizeof(half), sizeof(__nv_bfloat16));
  }
}

TEST_F(LlamaNvidiaCastTestSuit, DISABLED_PerfBenchmarkBFloat16ToHalf) {
  std::cout << "\n=== Performance Benchmark: BFloat16ToHalf (in-place) ===" << std::endl;
  for (size_t data_size : perf_data_size) {
    BufferMeta buffer = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {data_size}, /*is_random_init*/ true);

    auto run = [&]() { BFloat16ToHalf(buffer.data_ptr, data_size, stream); };
    float time_ms = MeasureCudaExecutionTime(run, stream, warmup_times, test_times);
    PrintPerfResult("BFloat16ToHalf", data_size, time_ms, sizeof(__nv_bfloat16), sizeof(half));
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
