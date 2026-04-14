/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "pooling.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace test {

// ============================================================================
// CPU Reference Implementation
// ============================================================================

// Compute L2 normalize on CPU in float64 for maximum precision reference.
void CpuL2NormalizeFloat64(const float* input, float* output, int32_t batch_size, int32_t hidden_size) {
  for (int32_t row = 0; row < batch_size; ++row) {
    const float* row_in = input + row * hidden_size;
    float* row_out = output + row * hidden_size;

    // Accumulate in double for reference precision.
    double sum_sq = 0.0;
    for (int32_t i = 0; i < hidden_size; ++i) {
      double val = static_cast<double>(row_in[i]);
      sum_sq += val * val;
    }
    double norm = std::sqrt(sum_sq);
    double inv_norm = (norm > 0.0) ? (1.0 / norm) : 1.0;

    for (int32_t i = 0; i < hidden_size; ++i) {
      row_out[i] = static_cast<float>(static_cast<double>(row_in[i]) * inv_norm);
    }
  }
}

// ============================================================================
// Test Fixture
// ============================================================================

class PoolingL2NormalizeTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  // -------------------------------------------------------------------
  // Test parameter sets
  // -------------------------------------------------------------------

  // Correctness test: cover various hidden_size values including edge cases.
  // {batch_size, hidden_size}
  const std::vector<std::pair<int32_t, int32_t>> correctness_configs = {
      // Edge cases: very small
      {1, 1},
      {1, 3},
      {1, 7},
      // Non-power-of-2 (exercises tail handling in vectorized kernel)
      {1, 13},
      {2, 33},
      {4, 127},
      {1, 255},
      {2, 513},
      // Power-of-2 (aligned, no tail)
      {1, 64},
      {2, 128},
      {4, 256},
      {2, 512},
      {1, 1024},
      {2, 2048},
      // Production embedding dimensions
      {1, 768},     // BERT-base
      {1, 1024},    // many small models
      {1, 3584},    // some Qwen variants
      {4, 4096},    // Qwen3-Embedding
      {8, 4096},    // Qwen3-Embedding, larger batch
      {1, 5120},    // LLaMA-13B
      {1, 8192},    // LLaMA-70B
      // Large batch
      {32, 4096},
      {64, 4096},
      {128, 4096},
  };

  // Performance test: sweep batch_size × hidden_size.
  const std::vector<std::pair<int32_t, int32_t>> perf_configs = {
      // Fix hidden_size=4096 (Qwen3-Embedding), vary batch
      {1, 4096},
      {2, 4096},
      {4, 4096},
      {8, 4096},
      {16, 4096},
      {32, 4096},
      {64, 4096},
      {128, 4096},
      {256, 4096},
      // Fix batch_size=16, vary hidden_size
      {16, 768},
      {16, 1024},
      {16, 2048},
      {16, 3584},
      {16, 4096},
      {16, 5120},
      {16, 8192},
  };

  static constexpr int32_t kWarmupIters = 20;
  static constexpr int32_t kBenchIters = 200;

  // -------------------------------------------------------------------
  // Helper: run correctness test for a given dtype T
  // -------------------------------------------------------------------
  template <typename T>
  void RunCorrectnessTest(int32_t batch_size, int32_t hidden_size, float atol, float rtol,
                          const std::string& type_name) {
    const size_t n_elems = static_cast<size_t>(batch_size) * hidden_size;

    // 1. Create random input on CPU (float), then cast to T and upload to GPU.
    BufferMeta input_f32_cpu = CreateBuffer<float>(MEMORY_CPU, {n_elems}, /*is_random_init=*/true, -1.0f, 1.0f);

    // Compute CPU reference in float (from float input).
    BufferMeta ref_f32_cpu = CreateBuffer<float>(MEMORY_CPU, {n_elems}, /*is_random_init=*/false);
    CpuL2NormalizeFloat64(reinterpret_cast<float*>(input_f32_cpu.data_ptr),
                          reinterpret_cast<float*>(ref_f32_cpu.data_ptr), batch_size, hidden_size);

    // Cast input float → T on CPU, then upload to GPU.
    BufferMeta input_t_cpu = CreateBuffer<T>(MEMORY_CPU, {n_elems}, /*is_random_init=*/false);
    {
      float* src = reinterpret_cast<float*>(input_f32_cpu.data_ptr);
      T* dst = reinterpret_cast<T*>(input_t_cpu.data_ptr);
      for (size_t i = 0; i < n_elems; ++i) {
        dst[i] = static_cast<T>(src[i]);
      }
    }
    BufferMeta input_t_gpu = CopyToDevice<T>(input_t_cpu);

    // 2. Run kernel (in-place).
    InvokeL2Normalize<T>(reinterpret_cast<T*>(input_t_gpu.data_ptr), batch_size, hidden_size, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 3. Copy result back to CPU.
    BufferMeta output_t_cpu = CopyToHost<T>(input_t_gpu);

    // 4. Compare against CPU reference.
    //    We compare element-wise in float, accounting for dtype precision.
    float* ref_ptr = reinterpret_cast<float*>(ref_f32_cpu.data_ptr);
    T* out_ptr = reinterpret_cast<T*>(output_t_cpu.data_ptr);

    size_t failures = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    for (size_t i = 0; i < n_elems; ++i) {
      float out_val;
      if (std::is_same<T, __nv_bfloat16>::value) {
        // bf16 → float
        uint16_t raw = *reinterpret_cast<uint16_t*>(&out_ptr[i]);
        unsigned int tmp = static_cast<unsigned int>(raw) << 16;
        out_val = *reinterpret_cast<float*>(&tmp);
      } else {
        out_val = static_cast<float>(out_ptr[i]);
      }

      float ref_val = ref_ptr[i];
      float abs_err = std::fabs(out_val - ref_val);
      float rel_err = abs_err / (std::fabs(ref_val) + 1e-8f);

      max_abs_err = std::max(max_abs_err, abs_err);
      max_rel_err = std::max(max_rel_err, rel_err);

      if (!AlmostEqual(out_val, ref_val, atol, rtol)) {
        ++failures;
      }
    }

    // Allow up to 1% mismatch for low-precision types.
    size_t tol_failures = static_cast<size_t>(0.01f * n_elems);
    EXPECT_LE(failures, tol_failures)
        << type_name << " L2Normalize [" << batch_size << ", " << hidden_size << "] "
        << "failures=" << failures << "/" << n_elems << " max_abs_err=" << max_abs_err
        << " max_rel_err=" << max_rel_err;

    // 5. Verify L2 norm of each output row is ~1.0.
    for (int32_t row = 0; row < batch_size; ++row) {
      double row_norm_sq = 0.0;
      for (int32_t j = 0; j < hidden_size; ++j) {
        float val;
        if (std::is_same<T, __nv_bfloat16>::value) {
          uint16_t raw = *reinterpret_cast<uint16_t*>(&out_ptr[row * hidden_size + j]);
          unsigned int tmp = static_cast<unsigned int>(raw) << 16;
          val = *reinterpret_cast<float*>(&tmp);
        } else {
          val = static_cast<float>(out_ptr[row * hidden_size + j]);
        }
        row_norm_sq += static_cast<double>(val) * static_cast<double>(val);
      }
      double row_norm = std::sqrt(row_norm_sq);
      EXPECT_NEAR(row_norm, 1.0, atol)
          << type_name << " row " << row << " norm=" << row_norm << " (expected ~1.0)"
          << " [batch=" << batch_size << ", hidden=" << hidden_size << "]";
    }

    // Cleanup
    DeleteBuffer(input_f32_cpu);
    DeleteBuffer(ref_f32_cpu);
    DeleteBuffer(input_t_cpu);
    DeleteBuffer(input_t_gpu);
    DeleteBuffer(output_t_cpu);
  }

  // -------------------------------------------------------------------
  // Helper: run performance benchmark for a given dtype T
  // -------------------------------------------------------------------
  template <typename T>
  void RunPerfBenchmark(int32_t batch_size, int32_t hidden_size, const std::string& type_name) {
    const size_t n_elems = static_cast<size_t>(batch_size) * hidden_size;

    // Allocate random GPU buffer.
    BufferMeta data_gpu = CreateBuffer<T>(MEMORY_GPU, {n_elems}, /*is_random_init=*/true, -1.0f, 1.0f);

    auto kernel_run = [&]() {
      InvokeL2Normalize<T>(reinterpret_cast<T*>(data_gpu.data_ptr), batch_size, hidden_size, stream);
    };

    // MeasureCudaExecutionTime returns the average time per iteration (ms).
    float avg_ms = MeasureCudaExecutionTime(kernel_run, stream, kWarmupIters, kBenchIters);

    // Compute effective bandwidth.
    // With shared-memory optimization (aligned case): 1 global read + 1 global write = 2 × data_size.
    // Without (non-aligned fallback): 2 global reads + 1 global write = 3 × data_size.
    // We report the optimistic (smem) case since production hidden_sizes are aligned.
    size_t data_bytes = n_elems * sizeof(T);
    size_t total_bytes = 2 * data_bytes;  // Optimized: 1 read + 1 write
    double bandwidth_gb_s = (static_cast<double>(total_bytes) / (1.0e9)) / (static_cast<double>(avg_ms) / 1000.0);

    std::cout << std::fixed << std::setprecision(4) << "L2Normalize<" << std::setw(7) << type_name << "> "
              << "batch=" << std::setw(4) << batch_size << "  hidden=" << std::setw(5) << hidden_size << "  "
              << "avg=" << std::setw(10) << avg_ms << " ms  "
              << "BW=" << std::setw(8) << std::setprecision(2) << bandwidth_gb_s << " GB/s" << std::endl;

    DeleteBuffer(data_gpu);
  }

  // -------------------------------------------------------------------
  // Special test: zero vector, constant vector, single-element
  // -------------------------------------------------------------------
  template <typename T>
  void RunEdgeCaseTests(const std::string& type_name) {
    // Test 1: zero vector → should remain zero (norm=0 → skip normalize)
    {
      const int32_t hidden_size = 128;
      BufferMeta zero_gpu = CreateBuffer<T>(MEMORY_GPU, {static_cast<size_t>(hidden_size)}, /*is_random_init=*/false);
      InvokeL2Normalize<T>(reinterpret_cast<T*>(zero_gpu.data_ptr), 1, hidden_size, stream);
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

      BufferMeta zero_cpu = CopyToHost<T>(zero_gpu);
      T* ptr = reinterpret_cast<T*>(zero_cpu.data_ptr);
      for (int32_t i = 0; i < hidden_size; ++i) {
        EXPECT_EQ(static_cast<float>(ptr[i]), 0.0f)
            << type_name << " zero vector: element " << i << " should be 0.0";
      }
      DeleteBuffer(zero_gpu);
      DeleteBuffer(zero_cpu);
    }

    // Test 2: constant vector → all elements should be equal after normalize
    {
      const int32_t hidden_size = 256;
      BufferMeta const_cpu = CreateBuffer<T>(MEMORY_CPU, {static_cast<size_t>(hidden_size)}, /*is_random_init=*/false);
      T* ptr = reinterpret_cast<T*>(const_cpu.data_ptr);
      for (int32_t i = 0; i < hidden_size; ++i) {
        ptr[i] = static_cast<T>(1.0f);
      }
      BufferMeta const_gpu = CopyToDevice<T>(const_cpu);

      InvokeL2Normalize<T>(reinterpret_cast<T*>(const_gpu.data_ptr), 1, hidden_size, stream);
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

      BufferMeta result_cpu = CopyToHost<T>(const_gpu);
      T* res = reinterpret_cast<T*>(result_cpu.data_ptr);

      // Expected: each element = 1 / sqrt(hidden_size)
      float expected = 1.0f / std::sqrt(static_cast<float>(hidden_size));
      for (int32_t i = 0; i < hidden_size; ++i) {
        float val = static_cast<float>(res[i]);
        EXPECT_NEAR(val, expected, 1e-2f)
            << type_name << " constant vector: element " << i << " expected " << expected << " got " << val;
      }

      DeleteBuffer(const_cpu);
      DeleteBuffer(const_gpu);
      DeleteBuffer(result_cpu);
    }

    // Test 3: single element → should become 1.0 (or -1.0)
    {
      BufferMeta single_cpu = CreateBuffer<T>(MEMORY_CPU, {1}, /*is_random_init=*/false);
      T* ptr = reinterpret_cast<T*>(single_cpu.data_ptr);
      ptr[0] = static_cast<T>(3.14f);
      BufferMeta single_gpu = CopyToDevice<T>(single_cpu);

      InvokeL2Normalize<T>(reinterpret_cast<T*>(single_gpu.data_ptr), 1, 1, stream);
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

      BufferMeta result_cpu = CopyToHost<T>(single_gpu);
      float val = static_cast<float>(*reinterpret_cast<T*>(result_cpu.data_ptr));
      EXPECT_NEAR(val, 1.0f, 1e-3f) << type_name << " single element: expected 1.0 got " << val;

      DeleteBuffer(single_cpu);
      DeleteBuffer(single_gpu);
      DeleteBuffer(result_cpu);
    }

    // Test 4: batch_size=0 and hidden_size=0 → should not crash
    {
      InvokeL2Normalize<T>(nullptr, 0, 0, stream);
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      // If we reach here, no crash — test passes.
    }
  }

  // -------------------------------------------------------------------
  // Idempotency test: running L2 normalize twice should give same result
  // (already-normalized vector should stay the same)
  // -------------------------------------------------------------------
  template <typename T>
  void RunIdempotencyTest(int32_t batch_size, int32_t hidden_size, float atol, const std::string& type_name) {
    const size_t n_elems = static_cast<size_t>(batch_size) * hidden_size;

    BufferMeta data_gpu = CreateBuffer<T>(MEMORY_GPU, {n_elems}, /*is_random_init=*/true, -1.0f, 1.0f);

    // First normalize.
    InvokeL2Normalize<T>(reinterpret_cast<T*>(data_gpu.data_ptr), batch_size, hidden_size, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Save result of first normalize.
    BufferMeta after_first = CopyToHost<T>(data_gpu);

    // Second normalize (on already-normalized data).
    InvokeL2Normalize<T>(reinterpret_cast<T*>(data_gpu.data_ptr), batch_size, hidden_size, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta after_second = CopyToHost<T>(data_gpu);

    // Compare: should be identical (within dtype precision).
    T* first_ptr = reinterpret_cast<T*>(after_first.data_ptr);
    T* second_ptr = reinterpret_cast<T*>(after_second.data_ptr);

    for (size_t i = 0; i < n_elems; ++i) {
      float val1 = static_cast<float>(first_ptr[i]);
      float val2 = static_cast<float>(second_ptr[i]);
      EXPECT_NEAR(val1, val2, atol)
          << type_name << " idempotency failed at index " << i << ": first=" << val1 << " second=" << val2;
    }

    DeleteBuffer(data_gpu);
    DeleteBuffer(after_first);
    DeleteBuffer(after_second);
  }
};

// ============================================================================
// Correctness Tests
// ============================================================================

TEST_F(PoolingL2NormalizeTestSuit, FloatCorrectness) {
  for (const auto& cfg : correctness_configs) {
    RunCorrectnessTest<float>(cfg.first, cfg.second, /*atol=*/1e-5f, /*rtol=*/1e-5f, "float");
  }
}

TEST_F(PoolingL2NormalizeTestSuit, HalfCorrectness) {
  for (const auto& cfg : correctness_configs) {
    RunCorrectnessTest<half>(cfg.first, cfg.second, /*atol=*/1e-2f, /*rtol=*/1e-2f, "half");
  }
}

TEST_F(PoolingL2NormalizeTestSuit, BFloat16Correctness) {
  for (const auto& cfg : correctness_configs) {
    RunCorrectnessTest<__nv_bfloat16>(cfg.first, cfg.second, /*atol=*/1e-2f, /*rtol=*/1e-2f, "bf16");
  }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(PoolingL2NormalizeTestSuit, FloatEdgeCases) {
  RunEdgeCaseTests<float>("float");
}

TEST_F(PoolingL2NormalizeTestSuit, HalfEdgeCases) {
  RunEdgeCaseTests<half>("half");
}

TEST_F(PoolingL2NormalizeTestSuit, BFloat16EdgeCases) {
  RunEdgeCaseTests<__nv_bfloat16>("bf16");
}

// ============================================================================
// Idempotency Tests
// ============================================================================

TEST_F(PoolingL2NormalizeTestSuit, FloatIdempotency) {
  RunIdempotencyTest<float>(4, 4096, /*atol=*/1e-5f, "float");
}

TEST_F(PoolingL2NormalizeTestSuit, HalfIdempotency) {
  RunIdempotencyTest<half>(4, 4096, /*atol=*/1e-2f, "half");
}

TEST_F(PoolingL2NormalizeTestSuit, BFloat16Idempotency) {
  RunIdempotencyTest<__nv_bfloat16>(4, 4096, /*atol=*/1e-2f, "bf16");
}

// ============================================================================
// Performance Benchmarks
// (Prefixed with DISABLED_ so they don't run in CI; use --gtest_also_run_disabled_tests
//  or --gtest_filter=*DISABLED_*Perf* to run them manually.)
// ============================================================================

TEST_F(PoolingL2NormalizeTestSuit, DISABLED_FloatPerf) {
  std::cout << "\n========== L2Normalize Performance Benchmark (float) ==========" << std::endl;
  for (const auto& cfg : perf_configs) {
    RunPerfBenchmark<float>(cfg.first, cfg.second, "float");
  }
}

TEST_F(PoolingL2NormalizeTestSuit, DISABLED_HalfPerf) {
  std::cout << "\n========== L2Normalize Performance Benchmark (half) ==========" << std::endl;
  for (const auto& cfg : perf_configs) {
    RunPerfBenchmark<half>(cfg.first, cfg.second, "half");
  }
}

TEST_F(PoolingL2NormalizeTestSuit, DISABLED_BFloat16Perf) {
  std::cout << "\n========== L2Normalize Performance Benchmark (bf16) ==========" << std::endl;
  for (const auto& cfg : perf_configs) {
    RunPerfBenchmark<__nv_bfloat16>(cfg.first, cfg.second, "bf16");
  }
}

// Combined: all types in one table for easy comparison.
TEST_F(PoolingL2NormalizeTestSuit, DISABLED_AllTypesPerf) {
  std::cout << "\n========== L2Normalize Performance Comparison (all types) ==========" << std::endl;
  std::cout << std::string(90, '-') << std::endl;

  // Focus on Qwen3-Embedding production config.
  const std::vector<std::pair<int32_t, int32_t>> focus_configs = {
      {1, 4096},  {4, 4096},  {8, 4096},  {16, 4096},
      {32, 4096}, {64, 4096}, {128, 4096}, {256, 4096},
  };

  for (const auto& cfg : focus_configs) {
    RunPerfBenchmark<float>(cfg.first, cfg.second, "float");
    RunPerfBenchmark<half>(cfg.first, cfg.second, "half");
    RunPerfBenchmark<__nv_bfloat16>(cfg.first, cfg.second, "bf16");
    std::cout << std::string(90, '-') << std::endl;
  }
}

// ============================================================================
// MeanPooling: CPU Reference Implementation
// ============================================================================

// Compute mean pooling on CPU in float64 for maximum precision reference.
// input:   [total_tokens, hidden_size]
// offsets: [batch_size + 1]  (offsets[i] = start token of seq i)
// output:  [batch_size, hidden_size]
void CpuMeanPoolingFloat64(const float* input, const size_t* offsets, int32_t batch_size, int32_t hidden_size,
                            float* output) {
  for (int32_t b = 0; b < batch_size; ++b) {
    size_t seq_start = offsets[b];
    size_t seq_end = offsets[b + 1];
    size_t seq_len = seq_end - seq_start;
    float* row_out = output + b * hidden_size;

    if (seq_len == 0) {
      for (int32_t j = 0; j < hidden_size; ++j) {
        row_out[j] = 0.0f;
      }
      continue;
    }

    for (int32_t j = 0; j < hidden_size; ++j) {
      double sum = 0.0;
      for (size_t t = seq_start; t < seq_end; ++t) {
        sum += static_cast<double>(input[t * hidden_size + j]);
      }
      row_out[j] = static_cast<float>(sum / static_cast<double>(seq_len));
    }
  }
}

// ============================================================================
// MeanPooling Test Fixture
// ============================================================================

class PoolingMeanPoolingTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  // -------------------------------------------------------------------
  // Test parameter sets: {seq_lengths, hidden_size}
  // -------------------------------------------------------------------

  struct MeanPoolingConfig {
    std::vector<int32_t> seq_lengths;
    int32_t hidden_size;
  };

  const std::vector<MeanPoolingConfig> correctness_configs = {
      // Single sequence
      {{10}, 128},
      {{1}, 64},
      {{256}, 4096},
      // Multiple uniform sequences
      {{8, 8, 8, 8}, 256},
      {{16, 16, 16, 16}, 4096},
      // Multiple variable-length sequences
      {{3, 7, 15, 1}, 128},
      {{1, 50, 5, 20, 10}, 512},
      {{100, 1, 30, 2, 50, 8}, 768},
      // Non-aligned hidden_size
      {{4, 6}, 13},
      {{3, 5, 7}, 33},
      {{2, 4}, 127},
      {{8, 8}, 255},
      {{4, 4}, 513},
      // Large batch
      {{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}, 4096},
      {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 256},
  };

  // -------------------------------------------------------------------
  // Helper: run correctness test for a given dtype T
  // -------------------------------------------------------------------
  template <typename T>
  void RunCorrectnessTest(const MeanPoolingConfig& cfg, float atol, float rtol, const std::string& type_name) {
    const int32_t batch_size = static_cast<int32_t>(cfg.seq_lengths.size());
    const int32_t hidden_size = cfg.hidden_size;

    // Build offsets from seq_lengths.
    std::vector<size_t> offsets(batch_size + 1);
    offsets[0] = 0;
    for (int32_t i = 0; i < batch_size; ++i) {
      offsets[i + 1] = offsets[i] + static_cast<size_t>(cfg.seq_lengths[i]);
    }
    const size_t total_tokens = offsets[batch_size];
    const size_t input_elems = total_tokens * hidden_size;
    const size_t output_elems = static_cast<size_t>(batch_size) * hidden_size;

    // 1. Create random input on CPU (float), then cast to T and upload to GPU.
    BufferMeta input_f32_cpu = CreateBuffer<float>(MEMORY_CPU, {input_elems}, /*is_random_init=*/true, -1.0f, 1.0f);

    // Compute CPU reference in float.
    BufferMeta ref_f32_cpu = CreateBuffer<float>(MEMORY_CPU, {output_elems}, /*is_random_init=*/false);
    CpuMeanPoolingFloat64(reinterpret_cast<float*>(input_f32_cpu.data_ptr), offsets.data(), batch_size, hidden_size,
                           reinterpret_cast<float*>(ref_f32_cpu.data_ptr));

    // Cast input float → T on CPU, then upload to GPU.
    BufferMeta input_t_cpu = CreateBuffer<T>(MEMORY_CPU, {input_elems}, /*is_random_init=*/false);
    {
      float* src = reinterpret_cast<float*>(input_f32_cpu.data_ptr);
      T* dst = reinterpret_cast<T*>(input_t_cpu.data_ptr);
      for (size_t i = 0; i < input_elems; ++i) {
        dst[i] = static_cast<T>(src[i]);
      }
    }
    BufferMeta input_t_gpu = CopyToDevice<T>(input_t_cpu);

    // 2. Allocate GPU offsets (use cudaMalloc, not CreateBuffer — offsets are size_t, not a tensor type).
    size_t* d_offsets = nullptr;
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_offsets, (batch_size + 1) * sizeof(size_t)));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_offsets, offsets.data(), (batch_size + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

    // 3. Allocate GPU output.
    BufferMeta output_t_gpu = CreateBuffer<T>(MEMORY_GPU, {output_elems}, /*is_random_init=*/false);

    // 4. Run kernel.
    InvokeMeanPooling<T>(reinterpret_cast<T*>(input_t_gpu.data_ptr), d_offsets, batch_size, hidden_size,
                          reinterpret_cast<T*>(output_t_gpu.data_ptr), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 5. Copy result back to CPU.
    BufferMeta output_t_cpu = CopyToHost<T>(output_t_gpu);

    // 6. Compare against CPU reference.
    float* ref_ptr = reinterpret_cast<float*>(ref_f32_cpu.data_ptr);
    T* out_ptr = reinterpret_cast<T*>(output_t_cpu.data_ptr);

    size_t failures = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    for (size_t i = 0; i < output_elems; ++i) {
      float out_val;
      if (std::is_same<T, __nv_bfloat16>::value) {
        uint16_t raw = *reinterpret_cast<uint16_t*>(&out_ptr[i]);
        unsigned int tmp = static_cast<unsigned int>(raw) << 16;
        out_val = *reinterpret_cast<float*>(&tmp);
      } else {
        out_val = static_cast<float>(out_ptr[i]);
      }

      float ref_val = ref_ptr[i];
      float abs_err = std::fabs(out_val - ref_val);
      float rel_err = abs_err / (std::fabs(ref_val) + 1e-8f);

      max_abs_err = std::max(max_abs_err, abs_err);
      max_rel_err = std::max(max_rel_err, rel_err);

      if (!AlmostEqual(out_val, ref_val, atol, rtol)) {
        ++failures;
      }
    }

    // Allow up to 1% mismatch for low-precision types.
    size_t tol_failures = static_cast<size_t>(0.01f * output_elems);
    EXPECT_LE(failures, tol_failures)
        << type_name << " MeanPooling [batch=" << batch_size << ", hidden=" << hidden_size << "] "
        << "failures=" << failures << "/" << output_elems << " max_abs_err=" << max_abs_err
        << " max_rel_err=" << max_rel_err;

    // Cleanup
    DeleteBuffer(input_f32_cpu);
    DeleteBuffer(ref_f32_cpu);
    DeleteBuffer(input_t_cpu);
    DeleteBuffer(input_t_gpu);
    DeleteBuffer(output_t_gpu);
    DeleteBuffer(output_t_cpu);
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(d_offsets));
  }

  // -------------------------------------------------------------------
  // Helper: empty sequence test — a sequence with seq_len=0 should produce zeros
  // -------------------------------------------------------------------
  template <typename T>
  void RunEmptySequenceTest(const std::string& type_name) {
    // Batch of 3 sequences: lengths = {4, 0, 3}. The middle one is empty.
    const int32_t hidden_size = 128;
    const std::vector<int32_t> seq_lengths = {4, 0, 3};
    const int32_t batch_size = static_cast<int32_t>(seq_lengths.size());

    std::vector<size_t> offsets(batch_size + 1);
    offsets[0] = 0;
    for (int32_t i = 0; i < batch_size; ++i) {
      offsets[i + 1] = offsets[i] + static_cast<size_t>(seq_lengths[i]);
    }
    const size_t total_tokens = offsets[batch_size];  // 7
    const size_t input_elems = total_tokens * hidden_size;
    const size_t output_elems = static_cast<size_t>(batch_size) * hidden_size;

    // Create random input.
    BufferMeta input_t_cpu = CreateBuffer<T>(MEMORY_CPU, {input_elems}, /*is_random_init=*/true, -1.0f, 1.0f);
    BufferMeta input_t_gpu = CopyToDevice<T>(input_t_cpu);

    // Upload offsets.
    size_t* d_offsets = nullptr;
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_offsets, (batch_size + 1) * sizeof(size_t)));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_offsets, offsets.data(), (batch_size + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

    // Allocate output (init to non-zero to verify the kernel writes zeros).
    BufferMeta output_t_cpu_init = CreateBuffer<T>(MEMORY_CPU, {output_elems}, /*is_random_init=*/false);
    {
      T* ptr = reinterpret_cast<T*>(output_t_cpu_init.data_ptr);
      for (size_t i = 0; i < output_elems; ++i) {
        ptr[i] = static_cast<T>(1.0f);
      }
    }
    BufferMeta output_t_gpu = CopyToDevice<T>(output_t_cpu_init);

    // Run kernel.
    InvokeMeanPooling<T>(reinterpret_cast<T*>(input_t_gpu.data_ptr), d_offsets, batch_size, hidden_size,
                          reinterpret_cast<T*>(output_t_gpu.data_ptr), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta output_t_cpu = CopyToHost<T>(output_t_gpu);
    T* out_ptr = reinterpret_cast<T*>(output_t_cpu.data_ptr);

    // Check that the empty sequence (index 1) produced all zeros.
    for (int32_t j = 0; j < hidden_size; ++j) {
      float val;
      if (std::is_same<T, __nv_bfloat16>::value) {
        uint16_t raw = *reinterpret_cast<uint16_t*>(&out_ptr[1 * hidden_size + j]);
        unsigned int tmp = static_cast<unsigned int>(raw) << 16;
        val = *reinterpret_cast<float*>(&tmp);
      } else {
        val = static_cast<float>(out_ptr[1 * hidden_size + j]);
      }
      EXPECT_EQ(val, 0.0f) << type_name << " empty sequence: element " << j << " should be 0.0, got " << val;
    }

    // Cleanup
    DeleteBuffer(input_t_cpu);
    DeleteBuffer(input_t_gpu);
    DeleteBuffer(output_t_cpu_init);
    DeleteBuffer(output_t_gpu);
    DeleteBuffer(output_t_cpu);
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(d_offsets));
  }
};

// ============================================================================
// MeanPooling Correctness Tests
// ============================================================================

TEST_F(PoolingMeanPoolingTestSuit, FloatCorrectness) {
  for (const auto& cfg : correctness_configs) {
    RunCorrectnessTest<float>(cfg, /*atol=*/1e-5f, /*rtol=*/1e-5f, "float");
  }
}

TEST_F(PoolingMeanPoolingTestSuit, HalfCorrectness) {
  for (const auto& cfg : correctness_configs) {
    RunCorrectnessTest<half>(cfg, /*atol=*/1e-2f, /*rtol=*/1e-2f, "half");
  }
}

TEST_F(PoolingMeanPoolingTestSuit, BFloat16Correctness) {
  for (const auto& cfg : correctness_configs) {
    RunCorrectnessTest<__nv_bfloat16>(cfg, /*atol=*/1e-2f, /*rtol=*/1e-2f, "bf16");
  }
}

// ============================================================================
// MeanPooling Edge Case Tests
// ============================================================================

TEST_F(PoolingMeanPoolingTestSuit, FloatEmptySequence) {
  RunEmptySequenceTest<float>("float");
}

TEST_F(PoolingMeanPoolingTestSuit, BFloat16EmptySequence) {
  RunEmptySequenceTest<__nv_bfloat16>("bf16");
}

TEST_F(PoolingMeanPoolingTestSuit, NullInputNoCrash) {
  // Calling with nullptr and batch_size=0 should not crash.
  InvokeMeanPooling<float>(nullptr, nullptr, 0, 0, nullptr, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  // If we reach here, no crash — test passes.

  InvokeMeanPooling<__nv_bfloat16>(nullptr, nullptr, 0, 0, nullptr, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
