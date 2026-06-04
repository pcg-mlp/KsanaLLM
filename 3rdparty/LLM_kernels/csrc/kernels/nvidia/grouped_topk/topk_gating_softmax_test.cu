/* Copyright 2026 Tencent Inc.  All rights reserved.
 * Unit test comparing BasicSoftmaxTopk (old) vs TopkGatingSoftmax (new, sglang-derived)
 ==============================================================================*/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "csrc/kernels/nvidia/grouped_topk/grouped_topk.h"
#include "csrc/kernels/nvidia/grouped_topk/topk_gating_softmax.h"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class TopkGatingSoftmaxTest : public testing::Test {
 protected:
  void SetUp() override { cudaStreamCreate(&stream_); }
  void TearDown() override {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }

  cudaStream_t stream_;

  // CPU reference: softmax -> topk -> renormalize -> scale
  void CpuSoftmaxTopk(const std::vector<float>& logits, int num_rows, int num_experts, int topk, bool renormalize,
                       float scaling, std::vector<float>& out_weights, std::vector<int32_t>& out_ids) {
    out_weights.resize(num_rows * topk);
    out_ids.resize(num_rows * topk);
    for (int r = 0; r < num_rows; ++r) {
      const float* row = logits.data() + r * num_experts;
      float max_val = *std::max_element(row, row + num_experts);
      std::vector<float> scores(num_experts);
      float sum = 0;
      for (int i = 0; i < num_experts; ++i) {
        scores[i] = std::exp(static_cast<double>(row[i]) - static_cast<double>(max_val));
        sum += scores[i];
      }
      for (int i = 0; i < num_experts; ++i) scores[i] /= sum;

      // Greedy topk
      std::vector<int> indices(num_experts);
      std::iota(indices.begin(), indices.end(), 0);
      std::partial_sort(indices.begin(), indices.begin() + topk, indices.end(),
                        [&](int a, int b) { return scores[a] > scores[b]; });

      float renorm_scale = scaling;
      if (renormalize) {
        float topk_sum = 0;
        for (int i = 0; i < topk; ++i) topk_sum += scores[indices[i]];
        renorm_scale = scaling / topk_sum;
      }
      for (int i = 0; i < topk; ++i) {
        out_ids[r * topk + i] = indices[i];
        out_weights[r * topk + i] = scores[indices[i]] * renorm_scale;
      }
    }
  }

  template <typename T>
  void TestBothKernels(int M, int E, int topk, bool renormalize, float scaling) {
    // Generate random logits
    std::mt19937 gen(42 + M * 1000 + E);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> h_logits(M * E);
    for (auto& v : h_logits) v = dist(gen);

    // CPU reference
    std::vector<float> ref_weights;
    std::vector<int32_t> ref_ids;
    CpuSoftmaxTopk(h_logits, M, E, topk, renormalize, scaling, ref_weights, ref_ids);

    // Convert to T
    std::vector<T> h_logits_t(M * E);
    for (int i = 0; i < M * E; ++i) h_logits_t[i] = static_cast<T>(h_logits[i]);

    // Allocate GPU memory
    void *d_logits, *d_weights_old, *d_ids_old, *d_weights_new, *d_ids_new;
    cudaMalloc(&d_logits, M * E * sizeof(T));
    cudaMalloc(&d_weights_old, M * topk * sizeof(float));
    cudaMalloc(&d_ids_old, M * topk * sizeof(int32_t));
    cudaMalloc(&d_weights_new, M * topk * sizeof(float));
    cudaMalloc(&d_ids_new, M * topk * sizeof(int32_t));

    cudaMemcpy(d_logits, h_logits_t.data(), M * E * sizeof(T), cudaMemcpyHostToDevice);

    // Run OLD kernel (BasicSoftmaxTopk) — note: master's BasicSoftmaxTopk does not support renormalize
    InvokeBasicSoftmaxTopk<T>(d_logits, d_weights_old, d_ids_old, M, E, topk, scaling, stream_);

    // Run NEW kernel (TopkGatingSoftmax)
    InvokeTopkGatingSoftmax<T>(d_logits, d_weights_new, d_ids_new, M, E, topk, renormalize, scaling, stream_);

    cudaStreamSynchronize(stream_);

    // Read back
    std::vector<float> old_weights(M * topk), new_weights(M * topk);
    std::vector<int32_t> old_ids(M * topk), new_ids(M * topk);
    cudaMemcpy(old_weights.data(), d_weights_old, M * topk * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(old_ids.data(), d_ids_old, M * topk * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(new_weights.data(), d_weights_new, M * topk * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(new_ids.data(), d_ids_new, M * topk * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Compare NEW kernel vs CPU reference (set comparison)
    int set_mismatches = 0;
    float max_weight_err = 0;
    for (int r = 0; r < M; ++r) {
      std::set<int32_t> new_set(new_ids.begin() + r * topk, new_ids.begin() + (r + 1) * topk);
      std::set<int32_t> ref_set(ref_ids.begin() + r * topk, ref_ids.begin() + (r + 1) * topk);
      if (new_set != ref_set) {
        set_mismatches++;
        continue;
      }
      // Compare weights on matching rows (sorted)
      std::vector<float> new_w(new_weights.begin() + r * topk, new_weights.begin() + (r + 1) * topk);
      std::vector<float> ref_w(ref_weights.begin() + r * topk, ref_weights.begin() + (r + 1) * topk);
      std::sort(new_w.begin(), new_w.end());
      std::sort(ref_w.begin(), ref_w.end());
      for (int k = 0; k < topk; ++k) {
        max_weight_err = std::max(max_weight_err, std::abs(new_w[k] - ref_w[k]));
      }
    }

    // Compare OLD vs NEW directly (only meaningful when renormalize=false,
    // since BasicSoftmaxTopk does not support renormalize)
    int old_new_mismatches = 0;
    float old_new_max_err = 0;
    if (!renormalize) {
      for (int r = 0; r < M; ++r) {
        std::set<int32_t> old_set(old_ids.begin() + r * topk, old_ids.begin() + (r + 1) * topk);
        std::set<int32_t> new_set(new_ids.begin() + r * topk, new_ids.begin() + (r + 1) * topk);
        if (old_set != new_set) {
          old_new_mismatches++;
          continue;
        }
        std::vector<float> old_w(old_weights.begin() + r * topk, old_weights.begin() + (r + 1) * topk);
        std::vector<float> new_w(new_weights.begin() + r * topk, new_weights.begin() + (r + 1) * topk);
        std::sort(old_w.begin(), old_w.end());
        std::sort(new_w.begin(), new_w.end());
        for (int k = 0; k < topk; ++k) {
          old_new_max_err = std::max(old_new_max_err, std::abs(old_w[k] - new_w[k]));
        }
      }
    }

    std::cout << "  M=" << M << " E=" << E << " topk=" << topk << " renorm=" << renormalize << " scale=" << scaling
              << " new_vs_ref_mismatch=" << set_mismatches << " max_weight_err=" << max_weight_err
              << " old_vs_new_mismatch=" << old_new_mismatches << " old_new_max_err=" << old_new_max_err << std::endl;

    // Allow <10% set mismatches (BF16 boundary ties)
    EXPECT_LT(set_mismatches, M * 0.1 + 1) << "Too many expert set mismatches vs CPU reference";
    EXPECT_LT(max_weight_err, 0.02f) << "Weight error too large vs CPU reference";
    // Old vs new should be very close (only when renormalize=false)
    if (!renormalize) {
      EXPECT_LT(old_new_mismatches, M * 0.05 + 1) << "Old and new kernels disagree on expert selection";
      EXPECT_LT(old_new_max_err, 1e-5f) << "Old and new kernels have different weight values";
    }

    cudaFree(d_logits);
    cudaFree(d_weights_old);
    cudaFree(d_ids_old);
    cudaFree(d_weights_new);
    cudaFree(d_ids_new);
  }

  // Performance comparison
  template <typename T>
  void BenchmarkBothKernels(int M, int E, int topk, bool renormalize, float scaling) {
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> h_logits(M * E);
    for (auto& v : h_logits) v = dist(gen);
    std::vector<T> h_logits_t(M * E);
    for (int i = 0; i < M * E; ++i) h_logits_t[i] = static_cast<T>(h_logits[i]);

    void *d_logits, *d_weights, *d_ids;
    cudaMalloc(&d_logits, M * E * sizeof(T));
    cudaMalloc(&d_weights, M * topk * sizeof(float));
    cudaMalloc(&d_ids, M * topk * sizeof(int32_t));
    cudaMemcpy(d_logits, h_logits_t.data(), M * E * sizeof(T), cudaMemcpyHostToDevice);

    auto old_kernel = [&]() {
      InvokeBasicSoftmaxTopk<T>(d_logits, d_weights, d_ids, M, E, topk, scaling, stream_);
    };
    auto new_kernel = [&]() {
      InvokeTopkGatingSoftmax<T>(d_logits, d_weights, d_ids, M, E, topk, renormalize, scaling, stream_);
    };

    float old_ms = llm_kernels::utils::MeasureCudaExecutionTime(old_kernel, stream_, 10, 100);
    float new_ms = llm_kernels::utils::MeasureCudaExecutionTime(new_kernel, stream_, 10, 100);

    std::cout << "  M=" << M << " E=" << E << " topk=" << topk
              << " OLD=" << old_ms * 1000 << "us"
              << " NEW=" << new_ms * 1000 << "us"
              << " speedup=" << old_ms / new_ms << "x" << std::endl;

    cudaFree(d_logits);
    cudaFree(d_weights);
    cudaFree(d_ids);
  }
};

// ========= Correctness tests =========
TEST_F(TopkGatingSoftmaxTest, BF16_Qwen3_Renorm) { TestBothKernels<__nv_bfloat16>(32, 128, 8, true, 1.0f); }
TEST_F(TopkGatingSoftmaxTest, BF16_Qwen3_NoRenorm) { TestBothKernels<__nv_bfloat16>(32, 128, 8, false, 1.0f); }
TEST_F(TopkGatingSoftmaxTest, FP16_Qwen3_Renorm) { TestBothKernels<half>(32, 128, 8, true, 1.0f); }
TEST_F(TopkGatingSoftmaxTest, BF16_Decode) { TestBothKernels<__nv_bfloat16>(1, 128, 8, true, 1.0f); }
TEST_F(TopkGatingSoftmaxTest, BF16_Scaled) { TestBothKernels<__nv_bfloat16>(32, 128, 8, true, 2.5f); }
TEST_F(TopkGatingSoftmaxTest, BF16_Small) { TestBothKernels<__nv_bfloat16>(4, 8, 2, true, 1.0f); }
TEST_F(TopkGatingSoftmaxTest, BF16_Large_Batch) { TestBothKernels<__nv_bfloat16>(256, 128, 8, true, 1.0f); }
TEST_F(TopkGatingSoftmaxTest, Float_Qwen3) { TestBothKernels<float>(32, 128, 8, true, 1.0f); }

// ========= Performance tests =========
TEST_F(TopkGatingSoftmaxTest, Perf_BF16_Decode) { BenchmarkBothKernels<__nv_bfloat16>(1, 128, 8, true, 1.0f); }
TEST_F(TopkGatingSoftmaxTest, Perf_BF16_SmallBatch) { BenchmarkBothKernels<__nv_bfloat16>(32, 128, 8, true, 1.0f); }
TEST_F(TopkGatingSoftmaxTest, Perf_BF16_LargeBatch) { BenchmarkBothKernels<__nv_bfloat16>(256, 128, 8, true, 1.0f); }

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
