/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Unit tests for topkSigmoidKernelLauncher: sigmoid routing with aux-loss-free bias
 * correction, top-k renormalization and routed scaling factor.
 *
 * Reference semantics (HF / vLLM grouped_topk with scoring_func="sigmoid"):
 *   s      = sigmoid(logits)                    for ALL experts
 *   select = topk(s + e_score_correction_bias)  bias affects SELECTION ONLY
 *   w      = s[select]                          weights use the UNBIASED scores
 *   if renormalize: w /= sum(w)
 *   w     *= routed_scaling_factor
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "csrc/kernels/nvidia/mixture_of_experts/moe_wrapper.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class TopkSigmoidTestSuit : public testing::Test {
 protected:
  void SetUp() override { CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreate(&stream)); }
  void TearDown() override { CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(stream)); }
  cudaStream_t stream;

  // CPU reference implementation.
  void Reference(const std::vector<float>& logits, const std::vector<float>* bias, int num_rows, int num_experts,
                 int topk, bool renormalize, float scaling, std::vector<float>& out_w, std::vector<int>& out_idx) {
    out_w.assign(num_rows * topk, 0.f);
    out_idx.assign(num_rows * topk, -1);
    for (int r = 0; r < num_rows; ++r) {
      std::vector<float> s(num_experts), sel(num_experts);
      for (int j = 0; j < num_experts; ++j) {
        s[j] = 1.0f / (1.0f + std::exp(-logits[r * num_experts + j]));
        sel[j] = bias ? s[j] + (*bias)[j] : s[j];
      }
      std::vector<int> order(num_experts);
      std::iota(order.begin(), order.end(), 0);
      // stable_sort by descending selection score reproduces the kernel's insertion order,
      // which keeps the lower expert index first on exact ties.
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sel[a] > sel[b]; });
      float sum = 0.f;
      for (int k = 0; k < topk; ++k) sum += s[order[k]];
      float scale = scaling;
      if (renormalize && sum > 0.f) scale = scaling / sum;
      for (int k = 0; k < topk; ++k) {
        out_idx[r * topk + k] = order[k];
        out_w[r * topk + k] = s[order[k]] * scale;
      }
    }
  }

  void RunCase(int num_rows, int num_experts, int topk, bool use_bias, bool renormalize, float scaling,
               unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> nd(0.f, 1.5f);
    std::vector<float> h_logits(num_rows * num_experts);
    for (auto& v : h_logits) v = nd(gen);
    std::vector<float> h_bias(num_experts);
    std::normal_distribution<float> bd(0.f, 0.0114f);  // matches Hy3 checkpoint bias std
    for (auto& v : h_bias) v = bd(gen);

    float *d_logits = nullptr, *d_out = nullptr, *d_bias = nullptr;
    int *d_idx = nullptr, *d_src = nullptr;
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_logits, h_logits.size() * sizeof(float)));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_out, num_rows * topk * sizeof(float)));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_idx, num_rows * topk * sizeof(int)));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_src, num_rows * topk * sizeof(int)));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice));
    if (use_bias) {
      CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));
      CHECK_NVIDIA_CUDA_ERROR(
          cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    topkSigmoidKernelLauncher(d_logits, d_out, d_idx, d_src, num_rows, num_experts, topk, stream, d_bias, renormalize,
                              scaling);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    std::vector<float> got_w(num_rows * topk);
    std::vector<int> got_idx(num_rows * topk);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(got_w.data(), d_out, got_w.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(got_idx.data(), d_idx, got_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<float> ref_w;
    std::vector<int> ref_idx;
    Reference(h_logits, use_bias ? &h_bias : nullptr, num_rows, num_experts, topk, renormalize, scaling, ref_w,
              ref_idx);

    for (int i = 0; i < num_rows * topk; ++i) {
      EXPECT_EQ(got_idx[i], ref_idx[i]) << "expert index mismatch at " << i;
      EXPECT_NEAR(got_w[i], ref_w[i], 1e-5f) << "weight mismatch at " << i;
    }

    // Renormalization invariant: selected weights must sum to routed_scaling_factor.
    if (renormalize) {
      for (int r = 0; r < num_rows; ++r) {
        float sum = 0.f;
        for (int k = 0; k < topk; ++k) sum += got_w[r * topk + k];
        EXPECT_NEAR(sum, scaling, 1e-4f) << "row " << r << " weights must sum to routed_scaling_factor";
      }
    }

    cudaFree(d_logits);
    cudaFree(d_out);
    cudaFree(d_idx);
    cudaFree(d_src);
    if (d_bias) cudaFree(d_bias);
  }
};

// Hy3-FP8 shape: 192 experts, top-8, route_norm=true, router_scaling_factor=2.826.
TEST_F(TopkSigmoidTestSuit, Hy3ShapeBiasRenormScaling) { RunCase(64, 192, 8, true, true, 2.826f, 1234); }

// Each new feature in isolation.
TEST_F(TopkSigmoidTestSuit, BiasOnlyAffectsSelection) { RunCase(32, 192, 8, true, false, 1.0f, 7); }
TEST_F(TopkSigmoidTestSuit, RenormalizeOnly) { RunCase(32, 192, 8, false, true, 1.0f, 8); }
TEST_F(TopkSigmoidTestSuit, ScalingOnly) { RunCase(32, 192, 8, false, false, 2.826f, 9); }

// Backwards compatibility: no bias, no renorm, scaling 1.0 must reproduce the legacy
// behaviour, i.e. plain sigmoid of the top-k raw logits.
TEST_F(TopkSigmoidTestSuit, LegacyBehaviourUnchanged) {
  int num_rows = 16, num_experts = 192, topk = 8;
  std::mt19937 gen(42);
  std::normal_distribution<float> nd(0.f, 1.5f);
  std::vector<float> h_logits(num_rows * num_experts);
  for (auto& v : h_logits) v = nd(gen);

  float *d_logits = nullptr, *d_out = nullptr;
  int *d_idx = nullptr, *d_src = nullptr;
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_logits, h_logits.size() * sizeof(float)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_out, num_rows * topk * sizeof(float)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_idx, num_rows * topk * sizeof(int)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_src, num_rows * topk * sizeof(int)));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Called exactly as the pre-change code did (defaults for the new trailing args).
  topkSigmoidKernelLauncher(d_logits, d_out, d_idx, d_src, num_rows, num_experts, topk, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<float> got_w(num_rows * topk);
  std::vector<int> got_idx(num_rows * topk);
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(got_w.data(), d_out, got_w.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(got_idx.data(), d_idx, got_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));

  for (int r = 0; r < num_rows; ++r) {
    std::vector<int> order(num_experts);
    std::iota(order.begin(), order.end(), 0);
    const float* row = h_logits.data() + r * num_experts;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return row[a] > row[b]; });
    for (int k = 0; k < topk; ++k) {
      EXPECT_EQ(got_idx[r * topk + k], order[k]);
      EXPECT_NEAR(got_w[r * topk + k], 1.0f / (1.0f + std::exp(-row[order[k]])), 1e-6f);
    }
  }
  cudaFree(d_logits);
  cudaFree(d_out);
  cudaFree(d_idx);
  cudaFree(d_src);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
