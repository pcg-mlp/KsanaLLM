/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <xpu/runtime.h>
#include <xpu/xdnn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "test.h"

#include "runtime/xpu_context.h"

#include "device/kunlun/xpu_utils.h"
#include "kernels/kunlun/basic_kernel_wrapper.h"
#include "kernels/kunlun/moe_kernel_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {

class XpuTimer {
 public:
  XpuTimer(const std::string& name, XPUStream stream) : name_(name), stream_(stream) {
    XPU_CHECK(xpu_wait(stream_));
    start_ = std::chrono::steady_clock::now();
  }
  ~XpuTimer() {
    XPU_CHECK(xpu_wait(stream_));
    auto ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_).count();
    KLLM_LOG_INFO << fmt::format("{} cost: {:.3f} ms", name_, ms);
  }

 private:
  std::string name_;
  XPUStream stream_;
  std::chrono::steady_clock::time_point start_;
};

class KunlunMoeKernelTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    XPU_CHECK(xpu_current_device(&device_));
    XPU_CHECK(xpu_stream_create(&stream_));
    ctx_ = xdnn::create_context();
    ctx_->set_stream(stream_);
    xpuctx_ = xmlir_rt::getXpuKernelContext(/*useCpu=*/false);
    xpuctx_->set_stream(stream_, false);
    XPU_CHECK(xpu_wait(stream_));
  }

  static void TearDownTestSuite() { XPU_CHECK(xpu_wait(stream_)); }

  void SetUp() override { XPU_CHECK(xpu_wait(stream_)); }
  void TearDown() override { XPU_CHECK(xpu_wait(stream_)); }

 protected:
  XPUStream& stream = stream_;
  xdnn::Context*& ctx = ctx_;
  xpytorch::xpu::api::Context*& xpuctx = xpuctx_;
  int& device = device_;

  static int device_;
  static XPUStream stream_;
  static xdnn::Context* ctx_;
  static xpytorch::xpu::api::Context* xpuctx_;
};

int KunlunMoeKernelTest::device_ = -1;
XPUStream KunlunMoeKernelTest::stream_ = nullptr;
xdnn::Context* KunlunMoeKernelTest::ctx_ = nullptr;
xpytorch::xpu::api::Context* KunlunMoeKernelTest::xpuctx_ = nullptr;

#ifdef ENABLE_KUNLUN

// ═══════════════════════════════════════════════════════════════════════════════
// XpuMoeSoftmaxTopk<bfloat16, float, int>
// gate [m, expert_num] → topk_score [m, topk], topk_idx [m, topk]
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, MoeSoftmaxTopk) {
  constexpr int64_t m = 4, expert_num = 8, topk = 2;
  const int64_t ncluster = ctx->ncluster();
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> host_gate(m * expert_num);
  for (auto& v : host_gate) v = dist(rng);

  // CPU reference: softmax per row, then pick top-K
  std::vector<float> ref_scores(m * topk);
  std::vector<int> ref_indices(m * topk);
  for (int i = 0; i < m; ++i) {
    // softmax
    float max_val = *std::max_element(&host_gate[i * expert_num], &host_gate[(i + 1) * expert_num]);
    std::vector<float> sm(expert_num);
    float sum = 0.0f;
    for (int e = 0; e < expert_num; ++e) {
      sm[e] = std::exp(host_gate[i * expert_num + e] - max_val);
      sum += sm[e];
    }
    for (int e = 0; e < expert_num; ++e) sm[e] /= sum;

    // topk
    std::vector<int> idx(expert_num);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int b) { return sm[a] > sm[b]; });
    for (int t = 0; t < topk; ++t) {
      ref_indices[i * topk + t] = idx[t];
      ref_scores[i * topk + t] = sm[idx[t]];
    }
  }

  // Upload to device
  float* dev_fgate;
  bfloat16* dev_gate;
  float* dev_scores;
  int* dev_indices;
  int32_t* dev_block_stat;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fgate), m * expert_num * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_gate), m * expert_num * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_scores), m * topk * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_indices), m * topk * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(
      xpu_malloc(reinterpret_cast<void**>(&dev_block_stat), ncluster * expert_num * sizeof(int32_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fgate, host_gate.data(), m * expert_num * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fgate, dev_gate, m * expert_num)));

  {
    XpuTimer timer("XpuMoeSoftmaxTopk<bf16>", stream);
    ASSERT_EQ(0, (XpuMoeSoftmaxTopk<bfloat16, float, int>(ctx, dev_gate, dev_scores, dev_indices, dev_block_stat, m,
                                                          expert_num, topk, ncluster)));
  }

  std::vector<float> host_scores(m * topk);
  std::vector<int> host_indices(m * topk);
  XPU_CHECK(xpu_memcpy(host_scores.data(), dev_scores, m * topk * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(host_indices.data(), dev_indices, m * topk * sizeof(int), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m; ++i) {
    // Verify top-K indices are valid experts and scores are close
    std::set<int> ref_set(ref_indices.begin() + i * topk, ref_indices.begin() + (i + 1) * topk);
    for (int t = 0; t < topk; ++t) {
      EXPECT_GE(host_indices[i * topk + t], 0);
      EXPECT_LT(host_indices[i * topk + t], expert_num);
    }
    // Verify sum of topk scores is close (order may differ for equal values)
    float ref_sum = 0.0f, got_sum = 0.0f;
    for (int t = 0; t < topk; ++t) {
      ref_sum += ref_scores[i * topk + t];
      got_sum += host_scores[i * topk + t];
    }
    EXPECT_NEAR(got_sum, ref_sum, 0.05f) << "score sum mismatch at row " << i;
  }

  XPU_CHECK(xpu_free(dev_fgate));
  XPU_CHECK(xpu_free(dev_gate));
  XPU_CHECK(xpu_free(dev_scores));
  XPU_CHECK(xpu_free(dev_indices));
  XPU_CHECK(xpu_free(dev_block_stat));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuMoeSoftmaxTopkNorm<bfloat16, float, int>
// gate [m, expert_num] → norm_score [m, topk], topk_idx [m, topk]
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, MoeSoftmaxTopkNorm) {
  constexpr int64_t m = 4, expert_num = 8, topk = 2;
  const int64_t ncluster = ctx->ncluster();
  std::mt19937 rng(100);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> host_gate(m * expert_num);
  for (auto& v : host_gate) v = dist(rng);

  // CPU reference: softmax → topk → renormalize
  std::vector<float> ref_scores(m * topk);
  for (int i = 0; i < m; ++i) {
    float max_val = *std::max_element(&host_gate[i * expert_num], &host_gate[(i + 1) * expert_num]);
    std::vector<float> sm(expert_num);
    float sum = 0.0f;
    for (int e = 0; e < expert_num; ++e) {
      sm[e] = std::exp(host_gate[i * expert_num + e] - max_val);
      sum += sm[e];
    }
    for (int e = 0; e < expert_num; ++e) sm[e] /= sum;

    std::vector<int> idx(expert_num);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int b) { return sm[a] > sm[b]; });

    float topk_sum = 0.0f;
    for (int t = 0; t < topk; ++t) topk_sum += sm[idx[t]];
    for (int t = 0; t < topk; ++t) ref_scores[i * topk + t] = sm[idx[t]] / topk_sum;
  }

  float* dev_fgate;
  bfloat16* dev_gate;
  float* dev_scores;
  int* dev_indices;
  int32_t* dev_block_stat;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fgate), m * expert_num * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_gate), m * expert_num * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_scores), m * topk * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_indices), m * topk * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(
      xpu_malloc(reinterpret_cast<void**>(&dev_block_stat), ncluster * expert_num * sizeof(int32_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fgate, host_gate.data(), m * expert_num * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fgate, dev_gate, m * expert_num)));

  {
    XpuTimer timer("XpuMoeSoftmaxTopkNorm<bf16>", stream);
    ASSERT_EQ(0, (XpuMoeSoftmaxTopkNorm<bfloat16, float, int>(ctx, dev_gate, dev_scores, dev_indices, dev_block_stat, m,
                                                              expert_num, topk, ncluster, /*stable=*/true)));
  }

  std::vector<float> host_scores(m * topk);
  XPU_CHECK(xpu_memcpy(host_scores.data(), dev_scores, m * topk * sizeof(float), XPU_DEVICE_TO_HOST));

  // Verify normalized scores sum to ~1.0 per row
  for (int i = 0; i < m; ++i) {
    float row_sum = 0.0f;
    for (int t = 0; t < topk; ++t) row_sum += host_scores[i * topk + t];
    EXPECT_NEAR(row_sum, 1.0f, 0.05f) << "norm score sum != 1.0 at row " << i;
  }

  XPU_CHECK(xpu_free(dev_fgate));
  XPU_CHECK(xpu_free(dev_gate));
  XPU_CHECK(xpu_free(dev_scores));
  XPU_CHECK(xpu_free(dev_indices));
  XPU_CHECK(xpu_free(dev_block_stat));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuMoeSigmoidTopkNorm<bfloat16, float, int>
// gate [m, expert_num] + bias [expert_num] → topk_score [m, topk]
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, MoeSigmoidTopkNorm) {
  constexpr int64_t m = 4, expert_num = 16, topk = 2;
  constexpr int64_t num_expert_group = 2, topk_group = 1;
  constexpr float scale = 1.0f;
  const int64_t ncluster = ctx->ncluster();
  std::mt19937 rng(200);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<float> host_gate(m * expert_num);
  std::vector<float> host_bias(expert_num);
  for (auto& v : host_gate) v = dist(rng);
  for (auto& v : host_bias) v = dist(rng) * 0.1f;

  float* dev_fgate;
  bfloat16* dev_gate;
  float* dev_bias;
  float* dev_scores;
  int* dev_indices;
  int32_t* dev_block_stat;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fgate), m * expert_num * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_gate), m * expert_num * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_bias), expert_num * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_scores), m * topk * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_indices), m * topk * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(
      xpu_malloc(reinterpret_cast<void**>(&dev_block_stat), ncluster * expert_num * sizeof(int32_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fgate, host_gate.data(), m * expert_num * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fgate, dev_gate, m * expert_num)));
  XPU_CHECK(xpu_memcpy(dev_bias, host_bias.data(), expert_num * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuMoeSigmoidTopkNorm<bf16>", stream);
    ASSERT_EQ(0, (XpuMoeSigmoidTopkNorm<bfloat16, float, int>(ctx, dev_gate, dev_bias, scale, dev_scores, dev_indices,
                                                              dev_block_stat, m, expert_num, num_expert_group,
                                                              topk_group, topk, ncluster)));
  }

  std::vector<float> host_scores(m * topk);
  std::vector<int> host_indices(m * topk);
  XPU_CHECK(xpu_memcpy(host_scores.data(), dev_scores, m * topk * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(host_indices.data(), dev_indices, m * topk * sizeof(int), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m; ++i) {
    for (int t = 0; t < topk; ++t) {
      EXPECT_GE(host_indices[i * topk + t], 0);
      EXPECT_LT(host_indices[i * topk + t], expert_num);
      EXPECT_GT(host_scores[i * topk + t], 0.0f);
    }
  }

  XPU_CHECK(xpu_free(dev_fgate));
  XPU_CHECK(xpu_free(dev_gate));
  XPU_CHECK(xpu_free(dev_bias));
  XPU_CHECK(xpu_free(dev_scores));
  XPU_CHECK(xpu_free(dev_indices));
  XPU_CHECK(xpu_free(dev_block_stat));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuMoeFusedGateKimiK2<bfloat16, float, int>
// Sigmoid + TopK (Kimi-K2.6)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, MoeFusedGateKimiK2) {
  constexpr int64_t m = 4, expert_num = 8, topk = 2;
  constexpr float routed_scaling_factor = 2.0f;
  const int64_t ncluster = ctx->ncluster();
  std::mt19937 rng(300);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> host_gate(m * expert_num);
  std::vector<float> host_bias(expert_num);
  for (auto& v : host_gate) v = dist(rng);
  for (auto& v : host_bias) v = dist(rng) * 0.1f;

  float* dev_fgate;
  float* dev_fbias;
  bfloat16* dev_gate;
  bfloat16* dev_bias;
  float* dev_scores;
  int* dev_indices;
  int32_t* dev_block_stat;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fgate), m * expert_num * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fbias), expert_num * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_gate), m * expert_num * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_bias), expert_num * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_scores), m * topk * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_indices), m * topk * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(
      xpu_malloc(reinterpret_cast<void**>(&dev_block_stat), ncluster * expert_num * sizeof(int32_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fgate, host_gate.data(), m * expert_num * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_fbias, host_bias.data(), expert_num * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fgate, dev_gate, m * expert_num)));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fbias, dev_bias, expert_num)));

  {
    XpuTimer timer("XpuMoeFusedGateKimiK2<bf16>", stream);
    ASSERT_EQ(0, (XpuMoeFusedGateKimiK2<bfloat16, float, int>(ctx, dev_gate, dev_bias, dev_scores, dev_indices, m,
                                                              expert_num, topk, /*renormalize=*/true,
                                                              routed_scaling_factor, dev_block_stat, ncluster)));
  }

  std::vector<float> host_scores(m * topk);
  std::vector<int> host_indices(m * topk);
  XPU_CHECK(xpu_memcpy(host_scores.data(), dev_scores, m * topk * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(host_indices.data(), dev_indices, m * topk * sizeof(int), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m; ++i) {
    for (int t = 0; t < topk; ++t) {
      EXPECT_GE(host_indices[i * topk + t], 0);
      EXPECT_LT(host_indices[i * topk + t], expert_num);
      EXPECT_GT(host_scores[i * topk + t], 0.0f);
    }
  }

  XPU_CHECK(xpu_free(dev_fgate));
  XPU_CHECK(xpu_free(dev_fbias));
  XPU_CHECK(xpu_free(dev_gate));
  XPU_CHECK(xpu_free(dev_bias));
  XPU_CHECK(xpu_free(dev_scores));
  XPU_CHECK(xpu_free(dev_indices));
  XPU_CHECK(xpu_free(dev_block_stat));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuMoeFfnPreSorted<bfloat16, int>
// Scatter tokens to experts based on topk routing
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, MoeFfnPreSorted) {
  constexpr int64_t m = 4, n = 16, expert_num = 4, topk = 2;
  const int64_t ncluster = ctx->ncluster();
  std::mt19937 rng(400);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> host_x(m * n);
  for (auto& v : host_x) v = dist(rng);

  // Generate valid topk indices: each row picks topk experts from [0, expert_num)
  std::vector<int> host_topk_idx(m * topk);
  for (int i = 0; i < m; ++i) {
    std::vector<int> perm(expert_num);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int t = 0; t < topk; ++t) {
      host_topk_idx[i * topk + t] = perm[t];
    }
  }

  // Generate block_statistic via kernel
  float* dev_fx;
  bfloat16* dev_x;
  int* dev_topk_idx;
  int32_t* dev_block_stat;
  bfloat16* dev_sorted_x;
  int* dev_moe_index;
  int* dev_expert_m;
  int* dev_lod;

  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fx), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_topk_idx), m * topk * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(
      xpu_malloc(reinterpret_cast<void**>(&dev_block_stat), ncluster * expert_num * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_sorted_x), m * topk * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_moe_index), m * topk * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_expert_m), expert_num * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_lod), (expert_num + 1) * sizeof(int), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fx, host_x.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fx, dev_x, m * n)));
  XPU_CHECK(xpu_memcpy(dev_topk_idx, host_topk_idx.data(), m * topk * sizeof(int), XPU_HOST_TO_DEVICE));

  // Generate block_statistic
  ASSERT_EQ(0, (XpuGenBlockStatistic<int>(ctx, dev_topk_idx, dev_block_stat, m, expert_num, topk, ncluster)));

  {
    XpuTimer timer("XpuMoeFfnPreSorted<bf16>", stream);
    ASSERT_EQ(0,
              (XpuMoeFfnPreSorted<bfloat16, int>(ctx, dev_x, dev_topk_idx, dev_block_stat, dev_sorted_x, dev_moe_index,
                                                 dev_expert_m, dev_lod, m, n, expert_num, topk, ncluster)));
  }

  // Verify lod is valid: should be monotonically non-decreasing, lod[expert_num] == m*topk
  std::vector<int> host_lod(expert_num + 1);
  XPU_CHECK(xpu_memcpy(host_lod.data(), dev_lod, (expert_num + 1) * sizeof(int), XPU_DEVICE_TO_HOST));
  EXPECT_EQ(host_lod[0], 0);
  for (int e = 1; e <= expert_num; ++e) {
    EXPECT_GE(host_lod[e], host_lod[e - 1]) << "lod not monotonic at expert " << e;
  }
  EXPECT_EQ(host_lod[expert_num], static_cast<int>(m * topk));

  // Verify expert_m sums to m*topk
  std::vector<int> host_expert_m(expert_num);
  XPU_CHECK(xpu_memcpy(host_expert_m.data(), dev_expert_m, expert_num * sizeof(int), XPU_DEVICE_TO_HOST));
  int total = 0;
  for (int e = 0; e < expert_num; ++e) total += host_expert_m[e];
  EXPECT_EQ(total, static_cast<int>(m * topk));

  XPU_CHECK(xpu_free(dev_fx));
  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_topk_idx));
  XPU_CHECK(xpu_free(dev_block_stat));
  XPU_CHECK(xpu_free(dev_sorted_x));
  XPU_CHECK(xpu_free(dev_moe_index));
  XPU_CHECK(xpu_free(dev_expert_m));
  XPU_CHECK(xpu_free(dev_lod));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuSwiglu<bfloat16>
// x [m, 2*n] → z [m, n], SiLU(gate) * up
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, Swiglu) {
  constexpr int64_t m = 4, n = 16;
  std::mt19937 rng(500);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> host_x(m * 2 * n);
  for (auto& v : host_x) v = dist(rng);

  // CPU reference: turn=true means layout [gate|up], gate in first half
  // z[i,j] = silu(x[i,j]) * x[i,n+j]
  std::vector<float> ref(m * n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float gate = host_x[i * 2 * n + j];
      float up = host_x[i * 2 * n + n + j];
      float silu_gate = gate / (1.0f + std::exp(-gate));
      ref[i * n + j] = silu_gate * up;
    }
  }

  float* dev_fx;
  float* dev_fz;
  bfloat16* dev_x;
  bfloat16* dev_z;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fx), m * 2 * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fz), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * 2 * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_z), m * n * sizeof(bfloat16), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fx, host_x.data(), m * 2 * n * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fx, dev_x, m * 2 * n)));

  {
    XpuTimer timer("XpuSwiglu<bf16>", stream);
    ASSERT_EQ(0, (XpuSwiglu<bfloat16>(ctx, dev_x, dev_z, m, n)));
  }

  ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, dev_z, dev_fz, m * n)));
  std::vector<float> host_z(m * n);
  XPU_CHECK(xpu_memcpy(host_z.data(), dev_fz, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m * n; ++i) {
    EXPECT_NEAR(host_z[i], ref[i], 0.1f) << "swiglu mismatch at [" << i << "]";
  }

  XPU_CHECK(xpu_free(dev_fx));
  XPU_CHECK(xpu_free(dev_fz));
  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_z));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuMoeFcFusion<bfloat16, int>
// sorted_x [m*topk, k] × weight [E, n, k]^T → out [m*topk, n]
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, MoeFcFusion) {
  constexpr int64_t m = 4, topk = 2, n = 32, k = 16, expert_num = 4;
  const int64_t m_total = m * topk;
  std::mt19937 rng(600);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  // sorted_x [m_total, k]
  std::vector<float> host_x(m_total * k);
  for (auto& v : host_x) v = dist(rng);

  // weight [expert_num, n, k] (trans_b layout)
  std::vector<float> host_w(expert_num * n * k);
  for (auto& v : host_w) v = dist(rng);

  // Assign each token to an expert: simple round-robin
  // moe_index: for sorted_by_expert, it maps output position → original token index
  // lod: cumulative expert boundary [0, tokens_e0, tokens_e0+tokens_e1, ...]
  std::vector<int> host_moe_index(m_total);
  std::vector<int> host_lod(expert_num + 1, 0);
  int tokens_per_expert = m_total / expert_num;
  for (int e = 0; e < expert_num; ++e) {
    host_lod[e + 1] = host_lod[e] + tokens_per_expert;
  }
  // Distribute remainder
  host_lod[expert_num] = m_total;
  std::iota(host_moe_index.begin(), host_moe_index.end(), 0);

  // CPU reference: per expert, out[lod[e]:lod[e+1], :] = x[lod[e]:lod[e+1], :] * w[e]^T
  std::vector<float> ref(m_total * n, 0.0f);
  for (int e = 0; e < expert_num; ++e) {
    int start = host_lod[e];
    int end = host_lod[e + 1];
    for (int row = start; row < end; ++row) {
      for (int j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
          acc += host_x[row * k + kk] * host_w[e * n * k + j * k + kk];
        }
        ref[row * n + j] = acc;
      }
    }
  }

  float* dev_fx;
  float* dev_fw;
  float* dev_fo;
  bfloat16* dev_x;
  bfloat16* dev_w;
  bfloat16* dev_out;
  int* dev_moe_index;
  int* dev_lod;

  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fx), m_total * k * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fw), expert_num * n * k * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fo), m_total * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m_total * k * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_w), expert_num * n * k * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_out), m_total * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_moe_index), m_total * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_lod), (expert_num + 1) * sizeof(int), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fx, host_x.data(), m_total * k * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_fw, host_w.data(), expert_num * n * k * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fx, dev_x, m_total * k)));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fw, dev_w, expert_num * n * k)));
  XPU_CHECK(xpu_memcpy(dev_moe_index, host_moe_index.data(), m_total * sizeof(int), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_lod, host_lod.data(), (expert_num + 1) * sizeof(int), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuMoeFcFusion<bf16>", stream);
    ASSERT_EQ(0, (XpuMoeFcFusion<bfloat16, int>(ctx, dev_x, dev_w, dev_out, m, topk, m_total, n, k, expert_num,
                                                dev_moe_index, dev_lod)));
  }

  ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, dev_out, dev_fo, m_total * n)));
  std::vector<float> host_out(m_total * n);
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_fo, m_total * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m_total * n; ++i) {
    float abs_err = std::fabs(host_out[i] - ref[i]);
    float scale = std::fabs(ref[i]) + 1.0f;
    EXPECT_LT(abs_err / scale, 0.3f) << "MoeFcFusion mismatch at [" << i << "] got=" << host_out[i]
                                     << " ref=" << ref[i];
  }

  XPU_CHECK(xpu_free(dev_fx));
  XPU_CHECK(xpu_free(dev_fw));
  XPU_CHECK(xpu_free(dev_fo));
  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_w));
  XPU_CHECK(xpu_free(dev_out));
  XPU_CHECK(xpu_free(dev_moe_index));
  XPU_CHECK(xpu_free(dev_lod));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuMoeFfnPostFusionF32Score<bfloat16, int>
// expert_hidden [m*topk, n], moe_index [m*topk], topk_score [m, topk] → y [m, n]
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, MoeFfnPostFusionF32Score) {
  constexpr int64_t m = 4, n = 16, topk = 2;
  std::mt19937 rng(700);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // expert_hidden [m*topk, n]
  std::vector<float> host_hidden(m * topk * n);
  for (auto& v : host_hidden) v = dist(rng);

  // topk_score [m, topk] (positive, sums to ~1 per row)
  std::vector<float> host_scores(m * topk);
  for (int i = 0; i < m; ++i) {
    float s0 = 0.3f + 0.4f * static_cast<float>(rng() % 100) / 100.0f;
    host_scores[i * topk + 0] = s0;
    host_scores[i * topk + 1] = 1.0f - s0;
  }

  // moe_index [m*topk]: maps sorted position → original token*topk position
  // Simple identity for testing (already sorted by token order)
  std::vector<int> host_moe_index(m * topk);
  std::iota(host_moe_index.begin(), host_moe_index.end(), 0);

  // CPU reference: y[i, :] = sum_t(score[i,t] * hidden[moe_index_for(i,t), :])
  // With identity moe_index: y[i, :] = sum_t(score[i,t] * hidden[i*topk+t, :])
  std::vector<float> ref(m * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int t = 0; t < topk; ++t) {
      int src_row = i * topk + t;
      for (int j = 0; j < n; ++j) {
        ref[i * n + j] += host_scores[i * topk + t] * host_hidden[src_row * n + j];
      }
    }
  }

  float* dev_fhidden;
  float* dev_fy;
  bfloat16* dev_hidden;
  bfloat16* dev_y;
  float* dev_scores;
  int* dev_moe_index;

  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fhidden), m * topk * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fy), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_hidden), m * topk * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), m * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_scores), m * topk * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_moe_index), m * topk * sizeof(int), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fhidden, host_hidden.data(), m * topk * n * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fhidden, dev_hidden, m * topk * n)));
  XPU_CHECK(xpu_memcpy(dev_scores, host_scores.data(), m * topk * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_moe_index, host_moe_index.data(), m * topk * sizeof(int), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuMoeFfnPostFusionF32Score<bf16>", stream);
    ASSERT_EQ(
        0, (XpuMoeFfnPostFusionF32Score<bfloat16, int>(ctx, dev_hidden, dev_moe_index, dev_scores, dev_y, m, n, topk)));
  }

  ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, dev_y, dev_fy, m * n)));
  std::vector<float> host_y(m * n);
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_fy, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m * n; ++i) {
    EXPECT_NEAR(host_y[i], ref[i], 0.15f) << "PostFusion mismatch at [" << i << "]";
  }

  XPU_CHECK(xpu_free(dev_fhidden));
  XPU_CHECK(xpu_free(dev_fy));
  XPU_CHECK(xpu_free(dev_hidden));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_scores));
  XPU_CHECK(xpu_free(dev_moe_index));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuMoeEpFfnPreSorted<bfloat16, int>
// EP-aware pre-sort (expert parallel path)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, MoeEpFfnPreSorted) {
  constexpr int64_t m = 4, n = 16, expert_num = 8, topk = 2;
  constexpr int64_t ep_size = 8, ep_rank = 0;
  const int64_t ncluster = ctx->ncluster();
  const int64_t local_experts = expert_num / ep_size;  // 1
  std::mt19937 rng(800);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> host_x(m * n);
  for (auto& v : host_x) v = dist(rng);

  // Generate topk indices within [0, expert_num)
  std::vector<int> host_topk_idx(m * topk);
  for (int i = 0; i < m; ++i) {
    std::vector<int> perm(expert_num);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int t = 0; t < topk; ++t) {
      host_topk_idx[i * topk + t] = perm[t];
    }
  }

  float* dev_fx;
  bfloat16* dev_x;
  int* dev_topk_idx;
  int32_t* dev_block_stat;
  bfloat16* dev_sorted_x;
  int* dev_moe_index;
  int* dev_expert_m;
  int* dev_lod;

  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fx), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_topk_idx), m * topk * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(
      xpu_malloc(reinterpret_cast<void**>(&dev_block_stat), ncluster * expert_num * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_sorted_x), m * topk * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_moe_index), m * topk * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_expert_m), expert_num * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_lod), (local_experts + 1) * sizeof(int), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fx, host_x.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fx, dev_x, m * n)));
  XPU_CHECK(xpu_memcpy(dev_topk_idx, host_topk_idx.data(), m * topk * sizeof(int), XPU_HOST_TO_DEVICE));

  ASSERT_EQ(0, (XpuGenBlockStatistic<int>(ctx, dev_topk_idx, dev_block_stat, m, expert_num, topk, ncluster)));

  {
    XpuTimer timer("XpuMoeEpFfnPreSorted<bf16>", stream);
    ASSERT_EQ(0, (XpuMoeEpFfnPreSorted<bfloat16, int>(ctx, dev_x, dev_topk_idx, dev_block_stat, dev_sorted_x,
                                                      dev_moe_index, dev_expert_m, dev_lod, m, n, expert_num, topk,
                                                      ncluster, ep_size, ep_rank)));
  }

  // Verify lod: lod[local_experts] <= m*topk (may be less since only local experts receive tokens)
  std::vector<int> host_lod(local_experts + 1);
  XPU_CHECK(xpu_memcpy(host_lod.data(), dev_lod, (local_experts + 1) * sizeof(int), XPU_DEVICE_TO_HOST));
  EXPECT_EQ(host_lod[0], 0);
  EXPECT_LE(host_lod[local_experts], static_cast<int>(m * topk));

  XPU_CHECK(xpu_free(dev_fx));
  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_topk_idx));
  XPU_CHECK(xpu_free(dev_block_stat));
  XPU_CHECK(xpu_free(dev_sorted_x));
  XPU_CHECK(xpu_free(dev_moe_index));
  XPU_CHECK(xpu_free(dev_expert_m));
  XPU_CHECK(xpu_free(dev_lod));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuMoeEpFfnPostFusion<bfloat16, int>
// EP-aware post-fusion with float score
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, MoeEpFfnPostFusion) {
  constexpr int64_t m = 4, n = 16, topk = 2;
  std::mt19937 rng(900);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> host_hidden(m * topk * n);
  for (auto& v : host_hidden) v = dist(rng);

  std::vector<float> host_scores(m * topk);
  for (int i = 0; i < m; ++i) {
    float s0 = 0.3f + 0.4f * static_cast<float>(rng() % 100) / 100.0f;
    host_scores[i * topk + 0] = s0;
    host_scores[i * topk + 1] = 1.0f - s0;
  }

  std::vector<int> host_moe_index(m * topk);
  std::iota(host_moe_index.begin(), host_moe_index.end(), 0);

  // CPU reference
  std::vector<float> ref(m * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int t = 0; t < topk; ++t) {
      int src_row = i * topk + t;
      for (int j = 0; j < n; ++j) {
        ref[i * n + j] += host_scores[i * topk + t] * host_hidden[src_row * n + j];
      }
    }
  }

  float* dev_fhidden;
  float* dev_fy;
  bfloat16* dev_hidden;
  bfloat16* dev_y;
  float* dev_scores;
  int* dev_moe_index;

  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fhidden), m * topk * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fy), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_hidden), m * topk * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), m * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_scores), m * topk * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_moe_index), m * topk * sizeof(int), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fhidden, host_hidden.data(), m * topk * n * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fhidden, dev_hidden, m * topk * n)));
  XPU_CHECK(xpu_memcpy(dev_scores, host_scores.data(), m * topk * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_moe_index, host_moe_index.data(), m * topk * sizeof(int), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuMoeEpFfnPostFusion<bf16>", stream);
    ASSERT_EQ(0, (XpuMoeEpFfnPostFusion<bfloat16, int>(ctx, dev_hidden, dev_moe_index, dev_scores, dev_y, m, n, topk)));
  }

  ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, dev_y, dev_fy, m * n)));
  std::vector<float> host_y(m * n);
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_fy, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m * n; ++i) {
    EXPECT_NEAR(host_y[i], ref[i], 0.15f) << "EpPostFusion mismatch at [" << i << "]";
  }

  XPU_CHECK(xpu_free(dev_fhidden));
  XPU_CHECK(xpu_free(dev_fy));
  XPU_CHECK(xpu_free(dev_hidden));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_scores));
  XPU_CHECK(xpu_free(dev_moe_index));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuMoeEpPreSortedQuant2d<bfloat16, int>
// EP-aware fused pre-sort + INT8 quantization (W4A8 path)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, MoeEpPreSortedQuant2d) {
  constexpr int64_t m = 16, n = 1024, expert_num = 8, topk = 2;
  constexpr int64_t ep_size = 8, ep_rank = 0;
  const int64_t ncluster = ctx->ncluster();
  const int64_t local_experts = expert_num / ep_size;  // 1
  std::mt19937 rng(1000);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> host_x(m * n);
  for (auto& v : host_x) v = dist(rng);

  // Ensure at least some tokens route to expert 0 (ep_rank=0's local expert)
  std::vector<int> host_topk_idx(m * topk);
  for (int i = 0; i < m; ++i) {
    host_topk_idx[i * topk + 0] = 0;                           // first choice always expert 0
    host_topk_idx[i * topk + 1] = (i % (expert_num - 1)) + 1;  // second choice varies
  }

  float* dev_fx;
  bfloat16* dev_x;
  int* dev_topk_idx;
  int32_t* dev_block_stat;
  int8_t* dev_quanted_y;
  int* dev_moe_index;
  int* dev_expert_m;
  int* dev_lod;
  float* dev_quant_scale;

  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fx), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_topk_idx), m * topk * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(
      xpu_malloc(reinterpret_cast<void**>(&dev_block_stat), ncluster * expert_num * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_quanted_y), m * topk * n * sizeof(int8_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_moe_index), m * topk * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_expert_m), expert_num * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_lod), (local_experts + 1) * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_quant_scale), m * topk * sizeof(float), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fx, host_x.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fx, dev_x, m * n)));
  XPU_CHECK(xpu_memcpy(dev_topk_idx, host_topk_idx.data(), m * topk * sizeof(int), XPU_HOST_TO_DEVICE));

  ASSERT_EQ(0, (XpuGenBlockStatistic<int>(ctx, dev_topk_idx, dev_block_stat, m, expert_num, topk, ncluster)));

  {
    XpuTimer timer("XpuMoeEpPreSortedQuant2d<bf16>", stream);
    ASSERT_EQ(0, (XpuMoeEpPreSortedQuant2d<bfloat16, int>(ctx, dev_x, dev_topk_idx, dev_block_stat, dev_quanted_y,
                                                          dev_moe_index, dev_expert_m, dev_lod, dev_quant_scale, m, n,
                                                          expert_num, topk, ncluster, ep_size, ep_rank)));
  }

  // Verify quant_scale is positive for tokens routed to this EP rank
  std::vector<int> host_lod(local_experts + 1);
  XPU_CHECK(xpu_memcpy(host_lod.data(), dev_lod, (local_experts + 1) * sizeof(int), XPU_DEVICE_TO_HOST));
  int local_tokens = host_lod[local_experts];

  if (local_tokens > 0) {
    std::vector<float> host_scale(m * topk);
    XPU_CHECK(xpu_memcpy(host_scale.data(), dev_quant_scale, local_tokens * sizeof(float), XPU_DEVICE_TO_HOST));
    for (int i = 0; i < local_tokens; ++i) {
      EXPECT_GT(host_scale[i], 0.0f) << "quant_scale should be positive at [" << i << "]";
    }
  }

  XPU_CHECK(xpu_free(dev_fx));
  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_topk_idx));
  XPU_CHECK(xpu_free(dev_block_stat));
  XPU_CHECK(xpu_free(dev_quanted_y));
  XPU_CHECK(xpu_free(dev_moe_index));
  XPU_CHECK(xpu_free(dev_expert_m));
  XPU_CHECK(xpu_free(dev_lod));
  XPU_CHECK(xpu_free(dev_quant_scale));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuMoeW4A8FcFusion<bfloat16, int>
// INT8 activation × INT4x2 packed weight expert projection
// Follows kunlun_w4a8_layer_test.cpp MoeW4A8KernelFcFusion exactly
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, MoeW4A8FcFusion) {
  constexpr int64_t m_total = 8, n = 128, k = 64, expert_num = 4;
  const int64_t packed_k = k / 2;
  std::mt19937 rng(99);
  std::uniform_int_distribution<int> int_dist(-60, 60);
  std::uniform_real_distribution<float> scale_dist(0.005f, 0.02f);

  // Input activation [m_total, k] INT8
  std::vector<int8_t> host_act(m_total * k);
  for (auto& v : host_act) v = static_cast<int8_t>(int_dist(rng));

  // Per-token activation scale [m_total] FP32
  std::vector<float> host_act_scale(m_total);
  for (auto& v : host_act_scale) v = scale_dist(rng);

  // Weight [expert_num, n, k/2] INT4x2 packed
  // Each byte stores 2 INT4 values: low_nibble (even k) | high_nibble (odd k)
  std::vector<int8_t> host_w(expert_num * n * packed_k);
  std::uniform_int_distribution<int> nibble_dist(-7, 7);
  for (auto& v : host_w) {
    int low = nibble_dist(rng) & 0xF;
    int high = nibble_dist(rng) & 0xF;
    v = static_cast<int8_t>(low | (high << 4));
  }

  // Per-channel weight scale [expert_num * n] FP32 (×7 format for SDK)
  std::vector<float> host_w_scale(expert_num * n);
  for (auto& v : host_w_scale) v = scale_dist(rng);

  // Routing: tokens evenly distributed across experts
  // sorted_ids: identity, lod: [0, 2, 4, 6, 8]
  std::vector<int> host_moe_index(m_total);
  std::iota(host_moe_index.begin(), host_moe_index.end(), 0);
  std::vector<int> host_lod(expert_num + 1);
  int tokens_per_expert = m_total / expert_num;
  for (int e = 0; e <= expert_num; ++e) host_lod[e] = e * tokens_per_expert;

  // CPU reference: y[t,j] = sum_kk(act[t,kk] * unpack_int4(w[expert,j,kk/2], kk%2)) * act_scale[t] *
  // w_scale[expert*n+j] / (127*7)
  std::vector<float> ref(m_total * n, 0.0f);
  for (int t = 0; t < m_total; ++t) {
    int expert_id = -1;
    for (int e = 0; e < expert_num; ++e) {
      if (t >= host_lod[e] && t < host_lod[e + 1]) {
        expert_id = e;
        break;
      }
    }
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        int packed_idx = expert_id * n * packed_k + j * packed_k + kk / 2;
        int8_t packed = host_w[packed_idx];
        int w_val;
        if (kk % 2 == 0) {
          w_val = (packed & 0xF);
          if (w_val >= 8) w_val -= 16;
        } else {
          w_val = ((packed >> 4) & 0xF);
          if (w_val >= 8) w_val -= 16;
        }
        sum += static_cast<float>(host_act[t * k + kk]) * static_cast<float>(w_val);
      }
      ref[t * n + j] = sum * host_act_scale[t] * host_w_scale[expert_id * n + j] / (127.0f * 7.0f);
    }
  }

  int8_t* dev_act;
  void* dev_w;
  float* dev_act_scale;
  float* dev_w_scale;
  bfloat16* dev_out;
  float* dev_fo;
  int* dev_moe_index;
  int* dev_lod;

  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_act), m_total * k * sizeof(int8_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_w), expert_num * n * packed_k * sizeof(int8_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_act_scale), m_total * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_w_scale), expert_num * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_out), m_total * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fo), m_total * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_moe_index), m_total * sizeof(int), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_lod), (expert_num + 1) * sizeof(int), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_act, host_act.data(), m_total * k * sizeof(int8_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_w, host_w.data(), expert_num * n * packed_k * sizeof(int8_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_act_scale, host_act_scale.data(), m_total * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_w_scale, host_w_scale.data(), expert_num * n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_moe_index, host_moe_index.data(), m_total * sizeof(int), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_lod, host_lod.data(), (expert_num + 1) * sizeof(int), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuMoeW4A8FcFusion<bf16>", stream);
    ASSERT_EQ(0, (XpuMoeW4A8FcFusion<bfloat16, int>(ctx, dev_act, dev_w, dev_out, dev_act_scale, dev_w_scale,
                                                    /*tokens=*/m_total, /*topk=*/1, m_total, n, k, expert_num,
                                                    dev_moe_index, dev_lod)));
  }

  ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, dev_out, dev_fo, m_total * n)));
  std::vector<float> host_out(m_total * n);
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_fo, m_total * n * sizeof(float), XPU_DEVICE_TO_HOST));

  // Relative error check (same as kunlun_w4a8_layer_test.cpp CpuRelError style)
  float max_abs_ref = 0.0f;
  for (auto& v : ref) max_abs_ref = std::max(max_abs_ref, std::fabs(v));
  float skip_threshold = max_abs_ref * 0.01f;
  float max_rel = 0.0f;
  int checked = 0;
  for (int i = 0; i < m_total * n; ++i) {
    if (std::fabs(ref[i]) < skip_threshold) continue;
    float rel = std::fabs(host_out[i] - ref[i]) / std::fabs(ref[i]);
    if (rel > max_rel) max_rel = rel;
    ++checked;
  }
  KLLM_LOG_INFO << fmt::format("MoeW4A8FcFusion: max_rel_err={:.4f} checked={}/{}", max_rel, checked,
                               static_cast<int>(m_total * n));
  EXPECT_LT(max_rel, 0.15f) << "W4A8 FcFusion relative error too large";

  XPU_CHECK(xpu_free(dev_act));
  XPU_CHECK(xpu_free(dev_w));
  XPU_CHECK(xpu_free(dev_act_scale));
  XPU_CHECK(xpu_free(dev_w_scale));
  XPU_CHECK(xpu_free(dev_out));
  XPU_CHECK(xpu_free(dev_fo));
  XPU_CHECK(xpu_free(dev_moe_index));
  XPU_CHECK(xpu_free(dev_lod));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuSiluAndMulQuantUnt<bfloat16>
// Fused SwiGLU + INT8 quantization
// x [m, 2*n] → y [m, n, int8] + scale [m]
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeKernelTest, SiluAndMulQuantUnt) {
  constexpr int64_t m = 4, n = 32;
  const int64_t input_n = 2 * n;
  std::mt19937 rng(1200);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> host_x(m * input_n);
  for (auto& v : host_x) v = dist(rng);

  // CPU reference: swiglu + per-row quantization
  // turn=true: gate in first half, up in second half
  // z[i,j] = silu(x[i,j]) * x[i,n+j], then quantize each row to int8
  std::vector<float> ref_z(m * n);
  std::vector<float> ref_scale(m);
  for (int i = 0; i < m; ++i) {
    float row_max_abs = 0.0f;
    for (int j = 0; j < n; ++j) {
      float gate = host_x[i * input_n + j];
      float up = host_x[i * input_n + n + j];
      float silu_gate = gate / (1.0f + std::exp(-gate));
      ref_z[i * n + j] = silu_gate * up;
      row_max_abs = std::max(row_max_abs, std::fabs(ref_z[i * n + j]));
    }
    ref_scale[i] = row_max_abs / 127.0f;
  }

  // recv_counts: pointer to lod[E] (end boundary). For single-expert test, just use m
  int32_t host_recv_counts = static_cast<int32_t>(m);

  float* dev_fx;
  bfloat16* dev_x;
  int8_t* dev_y;
  float* dev_scale;
  int32_t* dev_recv_counts;

  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fx), m * input_n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * input_n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), m * n * sizeof(int8_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_scale), m * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_recv_counts), sizeof(int32_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fx, host_x.data(), m * input_n * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fx, dev_x, m * input_n)));
  XPU_CHECK(xpu_memcpy(dev_recv_counts, &host_recv_counts, sizeof(int32_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuSiluAndMulQuantUnt<bf16>", stream);
    ASSERT_EQ(0, (XpuSiluAndMulQuantUnt<bfloat16>(ctx, dev_x, dev_y, dev_scale, m, input_n, m, dev_recv_counts)));
  }

  std::vector<float> host_scale(m);
  std::vector<int8_t> host_y(m * n);
  XPU_CHECK(xpu_memcpy(host_scale.data(), dev_scale, m * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, m * n * sizeof(int8_t), XPU_DEVICE_TO_HOST));

  // Verify scales are positive
  for (int i = 0; i < m; ++i) {
    EXPECT_GT(host_scale[i], 0.0f) << "scale should be positive at row " << i;
  }

  // Verify dequantized output is close to reference
  // Scale is in max_abs format, so dequant = int8_val * scale / 127
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float dequant = static_cast<float>(host_y[i * n + j]) * host_scale[i] / 127.0f;
      float abs_err = std::fabs(dequant - ref_z[i * n + j]);
      float scale = std::fabs(ref_z[i * n + j]) + 1.0f;
      EXPECT_LT(abs_err / scale, 0.3f) << "SiluAndMulQuantUnt mismatch at [" << i << "," << j << "] dequant=" << dequant
                                       << " ref=" << ref_z[i * n + j];
    }
  }

  XPU_CHECK(xpu_free(dev_fx));
  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_scale));
  XPU_CHECK(xpu_free(dev_recv_counts));
}

#endif  // ENABLE_KUNLUN

}  // namespace ksana_llm
