/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <xpu/runtime.h>
#include <xpu/xdnn.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "runtime/xpu_context.h"  // xmlir_rt::getXpuKernelContext

// Suppress framework XPU_CHECK macro to avoid conflict
#ifdef XPU_CHECK
#  undef XPU_CHECK
#endif

#include "kernels/kunlun/moe_kernel_wrapper.h"
#include "test.h"

// MoE infer_ops header
#include "infer_ops.h"

// NOTE: xblas::moe_fc_fusion doesn't have fp32 template instantiation in libxpu_blas.so
// (only float16/bfloat16 variants exist). For fp32 testing, we use per-expert xdnn::fc
// calls (Plan C from design doc). Production code would use xblas with half precision.

// xblas header for half-precision MoE FC tests
#ifdef ENABLE_KUNLUN
#  include "xblas_legacy_api.h"
namespace xblas = baidu::xpu::xblas;
#endif

namespace ksana_llm {

// ═══════════════════════════════════════════════════════════════════════════════
// Helper utilities (no macros to avoid preprocessor issues with template commas)
// ═══════════════════════════════════════════════════════════════════════════════

// Helper: allocate device memory and copy from host
template <typename T>
T* AllocAndCopy(const std::vector<T>& host_data) {
  T* dev_ptr = nullptr;
  if (xpu_malloc(reinterpret_cast<void**>(&dev_ptr), host_data.size() * sizeof(T)) != 0) {
    return nullptr;
  }
  xpu_memcpy(dev_ptr, host_data.data(), host_data.size() * sizeof(T), XPU_HOST_TO_DEVICE);
  return dev_ptr;
}

// Helper: allocate device memory (zero-initialized)
template <typename T>
T* AllocDevice(size_t num_elements) {
  T* dev_ptr = nullptr;
  if (xpu_malloc(reinterpret_cast<void**>(&dev_ptr), num_elements * sizeof(T)) != 0) {
    return nullptr;
  }
  std::vector<T> zeros(num_elements, T(0));
  xpu_memcpy(dev_ptr, zeros.data(), num_elements * sizeof(T), XPU_HOST_TO_DEVICE);
  return dev_ptr;
}

// Helper: copy device to host
template <typename T>
void CopyToHost(std::vector<T>& host_data, const T* dev_ptr) {
  xpu_memcpy(host_data.data(), dev_ptr, host_data.size() * sizeof(T), XPU_DEVICE_TO_HOST);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU reference implementations
// ═══════════════════════════════════════════════════════════════════════════════

void CpuSoftmax(const float* input, float* output, int m, int n) {
  for (int i = 0; i < m; ++i) {
    float max_val = input[i * n];
    for (int j = 1; j < n; ++j) max_val = std::max(max_val, input[i * n + j]);
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
      output[i * n + j] = std::exp(input[i * n + j] - max_val);
      sum += output[i * n + j];
    }
    for (int j = 0; j < n; ++j) output[i * n + j] /= sum;
  }
}

void CpuTopk(const float* scores, float* topk_scores, int* topk_indices, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return scores[i * n + a] > scores[i * n + b]; });
    float sum = 0.0f;
    for (int j = 0; j < k; ++j) {
      topk_scores[i * k + j] = scores[i * n + indices[j]];
      sum += topk_scores[i * k + j];
    }
    for (int j = 0; j < k; ++j) {
      topk_scores[i * k + j] /= sum;
      topk_indices[i * k + j] = indices[j];
    }
  }
}

// C[m,n] = A[m,k] * B^T[n,k] (B stored row-major as [n, k])
void CpuMatmulTransB(const float* A, const float* B, float* C, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < k; ++l) {
        sum += A[i * k + l] * B[j * k + l];
      }
      C[i * n + j] = sum;
    }
  }
}

float CpuSilu(float x) { return x / (1.0f + std::exp(-x)); }
float CpuSigmoidScalar(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// input [m, 2*n] -> output [m, n]: SiLU(first_half) * second_half
void CpuSwiglu(const float* input, float* output, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float gate = input[i * 2 * n + j];
      float up = input[i * 2 * n + n + j];
      output[i * n + j] = CpuSilu(gate) * up;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test Fixture
// ═══════════════════════════════════════════════════════════════════════════════

class KunlunMoeMlpTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(xpu_set_device(0), 0);
    ASSERT_EQ(xpu_stream_create(&stream_), 0);
    xctx_ = std::make_unique<XdnnContext>(stream_);
  }

  void TearDown() override {
    xctx_.reset();
    xpu_stream_destroy(stream_);
  }

  xdnn::Context* ctx() { return xctx_->ctx; }
  // Persistent xpukernel::Context* — process-level singleton from xmlir_rt,
  // bound to our test stream. Same path production layer code uses via
  // KunlunDeviceContextExtension::GetXpuKernelContext.
  xpytorch::xpu::api::Context* xpuctx() {
    auto* c = xmlir_rt::getXpuKernelContext(/*useCpu=*/false);
    c->set_stream(xctx_->ctx->get_stream(), false);
    return c;
  }
  XPUStream stream() { return stream_; }

  // Small-scale test parameters
  static constexpr int kNumTokens = 4;
  static constexpr int kHiddenSize = 64;
  static constexpr int kInterSize = 128;
  static constexpr int kNumExperts = 4;
  static constexpr int kTopk = 2;

  // block_num = ncluster (number of XPU SDNN clusters), obtained from context
  int getBlockNum() { return ctx()->ncluster(); }

 private:
  XPUStream stream_;
  std::unique_ptr<XdnnContext> xctx_;
};

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Shared Expert Path
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeMlpTest, SharedExpertTest) {
  printf("\n=== SharedExpertTest ===\n");
  const int M = kNumTokens;
  const int H = kHiddenSize;
  const int I = kInterSize;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

  std::vector<float> h_input(M * H);
  std::vector<float> h_gate_up_weight(2 * I * H);
  std::vector<float> h_down_weight(H * I);
  std::vector<float> h_gate_weight(1 * H);

  for (auto& v : h_input) v = dist(rng);
  for (auto& v : h_gate_up_weight) v = dist(rng);
  for (auto& v : h_down_weight) v = dist(rng);
  for (auto& v : h_gate_weight) v = dist(rng);

  // CPU reference
  std::vector<float> cpu_gate_up(M * 2 * I);
  std::vector<float> cpu_act(M * I);
  std::vector<float> cpu_down(M * H);
  std::vector<float> cpu_output(M * H);

  CpuMatmulTransB(h_input.data(), h_gate_up_weight.data(), cpu_gate_up.data(), M, 2 * I, H);
  CpuSwiglu(cpu_gate_up.data(), cpu_act.data(), M, I);
  CpuMatmulTransB(cpu_act.data(), h_down_weight.data(), cpu_down.data(), M, H, I);

  for (int i = 0; i < M; ++i) {
    float dot = 0.0f;
    for (int j = 0; j < H; ++j) dot += h_input[i * H + j] * h_gate_weight[j];
    float gate_val = CpuSigmoidScalar(dot);
    for (int j = 0; j < H; ++j) {
      cpu_output[i * H + j] = gate_val * cpu_down[i * H + j];
    }
  }

  // XPU execution
  float* d_input = AllocAndCopy(h_input);
  float* d_gate_up_weight = AllocAndCopy(h_gate_up_weight);
  float* d_down_weight = AllocAndCopy(h_down_weight);
  float* d_gate_weight = AllocAndCopy(h_gate_weight);
  float* d_gate_up = AllocDevice<float>(M * 2 * I);
  float* d_act = AllocDevice<float>(M * I);
  float* d_down = AllocDevice<float>(M * H);
  float* d_gate_logit = AllocDevice<float>(M);
  float* d_gate_val = AllocDevice<float>(M);
  float* d_output = AllocDevice<float>(M * H);

  int ret;
  // gate_up_proj
  ret = XpuFc<float, float, float, int16_t>(xpuctx(), d_input, d_gate_up_weight, d_gate_up, M, 2 * I, H, false, true,
                                            nullptr, nullptr, nullptr);
  ASSERT_EQ(ret, 0) << "gate_up_proj failed";

  // SwiGLU
  ret = XpuRowBasedSiluMul<float>(ctx(), d_gate_up, d_act, M, I);
  ASSERT_EQ(ret, 0) << "SwiGLU failed";

  // down_proj
  ret = XpuFc<float, float, float, int16_t>(xpuctx(), d_act, d_down_weight, d_down, M, H, I, false, true, nullptr,
                                            nullptr, nullptr);
  ASSERT_EQ(ret, 0) << "down_proj failed";

  // gate = sigmoid(fc(input, gate_weight))
  ret = XpuFc<float, float, float, int16_t>(xpuctx(), d_input, d_gate_weight, d_gate_logit, M, 1, H, false, true,
                                            nullptr, nullptr, nullptr);
  ASSERT_EQ(ret, 0) << "gate_proj failed";

  ret = XpuSigmoid<float>(ctx(), d_gate_logit, d_gate_val, M);
  ASSERT_EQ(ret, 0) << "sigmoid failed";

  // output = gate_val * down (broadcast)
  std::vector<int64_t> x_shape = {M, H};
  std::vector<int64_t> y_shape = {M, 1};
  ret = xdnn::broadcast_mul<float>(ctx(), d_down, d_gate_val, d_output, x_shape, y_shape);
  ASSERT_EQ(ret, 0) << "broadcast_mul failed";

  ASSERT_EQ(xpu_wait(stream()), 0);

  // Verify
  std::vector<float> h_output(M * H);
  CopyToHost(h_output, d_output);

  float max_diff = 0.0f;
  for (int i = 0; i < M * H; ++i) {
    float diff = std::fabs(h_output[i] - cpu_output[i]);
    if (diff > max_diff) max_diff = diff;
  }
  printf("SharedExpertTest: max_diff = %e\n", max_diff);
  EXPECT_LT(max_diff, 1e-2) << "Shared expert output mismatch";

  xpu_free(d_input);
  xpu_free(d_gate_up_weight);
  xpu_free(d_down_weight);
  xpu_free(d_gate_weight);
  xpu_free(d_gate_up);
  xpu_free(d_act);
  xpu_free(d_down);
  xpu_free(d_gate_logit);
  xpu_free(d_gate_val);
  xpu_free(d_output);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Router Expert Path
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeMlpTest, RouterExpertTest) {
  printf("\n=== RouterExpertTest ===\n");
  const int M = kNumTokens;
  const int H = kHiddenSize;
  const int I = kInterSize;
  const int E = kNumExperts;
  const int K = kTopk;
  const int B = getBlockNum();

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

  std::vector<float> h_input(M * H);
  std::vector<float> h_router_weight(E * H);
  std::vector<float> h_w1(E * 2 * I * H);
  std::vector<float> h_w2(E * H * I);

  for (auto& v : h_input) v = dist(rng);
  for (auto& v : h_router_weight) v = dist(rng);
  for (auto& v : h_w1) v = dist(rng);
  for (auto& v : h_w2) v = dist(rng);

  // CPU reference
  std::vector<float> cpu_router_logits(M * E);
  CpuMatmulTransB(h_input.data(), h_router_weight.data(), cpu_router_logits.data(), M, E, H);
  std::vector<float> cpu_softmax(M * E);
  CpuSoftmax(cpu_router_logits.data(), cpu_softmax.data(), M, E);
  std::vector<float> cpu_topk_scores(M * K);
  std::vector<int> cpu_topk_indices(M * K);
  CpuTopk(cpu_softmax.data(), cpu_topk_scores.data(), cpu_topk_indices.data(), M, E, K);

  std::vector<float> cpu_router_output(M * H, 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      int expert_id = cpu_topk_indices[i * K + j];
      float weight = cpu_topk_scores[i * K + j];
      std::vector<float> gate_up(2 * I);
      CpuMatmulTransB(h_input.data() + i * H, h_w1.data() + expert_id * 2 * I * H, gate_up.data(), 1, 2 * I, H);
      std::vector<float> act(I);
      CpuSwiglu(gate_up.data(), act.data(), 1, I);
      std::vector<float> down(H);
      CpuMatmulTransB(act.data(), h_w2.data() + expert_id * H * I, down.data(), 1, H, I);
      for (int k_idx = 0; k_idx < H; ++k_idx) {
        cpu_router_output[i * H + k_idx] += weight * down[k_idx];
      }
    }
  }

  // XPU execution
  int ret;
  float* d_input = AllocAndCopy(h_input);
  float* d_router_weight = AllocAndCopy(h_router_weight);
  float* d_w1 = AllocAndCopy(h_w1);
  float* d_w2 = AllocAndCopy(h_w2);
  float* d_router_logits = AllocDevice<float>(M * E);
  float* d_topk_scores = AllocDevice<float>(M * K);
  int* d_topk_indices = AllocDevice<int>(M * K);
  int32_t* d_block_statistic = AllocDevice<int32_t>(B * E);
  float* d_sorted_x = AllocDevice<float>(M * K * H);
  int* d_moe_index = AllocDevice<int>(M * K);
  int* d_expert_m = AllocDevice<int>(E);
  int* d_sorted_tokens_num_lod = AllocDevice<int>(E + 1);
  float* d_fc1_out = AllocDevice<float>(M * K * 2 * I);
  float* d_act_out = AllocDevice<float>(M * K * I);
  float* d_fc2_out = AllocDevice<float>(M * K * H);
  float* d_router_output = AllocDevice<float>(M * H);

  // Step 1: Router gate
  ret = XpuFc<float, float, float, int16_t>(xpuctx(), d_input, d_router_weight, d_router_logits, M, E, H, false, true,
                                            nullptr, nullptr, nullptr);
  ASSERT_EQ(ret, 0) << "router gate failed";

  // Step 2: softmax + topk + normalize
  ret = infer_ops::moe_softmax_topk_norm_fusion<float, float, int>(
      ctx(), d_router_logits, d_topk_scores, d_topk_indices, d_block_statistic, M, E, K, B, nullptr, true);
  ASSERT_EQ(ret, 0) << "moe_softmax_topk_norm_fusion failed";

  // Step 3: Pre-sorted
  ret = infer_ops::moe_ffn_pre_sorted<float, int>(ctx(), d_input, d_topk_indices, d_block_statistic, d_sorted_x,
                                                  d_moe_index, d_expert_m, d_sorted_tokens_num_lod, M, H, E, K, B);
  ASSERT_EQ(ret, 0) << "moe_ffn_pre_sorted failed";

  // Step 4: MoE FC1 (gate_up) — per-expert xdnn::fc loop (Plan C fallback for fp32)
  // sorted_x is arranged by expert (from moe_ffn_pre_sorted with SortedByExpert mode)
  // We need sorted_tokens_num_lod on host to compute offsets
  {
    std::vector<int> h_lod(E + 1);
    CopyToHost(h_lod, d_sorted_tokens_num_lod);
    ASSERT_EQ(xpu_wait(stream()), 0);

    for (int e = 0; e < E; ++e) {
      int start = h_lod[e];
      int expert_tokens = h_lod[e + 1] - h_lod[e];
      if (expert_tokens == 0) continue;
      float* expert_in = d_sorted_x + start * H;
      float* expert_out = d_fc1_out + start * (2 * I);
      const float* expert_w = d_w1 + e * (2 * I * H);
      ret = XpuFc<float, float, float, int16_t>(xpuctx(), expert_in, expert_w, expert_out, expert_tokens, 2 * I, H,
                                                false, true, nullptr, nullptr, nullptr);
      ASSERT_EQ(ret, 0) << "fc1 expert " << e << " failed";
    }
  }

  // Step 5: SwiGLU
  {
    std::vector<int64_t> xshape = {static_cast<int64_t>(M * K), static_cast<int64_t>(2 * I)};
    ret = infer_ops::swiglu<float>(ctx(), d_fc1_out, (const float*)nullptr, d_act_out, xshape, 1, true);
    ASSERT_EQ(ret, 0) << "swiglu failed";
  }

  // Step 6: MoE FC2 (down) — per-expert xdnn::fc loop
  {
    std::vector<int> h_lod(E + 1);
    CopyToHost(h_lod, d_sorted_tokens_num_lod);
    ASSERT_EQ(xpu_wait(stream()), 0);

    for (int e = 0; e < E; ++e) {
      int start = h_lod[e];
      int expert_tokens = h_lod[e + 1] - h_lod[e];
      if (expert_tokens == 0) continue;
      float* expert_in = d_act_out + start * I;
      float* expert_out = d_fc2_out + start * H;
      const float* expert_w = d_w2 + e * (H * I);
      ret = XpuFc<float, float, float, int16_t>(xpuctx(), expert_in, expert_w, expert_out, expert_tokens, H, I, false,
                                                true, nullptr, nullptr, nullptr);
      ASSERT_EQ(ret, 0) << "fc2 expert " << e << " failed";
    }
  }

  // Step 7: Post fusion
  ret = infer_ops::moe_ffn_post_fusion<float, int, float>(ctx(), d_fc2_out, d_moe_index, d_topk_scores,
                                                          (const float*)nullptr, d_router_output, M, H, K);
  ASSERT_EQ(ret, 0) << "moe_ffn_post_fusion failed";

  ASSERT_EQ(xpu_wait(stream()), 0);

  // Verify
  std::vector<float> h_router_output(M * H);
  CopyToHost(h_router_output, d_router_output);

  float max_diff = 0.0f;
  for (int i = 0; i < M * H; ++i) {
    float diff = std::fabs(h_router_output[i] - cpu_router_output[i]);
    if (diff > max_diff) max_diff = diff;
  }
  printf("RouterExpertTest: max_abs_diff = %e\n", max_diff);
  EXPECT_LT(max_diff, 5e-2) << "Router expert output mismatch";

  xpu_free(d_input);
  xpu_free(d_router_weight);
  xpu_free(d_w1);
  xpu_free(d_w2);
  xpu_free(d_router_logits);
  xpu_free(d_topk_scores);
  xpu_free(d_topk_indices);
  xpu_free(d_block_statistic);
  xpu_free(d_sorted_x);
  xpu_free(d_moe_index);
  xpu_free(d_expert_m);
  xpu_free(d_sorted_tokens_num_lod);
  xpu_free(d_fc1_out);
  xpu_free(d_act_out);
  xpu_free(d_fc2_out);
  xpu_free(d_router_output);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Full MoE MLP (Router + Shared combined)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunMoeMlpTest, FullMoeMlpTest) {
  printf("\n=== FullMoeMlpTest ===\n");
  const int M = kNumTokens;
  const int H = kHiddenSize;
  const int I = kInterSize;
  const int E = kNumExperts;
  const int K = kTopk;
  const int B = getBlockNum();

  std::mt19937 rng(456);
  std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

  // Weights
  std::vector<float> h_input(M * H);
  std::vector<float> h_router_weight(E * H);
  std::vector<float> h_w1(E * 2 * I * H);
  std::vector<float> h_w2(E * H * I);
  std::vector<float> h_shared_gate_up_weight(2 * I * H);
  std::vector<float> h_shared_down_weight(H * I);
  std::vector<float> h_shared_gate_weight(1 * H);

  for (auto& v : h_input) v = dist(rng);
  for (auto& v : h_router_weight) v = dist(rng);
  for (auto& v : h_w1) v = dist(rng);
  for (auto& v : h_w2) v = dist(rng);
  for (auto& v : h_shared_gate_up_weight) v = dist(rng);
  for (auto& v : h_shared_down_weight) v = dist(rng);
  for (auto& v : h_shared_gate_weight) v = dist(rng);

  // --- CPU reference: Router ---
  std::vector<float> cpu_router_logits(M * E);
  CpuMatmulTransB(h_input.data(), h_router_weight.data(), cpu_router_logits.data(), M, E, H);
  std::vector<float> cpu_softmax(M * E);
  CpuSoftmax(cpu_router_logits.data(), cpu_softmax.data(), M, E);
  std::vector<float> cpu_topk_scores(M * K);
  std::vector<int> cpu_topk_indices(M * K);
  CpuTopk(cpu_softmax.data(), cpu_topk_scores.data(), cpu_topk_indices.data(), M, E, K);

  std::vector<float> cpu_router_output(M * H, 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      int expert_id = cpu_topk_indices[i * K + j];
      float w = cpu_topk_scores[i * K + j];
      std::vector<float> gu(2 * I), act(I), dn(H);
      CpuMatmulTransB(h_input.data() + i * H, h_w1.data() + expert_id * 2 * I * H, gu.data(), 1, 2 * I, H);
      CpuSwiglu(gu.data(), act.data(), 1, I);
      CpuMatmulTransB(act.data(), h_w2.data() + expert_id * H * I, dn.data(), 1, H, I);
      for (int k2 = 0; k2 < H; ++k2) cpu_router_output[i * H + k2] += w * dn[k2];
    }
  }

  // --- CPU reference: Shared ---
  std::vector<float> cpu_sgu(M * 2 * I), cpu_sact(M * I), cpu_sdown(M * H), cpu_sout(M * H);
  CpuMatmulTransB(h_input.data(), h_shared_gate_up_weight.data(), cpu_sgu.data(), M, 2 * I, H);
  CpuSwiglu(cpu_sgu.data(), cpu_sact.data(), M, I);
  CpuMatmulTransB(cpu_sact.data(), h_shared_down_weight.data(), cpu_sdown.data(), M, H, I);
  for (int i = 0; i < M; ++i) {
    float dot = 0.0f;
    for (int j = 0; j < H; ++j) dot += h_input[i * H + j] * h_shared_gate_weight[j];
    float gv = CpuSigmoidScalar(dot);
    for (int j = 0; j < H; ++j) cpu_sout[i * H + j] = gv * cpu_sdown[i * H + j];
  }

  // Final = router + shared
  std::vector<float> cpu_final(M * H);
  for (int i = 0; i < M * H; ++i) cpu_final[i] = cpu_router_output[i] + cpu_sout[i];

  // --- XPU ---
  int ret;
  float* d_input = AllocAndCopy(h_input);
  float* d_router_weight = AllocAndCopy(h_router_weight);
  float* d_w1 = AllocAndCopy(h_w1);
  float* d_w2 = AllocAndCopy(h_w2);
  float* d_shared_guw = AllocAndCopy(h_shared_gate_up_weight);
  float* d_shared_dw = AllocAndCopy(h_shared_down_weight);
  float* d_shared_gw = AllocAndCopy(h_shared_gate_weight);

  // Router buffers
  float* d_rl = AllocDevice<float>(M * E);
  float* d_ts = AllocDevice<float>(M * K);
  int* d_ti = AllocDevice<int>(M * K);
  int32_t* d_bs = AllocDevice<int32_t>(B * E);
  float* d_sx = AllocDevice<float>(M * K * H);
  int* d_mi = AllocDevice<int>(M * K);
  int* d_em = AllocDevice<int>(E);
  int* d_lod = AllocDevice<int>(E + 1);
  float* d_f1 = AllocDevice<float>(M * K * 2 * I);
  float* d_ao = AllocDevice<float>(M * K * I);
  float* d_f2 = AllocDevice<float>(M * K * H);
  float* d_ro = AllocDevice<float>(M * H);

  // Shared buffers
  float* d_sgu = AllocDevice<float>(M * 2 * I);
  float* d_sa = AllocDevice<float>(M * I);
  float* d_sd = AllocDevice<float>(M * H);
  float* d_sgl = AllocDevice<float>(M);
  float* d_sgv = AllocDevice<float>(M);
  float* d_so = AllocDevice<float>(M * H);
  float* d_final = AllocDevice<float>(M * H);

  // Router path
  ret = XpuFc<float, float, float, int16_t>(xpuctx(), d_input, d_router_weight, d_rl, M, E, H, false, true, nullptr,
                                            nullptr, nullptr);
  ASSERT_EQ(ret, 0);
  ret = infer_ops::moe_softmax_topk_norm_fusion<float, float, int>(ctx(), d_rl, d_ts, d_ti, d_bs, M, E, K, B, nullptr,
                                                                   true);
  ASSERT_EQ(ret, 0);
  ret = infer_ops::moe_ffn_pre_sorted<float, int>(ctx(), d_input, d_ti, d_bs, d_sx, d_mi, d_em, d_lod, M, H, E, K, B);
  ASSERT_EQ(ret, 0);

  // FC1 — per-expert loop
  {
    std::vector<int> h_lod(E + 1);
    CopyToHost(h_lod, d_lod);
    ASSERT_EQ(xpu_wait(stream()), 0);
    for (int e = 0; e < E; ++e) {
      int start = h_lod[e];
      int et = h_lod[e + 1] - h_lod[e];
      if (et == 0) continue;
      ret = XpuFc<float, float, float, int16_t>(xpuctx(), d_sx + start * H, d_w1 + e * 2 * I * H, d_f1 + start * 2 * I,
                                                et, 2 * I, H, false, true, nullptr, nullptr, nullptr);
      ASSERT_EQ(ret, 0);
    }
  }

  // SwiGLU
  {
    std::vector<int64_t> xshape = {static_cast<int64_t>(M * K), static_cast<int64_t>(2 * I)};
    ret = infer_ops::swiglu<float>(ctx(), d_f1, (const float*)nullptr, d_ao, xshape, 1, true);
    ASSERT_EQ(ret, 0);
  }

  // FC2 — per-expert loop
  {
    std::vector<int> h_lod(E + 1);
    CopyToHost(h_lod, d_lod);
    ASSERT_EQ(xpu_wait(stream()), 0);
    for (int e = 0; e < E; ++e) {
      int start = h_lod[e];
      int et = h_lod[e + 1] - h_lod[e];
      if (et == 0) continue;
      ret = XpuFc<float, float, float, int16_t>(xpuctx(), d_ao + start * I, d_w2 + e * H * I, d_f2 + start * H, et, H,
                                                I, false, true, nullptr, nullptr, nullptr);
      ASSERT_EQ(ret, 0);
    }
  }

  // Post fusion
  ret =
      infer_ops::moe_ffn_post_fusion<float, int, float>(ctx(), d_f2, d_mi, d_ts, (const float*)nullptr, d_ro, M, H, K);
  ASSERT_EQ(ret, 0);

  // Shared path
  ret = XpuFc<float, float, float, int16_t>(xpuctx(), d_input, d_shared_guw, d_sgu, M, 2 * I, H, false, true, nullptr,
                                            nullptr, nullptr);
  ASSERT_EQ(ret, 0);
  ret = XpuRowBasedSiluMul<float>(ctx(), d_sgu, d_sa, M, I);
  ASSERT_EQ(ret, 0);
  ret = XpuFc<float, float, float, int16_t>(xpuctx(), d_sa, d_shared_dw, d_sd, M, H, I, false, true, nullptr, nullptr,
                                            nullptr);
  ASSERT_EQ(ret, 0);
  ret = XpuFc<float, float, float, int16_t>(xpuctx(), d_input, d_shared_gw, d_sgl, M, 1, H, false, true, nullptr,
                                            nullptr, nullptr);
  ASSERT_EQ(ret, 0);
  ret = XpuSigmoid<float>(ctx(), d_sgl, d_sgv, M);
  ASSERT_EQ(ret, 0);
  {
    std::vector<int64_t> xs = {M, H};
    std::vector<int64_t> ys = {M, 1};
    ret = xdnn::broadcast_mul<float>(ctx(), d_sd, d_sgv, d_so, xs, ys);
    ASSERT_EQ(ret, 0);
  }

  // Combine
  ret = XpuAdd<float>(xpuctx(), d_ro, d_so, d_final, M * H);
  ASSERT_EQ(ret, 0);

  ASSERT_EQ(xpu_wait(stream()), 0);

  // Verify
  std::vector<float> h_final(M * H);
  CopyToHost(h_final, d_final);
  float max_diff = 0.0f;
  for (int i = 0; i < M * H; ++i) {
    float diff = std::fabs(h_final[i] - cpu_final[i]);
    if (diff > max_diff) max_diff = diff;
  }
  printf("FullMoeMlpTest: max_diff = %e\n", max_diff);
  EXPECT_LT(max_diff, 5e-2) << "Full MoE MLP output mismatch";

  // Cleanup
  xpu_free(d_input);
  xpu_free(d_router_weight);
  xpu_free(d_w1);
  xpu_free(d_w2);
  xpu_free(d_shared_guw);
  xpu_free(d_shared_dw);
  xpu_free(d_shared_gw);
  xpu_free(d_rl);
  xpu_free(d_ts);
  xpu_free(d_ti);
  xpu_free(d_bs);
  xpu_free(d_sx);
  xpu_free(d_mi);
  xpu_free(d_em);
  xpu_free(d_lod);
  xpu_free(d_f1);
  xpu_free(d_ao);
  xpu_free(d_f2);
  xpu_free(d_ro);
  xpu_free(d_sgu);
  xpu_free(d_sa);
  xpu_free(d_sd);
  xpu_free(d_sgl);
  xpu_free(d_sgv);
  xpu_free(d_so);
  xpu_free(d_final);
}

}  // namespace ksana_llm

// ═══════════════════════════════════════════════════════════════════════════════
// Half-precision MoE MLP Tests (bfloat16 / float16)
// Uses xblas::moe_fc_fusion via the kernel wrapper (XpuMoeFcFusion).
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef ENABLE_KUNLUN

namespace ksana_llm {

// ── Half-precision test fixture ─────────────────────────────────────────────

class MoeMlpHalfTest : public testing::Test {
 protected:
  static constexpr int M = 4;    // num_tokens
  static constexpr int H = 64;   // hidden_size
  static constexpr int I = 128;  // ffn_hidden_size
  static constexpr int E = 4;    // num_experts
  static constexpr int K = 2;    // topk

  XPUStream stream_ = nullptr;
  xdnn::Context* ctx_ = nullptr;

  // CPU reference data (float32)
  std::vector<float> h_input;          // [M, H]
  std::vector<float> h_router_logits;  // [M, E]
  std::vector<float> h_w1;             // [E, 2*I, H] (gate_up weights)
  std::vector<float> h_w2;             // [E, H, I] (down weights)

  // CPU reference results
  std::vector<float> cpu_output;  // [M, H]

  void SetUp() override {
    ASSERT_EQ(xpu_stream_create(&stream_), 0);
    ctx_ = xdnn::create_context();
    ctx_->set_stream(stream_);

    // Initialize random data
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    h_input.resize(M * H);
    h_router_logits.resize(M * E);
    h_w1.resize(E * 2 * I * H);
    h_w2.resize(E * H * I);

    for (auto& v : h_input) v = dist(rng);
    for (auto& v : h_router_logits) v = dist(rng);
    for (auto& v : h_w1) v = dist(rng);
    for (auto& v : h_w2) v = dist(rng);

    // Compute CPU reference
    ComputeCpuReference();
  }

  void TearDown() override {
    if (ctx_) xdnn::destroy_context(ctx_);
    if (stream_) xpu_stream_destroy(stream_);
  }

  xdnn::Context* ctx() { return ctx_; }
  // Persistent xpukernel::Context* — process-level singleton from xmlir_rt,
  // bound to our test stream. Same path production layer code uses via
  // KunlunDeviceContextExtension::GetXpuKernelContext.
  xpytorch::xpu::api::Context* xpuctx() {
    auto* c = xmlir_rt::getXpuKernelContext(/*useCpu=*/false);
    c->set_stream(stream_, false);
    return c;
  }
  XPUStream stream() { return stream_; }

  void ComputeCpuReference() {
    // Step 1: Softmax on router logits
    std::vector<float> softmax_out(M * E);
    for (int i = 0; i < M; ++i) {
      float max_val = *std::max_element(&h_router_logits[i * E], &h_router_logits[(i + 1) * E]);
      float sum = 0.0f;
      for (int j = 0; j < E; ++j) {
        softmax_out[i * E + j] = std::exp(h_router_logits[i * E + j] - max_val);
        sum += softmax_out[i * E + j];
      }
      for (int j = 0; j < E; ++j) softmax_out[i * E + j] /= sum;
    }

    // Step 2: TopK selection + normalize
    std::vector<float> topk_scores(M * K);
    std::vector<int> topk_indices(M * K);
    for (int i = 0; i < M; ++i) {
      std::vector<int> idx(E);
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                        [&](int a, int b) { return softmax_out[i * E + a] > softmax_out[i * E + b]; });
      float sum = 0.0f;
      for (int j = 0; j < K; ++j) {
        topk_scores[i * K + j] = softmax_out[i * E + idx[j]];
        sum += topk_scores[i * K + j];
      }
      for (int j = 0; j < K; ++j) {
        topk_scores[i * K + j] /= sum;
        topk_indices[i * K + j] = idx[j];
      }
    }

    // Step 3-7: Per-token expert computation + weighted sum
    cpu_output.assign(M * H, 0.0f);
    for (int i = 0; i < M; ++i) {
      for (int t = 0; t < K; ++t) {
        int expert_id = topk_indices[i * K + t];
        float score = topk_scores[i * K + t];

        // FC1: input[i] @ w1[expert_id].T → [2*I]
        std::vector<float> fc1_out(2 * I, 0.0f);
        for (int j = 0; j < 2 * I; ++j) {
          for (int kk = 0; kk < H; ++kk) {
            fc1_out[j] += h_input[i * H + kk] * h_w1[expert_id * 2 * I * H + j * H + kk];
          }
        }

        // SwiGLU: silu(fc1[:I]) * fc1[I:]
        std::vector<float> act_out(I);
        for (int j = 0; j < I; ++j) {
          float gate = fc1_out[j];
          float up = fc1_out[I + j];
          float silu_gate = gate / (1.0f + std::exp(-gate));
          act_out[j] = silu_gate * up;
        }

        // FC2: act_out @ w2[expert_id].T → [H]
        std::vector<float> fc2_out(H, 0.0f);
        for (int j = 0; j < H; ++j) {
          for (int kk = 0; kk < I; ++kk) {
            fc2_out[j] += act_out[kk] * h_w2[expert_id * H * I + j * I + kk];
          }
        }

        // Weighted sum
        for (int j = 0; j < H; ++j) {
          cpu_output[i * H + j] += score * fc2_out[j];
        }
      }
    }
  }

  // Run the full MoE pipeline using XPU wrapper functions with a given half type.
  // T = float16 or bfloat16 (KsanaLLM typedef)
  template <typename T>
  void RunHalfPrecisionMoePipeline(float tolerance) {
    int ret;

    // ─── Allocate fp32 buffers on device for routing ───
    float* d_router_logits = AllocAndCopy(h_router_logits);

    // ─── Step 1: moe_softmax_topk_norm (routing in half, scores in float) ───
    // Cast router logits to half type for softmax_topk input
    T* d_router_half = AllocDevice<T>(M * E);
    ret = XpuCast<float, T>(xpuctx(), d_router_logits, d_router_half, M * E);
    ASSERT_EQ(ret, 0) << "Cast router logits to half failed";

    float* d_topk_scores = AllocDevice<float>(M * K);
    int* d_topk_indices = AllocDevice<int>(M * K);
    int32_t* d_block_stat = AllocDevice<int32_t>(ctx()->ncluster() * E);

    ret = XpuMoeSoftmaxTopkNorm<T, float, int>(ctx(), d_router_half, d_topk_scores, d_topk_indices, d_block_stat, M, E,
                                               K, ctx()->ncluster(), /*stable=*/true);
    ASSERT_EQ(ret, 0) << "XpuMoeSoftmaxTopkNorm failed";

    // ─── Step 2: Cast input and weights to half ───
    T* d_input_half = AllocDevice<T>(M * H);
    T* d_w1_half = AllocDevice<T>(E * 2 * I * H);
    T* d_w2_half = AllocDevice<T>(E * H * I);

    {
      float* d_input_fp32 = AllocAndCopy(h_input);
      float* d_w1_fp32 = AllocAndCopy(h_w1);
      float* d_w2_fp32 = AllocAndCopy(h_w2);
      ret = XpuCast<float, T>(xpuctx(), d_input_fp32, d_input_half, M * H);
      ASSERT_EQ(ret, 0);
      ret = XpuCast<float, T>(xpuctx(), d_w1_fp32, d_w1_half, E * 2 * I * H);
      ASSERT_EQ(ret, 0);
      ret = XpuCast<float, T>(xpuctx(), d_w2_fp32, d_w2_half, E * H * I);
      ASSERT_EQ(ret, 0);
      xpu_free(d_input_fp32);
      xpu_free(d_w1_fp32);
      xpu_free(d_w2_fp32);
    }

    // ─── Step 3: moe_ffn_pre_sorted ───
    T* d_sorted_x = AllocDevice<T>(M * K * H);
    int* d_moe_index = AllocDevice<int>(M * K);
    int* d_expert_m = AllocDevice<int>(E);
    int* d_sorted_lod = AllocDevice<int>(E + 1);

    ret = XpuMoeFfnPreSorted<T, int>(ctx(), d_input_half, d_topk_indices, d_block_stat, d_sorted_x, d_moe_index,
                                     d_expert_m, d_sorted_lod, M, H, E, K, ctx()->ncluster());
    ASSERT_EQ(ret, 0) << "XpuMoeFfnPreSorted failed";

    // ─── Step 4: XpuMoeFcFusion FC1 (gate_up projection) ───
    T* d_fc1_out = AllocDevice<T>(M * K * 2 * I);

    ret = XpuMoeFcFusion<T, int>(ctx(), d_sorted_x, d_w1_half, d_fc1_out, M, K, M * K, 2 * I, H, E, d_moe_index,
                                 d_sorted_lod);
    ASSERT_EQ(ret, 0) << "XpuMoeFcFusion FC1 failed";

    // ─── Step 5: XpuSwiglu ───
    T* d_act_out = AllocDevice<T>(M * K * I);

    ret = XpuSwiglu<T>(ctx(), d_fc1_out, d_act_out, M * K, I);
    ASSERT_EQ(ret, 0) << "XpuSwiglu failed";

    // ─── Step 6: XpuMoeFcFusion FC2 (down projection) ───
    T* d_fc2_out = AllocDevice<T>(M * K * H);

    ret =
        XpuMoeFcFusion<T, int>(ctx(), d_act_out, d_w2_half, d_fc2_out, M, K, M * K, H, I, E, d_moe_index, d_sorted_lod);
    ASSERT_EQ(ret, 0) << "XpuMoeFcFusion FC2 failed";

    // ─── Step 7: XpuMoeFfnPostFusion (weighted scatter-add) ───
    // Cast topk_scores to T for post_fusion
    T* d_topk_scores_half = AllocDevice<T>(M * K);
    ret = XpuCast<float, T>(xpuctx(), d_topk_scores, d_topk_scores_half, M * K);
    ASSERT_EQ(ret, 0);

    T* d_output = AllocDevice<T>(M * H);

    ret = XpuMoeFfnPostFusion<T, int>(ctx(), d_fc2_out, d_moe_index, d_topk_scores_half, d_output, M, H, K);
    ASSERT_EQ(ret, 0) << "XpuMoeFfnPostFusion failed";

    // ─── Verify against CPU reference ───
    ASSERT_EQ(xpu_wait(stream()), 0);

    // Cast output back to float for comparison
    float* d_output_fp32 = AllocDevice<float>(M * H);
    ret = XpuCast<T, float>(xpuctx(), d_output, d_output_fp32, M * H);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(xpu_wait(stream()), 0);

    std::vector<float> h_output(M * H);
    CopyToHost(h_output, d_output_fp32);

    float max_diff = 0.0f;
    float max_ref = 0.0f;
    for (int i = 0; i < M * H; ++i) {
      float diff = std::fabs(h_output[i] - cpu_output[i]);
      if (diff > max_diff) max_diff = diff;
      if (std::fabs(cpu_output[i]) > max_ref) max_ref = std::fabs(cpu_output[i]);
    }
    float rel_err = (max_ref > 1e-6f) ? (max_diff / max_ref) : max_diff;
    printf("  max_abs_diff = %e, max_ref = %e, rel_err = %e\n", max_diff, max_ref, rel_err);
    EXPECT_LT(rel_err, tolerance) << "Half-precision MoE pipeline output mismatch";

    // Cleanup
    xpu_free(d_router_logits);
    xpu_free(d_router_half);
    xpu_free(d_topk_scores);
    xpu_free(d_topk_indices);
    xpu_free(d_block_stat);
    xpu_free(d_input_half);
    xpu_free(d_w1_half);
    xpu_free(d_w2_half);
    xpu_free(d_sorted_x);
    xpu_free(d_moe_index);
    xpu_free(d_expert_m);
    xpu_free(d_sorted_lod);
    xpu_free(d_fc1_out);
    xpu_free(d_act_out);
    xpu_free(d_fc2_out);
    xpu_free(d_topk_scores_half);
    xpu_free(d_output);
    xpu_free(d_output_fp32);
  }
};

// ── bfloat16 test ───────────────────────────────────────────────────────────

TEST_F(MoeMlpHalfTest, RouterExpertPathBf16) {
  printf("=== MoE MLP bfloat16 Test (XpuMoeFcFusion via xblas) ===\n");
  RunHalfPrecisionMoePipeline<bfloat16>(0.15f);  // bfloat16 has ~7-bit mantissa
}

// ── float16 test ────────────────────────────────────────────────────────────

TEST_F(MoeMlpHalfTest, RouterExpertPathFp16) {
  printf("=== MoE MLP float16 Test (XpuMoeFcFusion via xblas) ===\n");
  RunHalfPrecisionMoePipeline<float16>(0.10f);  // float16 has ~10-bit mantissa
}

}  // namespace ksana_llm

#endif  // ENABLE_KUNLUN
