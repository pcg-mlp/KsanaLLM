/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Cross-validate sglang topk_softplus_sqrt vs vllm backend on identical inputs.
 * Covers fp32 and bf16 router_logits at the actual DSv4 topk=6.
 */
#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <random>
#include <vector>

#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/sglang_gate_dtype.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/topk_softplus_sqrt_sglang.h"
#include "csrc/kernels/nvidia/others/vllm/main/topk_softplus_sqrt/topk_softplus_sqrt.h"

namespace {

using llm_kernels::nvidia::sglang_gate::GateInputDtype;

template <typename T>
T* DeviceAlloc(size_t n) {
  T* p = nullptr;
  cudaMalloc(&p, n * sizeof(T));
  return p;
}

template <typename T>
void H2D(T* d, const std::vector<T>& h) {
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
std::vector<T> D2H(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

void TestScorePath(int num_tokens, int num_experts, int topk, GateInputDtype dtype) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> logits(num_tokens * num_experts);
  std::vector<float> bias(num_experts);
  for (auto& x : logits) x = dist(rng);
  for (auto& x : bias) x = dist(rng) * 0.1f;

  // Allocate logits buffer in chosen dtype; both backends must see identical bits.
  void* d_logits = nullptr;
  llm_kernels::nvidia::tensorrt_llm::dev::ScalarType vllm_in_dtype;
  if (dtype == GateInputDtype::kFloat32) {
    auto* p = DeviceAlloc<float>(logits.size());
    H2D(p, logits);
    d_logits = p;
    vllm_in_dtype = llm_kernels::nvidia::tensorrt_llm::dev::ScalarType::Float;
  } else {
    std::vector<__nv_bfloat16> logits_bf16(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) logits_bf16[i] = __float2bfloat16(logits[i]);
    auto* p = DeviceAlloc<__nv_bfloat16>(logits_bf16.size());
    H2D(p, logits_bf16);
    d_logits = p;
    vllm_in_dtype = llm_kernels::nvidia::tensorrt_llm::dev::ScalarType::BFloat16;
  }

  float* d_bias = DeviceAlloc<float>(bias.size());
  H2D(d_bias, bias);

  float* d_w_vllm = DeviceAlloc<float>(num_tokens * topk);
  int32_t* d_i_vllm = DeviceAlloc<int32_t>(num_tokens * topk);
  int32_t* d_tei = DeviceAlloc<int32_t>(num_tokens * topk);
  float* d_w_sg = DeviceAlloc<float>(num_tokens * topk);
  int32_t* d_i_sg = DeviceAlloc<int32_t>(num_tokens * topk);

  llm_kernels::nvidia::InvokeTopkSoftplusSqrt(
      d_logits, d_w_vllm, d_i_vllm, d_tei, num_tokens, num_experts, topk, /*renormalize=*/true,
      /*scale=*/1.5, d_bias, /*input_ids=*/nullptr, /*tid2eid=*/nullptr, vllm_in_dtype,
      llm_kernels::nvidia::tensorrt_llm::dev::ScalarType::Int, /*stream=*/0);

  llm_kernels::nvidia::sglang_gate::InvokeTopkSoftplusSqrtSglang(
      d_logits, dtype, d_w_sg, d_i_sg, num_tokens, num_experts, topk, /*renormalize=*/true,
      /*scale=*/1.5f, d_bias, /*input_ids=*/nullptr, /*tid2eid=*/nullptr, /*stream=*/0);

  cudaDeviceSynchronize();

  auto w_vllm = D2H(d_w_vllm, num_tokens * topk);
  auto i_vllm = D2H(d_i_vllm, num_tokens * topk);
  auto w_sg = D2H(d_w_sg, num_tokens * topk);
  auto i_sg = D2H(d_i_sg, num_tokens * topk);

  for (int t = 0; t < num_tokens; ++t) {
    std::vector<int32_t> a(i_vllm.begin() + t * topk, i_vllm.begin() + (t + 1) * topk);
    std::vector<int32_t> b(i_sg.begin() + t * topk, i_sg.begin() + (t + 1) * topk);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    EXPECT_EQ(a, b) << "indices mismatch at token " << t;
  }
  // Sort per-token weight arrays before elementwise compare (selected indices may be
  // emitted in different orders). For bf16 inputs, both backends compute scores from
  // the same bf16 bits; small fp drift between vllm/sglang code paths still fits 5e-3.
  const float weight_tol = (dtype == GateInputDtype::kFloat32) ? 1e-3f : 5e-3f;
  for (int t = 0; t < num_tokens; ++t) {
    std::vector<float> a(w_vllm.begin() + t * topk, w_vllm.begin() + (t + 1) * topk);
    std::vector<float> b(w_sg.begin() + t * topk, w_sg.begin() + (t + 1) * topk);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    for (int k = 0; k < topk; ++k) {
      EXPECT_NEAR(a[k], b[k], weight_tol) << "weight mismatch at token " << t << " k=" << k;
    }
  }

  cudaFree(d_logits); cudaFree(d_bias);
  cudaFree(d_w_vllm); cudaFree(d_i_vllm); cudaFree(d_tei);
  cudaFree(d_w_sg);   cudaFree(d_i_sg);
}

void TestHashPath(int num_tokens, int num_experts, int topk, int vocab, GateInputDtype dtype) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::uniform_int_distribution<int> eid_dist(0, num_experts - 1);
  std::uniform_int_distribution<int32_t> tid_dist(0, vocab - 1);

  std::vector<float> logits(num_tokens * num_experts);
  std::vector<int32_t> input_ids(num_tokens);
  std::vector<int32_t> tid2eid(vocab * topk);
  for (auto& x : logits) x = dist(rng);
  for (auto& x : input_ids) x = tid_dist(rng);
  for (auto& x : tid2eid) x = eid_dist(rng);

  void* d_logits = nullptr;
  llm_kernels::nvidia::tensorrt_llm::dev::ScalarType vllm_in_dtype;
  if (dtype == GateInputDtype::kFloat32) {
    auto* p = DeviceAlloc<float>(logits.size());
    H2D(p, logits);
    d_logits = p;
    vllm_in_dtype = llm_kernels::nvidia::tensorrt_llm::dev::ScalarType::Float;
  } else {
    std::vector<__nv_bfloat16> logits_bf16(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) logits_bf16[i] = __float2bfloat16(logits[i]);
    auto* p = DeviceAlloc<__nv_bfloat16>(logits_bf16.size());
    H2D(p, logits_bf16);
    d_logits = p;
    vllm_in_dtype = llm_kernels::nvidia::tensorrt_llm::dev::ScalarType::BFloat16;
  }

  int32_t* d_ids = DeviceAlloc<int32_t>(input_ids.size());
  int32_t* d_t2e = DeviceAlloc<int32_t>(tid2eid.size());
  H2D(d_ids, input_ids);
  H2D(d_t2e, tid2eid);

  // vllm: index_dtype Int → indices/input_ids/tid2eid all int32.
  float* d_w_vllm = DeviceAlloc<float>(num_tokens * topk);
  int32_t* d_i_vllm = DeviceAlloc<int32_t>(num_tokens * topk);
  int32_t* d_tei = DeviceAlloc<int32_t>(num_tokens * topk);  // scratch, not used on hash path
  // sglang: topk_indices int32; input_ids int32, tid2eid int32.
  float* d_w_sg = DeviceAlloc<float>(num_tokens * topk);
  int32_t* d_i_sg = DeviceAlloc<int32_t>(num_tokens * topk);

  llm_kernels::nvidia::InvokeTopkSoftplusSqrt(
      d_logits, d_w_vllm, d_i_vllm, d_tei, num_tokens, num_experts, topk,
      /*renormalize=*/true, /*scale=*/1.5, /*bias=*/nullptr, d_ids, d_t2e, vllm_in_dtype,
      llm_kernels::nvidia::tensorrt_llm::dev::ScalarType::Int, /*stream=*/0);

  llm_kernels::nvidia::sglang_gate::InvokeTopkSoftplusSqrtSglang(
      d_logits, dtype, d_w_sg, d_i_sg, num_tokens, num_experts, topk, /*renormalize=*/true, /*scale=*/1.5f,
      /*bias=*/nullptr, d_ids, d_t2e, /*stream=*/0);

  cudaDeviceSynchronize();

  auto w_vllm = D2H(d_w_vllm, num_tokens * topk);
  auto i_vllm = D2H(d_i_vllm, num_tokens * topk);
  auto w_sg = D2H(d_w_sg, num_tokens * topk);
  auto i_sg = D2H(d_i_sg, num_tokens * topk);

  // Hash path indices come from tid2eid lookup, so order is fully determined and equal.
  // Weight tolerance bumps to 5e-3 for bf16 (same rationale as ScorePath).
  const float weight_tol = (dtype == GateInputDtype::kFloat32) ? 1e-3f : 5e-3f;
  for (int i = 0; i < num_tokens * topk; ++i) {
    EXPECT_EQ(i_vllm[i], i_sg[i]) << "hash idx mismatch at " << i;
    EXPECT_NEAR(w_vllm[i], w_sg[i], weight_tol) << "hash weight mismatch at " << i;
  }

  cudaFree(d_logits); cudaFree(d_ids); cudaFree(d_t2e);
  cudaFree(d_w_vllm); cudaFree(d_i_vllm); cudaFree(d_tei);
  cudaFree(d_w_sg); cudaFree(d_i_sg);
}

}  // namespace

TEST(SglangTopkSoftplusSqrt, ScorePath_E256_K6_FP32)            { TestScorePath(128, 256, 6, GateInputDtype::kFloat32); }
TEST(SglangTopkSoftplusSqrt, ScorePath_E384_K6_FP32)            { TestScorePath(128, 384, 6, GateInputDtype::kFloat32); }
TEST(SglangTopkSoftplusSqrt, ScorePath_E256_K6_BF16)            { TestScorePath(128, 256, 6, GateInputDtype::kBFloat16); }
TEST(SglangTopkSoftplusSqrt, ScorePath_E384_K6_BF16)            { TestScorePath(128, 384, 6, GateInputDtype::kBFloat16); }
TEST(SglangTopkSoftplusSqrt, ScorePath_E384_K6_BF16_LargeBatch) { TestScorePath(1024, 384, 6, GateInputDtype::kBFloat16); }
TEST(SglangTopkSoftplusSqrt, HashPath_E384_K6_BF16)             { TestHashPath(128, 384, 6, 2048, GateInputDtype::kBFloat16); }
TEST(SglangTopkSoftplusSqrt, HashPath_E256_K6_BF16)             { TestHashPath(128, 256, 6, 2048, GateInputDtype::kBFloat16); }

// ---------------------------------------------------------------------------
// Benchmark: vllm vs sglang backend on identical inputs.
// Prints per-kernel average latency (µs) to stdout for manual inspection.
// ---------------------------------------------------------------------------
namespace {

struct BenchResult {
  float vllm_us;
  float sglang_us;
};

BenchResult BenchScorePath(int num_tokens, int num_experts, int topk, int warmup, int iters) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<__nv_bfloat16> logits(num_tokens * num_experts);
  std::vector<float> bias(num_experts);
  {
    std::vector<float> tmp(logits.size());
    for (auto& x : tmp) x = dist(rng);
    for (size_t i = 0; i < tmp.size(); ++i) logits[i] = __float2bfloat16(tmp[i]);
  }
  for (auto& x : bias) x = dist(rng) * 0.1f;

  auto* d_logits = DeviceAlloc<__nv_bfloat16>(logits.size());
  auto* d_bias = DeviceAlloc<float>(bias.size());
  H2D(d_logits, logits);
  H2D(d_bias, bias);

  auto* d_w = DeviceAlloc<float>(num_tokens * topk);
  auto* d_i = DeviceAlloc<int32_t>(num_tokens * topk);
  auto* d_tei = DeviceAlloc<int32_t>(num_tokens * topk);

  auto vllm_dtype = llm_kernels::nvidia::tensorrt_llm::dev::ScalarType::BFloat16;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // --- vllm ---
  for (int i = 0; i < warmup; ++i)
    llm_kernels::nvidia::InvokeTopkSoftplusSqrt(d_logits, d_w, d_i, d_tei, num_tokens, num_experts, topk, true, 1.5,
                                                 d_bias, nullptr, nullptr, vllm_dtype,
                                                 llm_kernels::nvidia::tensorrt_llm::dev::ScalarType::Int, 0);
  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);
  for (int i = 0; i < iters; ++i)
    llm_kernels::nvidia::InvokeTopkSoftplusSqrt(d_logits, d_w, d_i, d_tei, num_tokens, num_experts, topk, true, 1.5,
                                                 d_bias, nullptr, nullptr, vllm_dtype,
                                                 llm_kernels::nvidia::tensorrt_llm::dev::ScalarType::Int, 0);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float vllm_ms = 0;
  cudaEventElapsedTime(&vllm_ms, start, stop);

  // --- sglang ---
  for (int i = 0; i < warmup; ++i)
    llm_kernels::nvidia::sglang_gate::InvokeTopkSoftplusSqrtSglang(d_logits, GateInputDtype::kBFloat16, d_w, d_i,
                                                                    num_tokens, num_experts, topk, true, 1.5f, d_bias,
                                                                    nullptr, nullptr, 0);
  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);
  for (int i = 0; i < iters; ++i)
    llm_kernels::nvidia::sglang_gate::InvokeTopkSoftplusSqrtSglang(d_logits, GateInputDtype::kBFloat16, d_w, d_i,
                                                                    num_tokens, num_experts, topk, true, 1.5f, d_bias,
                                                                    nullptr, nullptr, 0);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float sglang_ms = 0;
  cudaEventElapsedTime(&sglang_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_logits);
  cudaFree(d_bias);
  cudaFree(d_w);
  cudaFree(d_i);
  cudaFree(d_tei);

  return {vllm_ms * 1000.0f / iters, sglang_ms * 1000.0f / iters};
}

BenchResult BenchHashPath(int num_tokens, int num_experts, int topk, int vocab, int warmup, int iters) {
  std::mt19937 rng(456);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::uniform_int_distribution<int32_t> tid_dist(0, vocab - 1);
  std::uniform_int_distribution<int> eid_dist(0, num_experts - 1);

  std::vector<__nv_bfloat16> logits(num_tokens * num_experts);
  std::vector<int32_t> input_ids(num_tokens);
  std::vector<int32_t> tid2eid(vocab * topk);
  {
    std::vector<float> tmp(logits.size());
    for (auto& x : tmp) x = dist(rng);
    for (size_t i = 0; i < tmp.size(); ++i) logits[i] = __float2bfloat16(tmp[i]);
  }
  for (auto& x : input_ids) x = tid_dist(rng);
  for (auto& x : tid2eid) x = eid_dist(rng);

  auto* d_logits = DeviceAlloc<__nv_bfloat16>(logits.size());
  auto* d_ids = DeviceAlloc<int32_t>(input_ids.size());
  auto* d_t2e = DeviceAlloc<int32_t>(tid2eid.size());
  H2D(d_logits, logits);
  H2D(d_ids, input_ids);
  H2D(d_t2e, tid2eid);

  auto* d_w = DeviceAlloc<float>(num_tokens * topk);
  auto* d_i_vllm = DeviceAlloc<int32_t>(num_tokens * topk);
  auto* d_i_sg = DeviceAlloc<int32_t>(num_tokens * topk);
  auto* d_tei = DeviceAlloc<int32_t>(num_tokens * topk);

  auto vllm_dtype = llm_kernels::nvidia::tensorrt_llm::dev::ScalarType::BFloat16;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // --- vllm ---
  for (int i = 0; i < warmup; ++i)
    llm_kernels::nvidia::InvokeTopkSoftplusSqrt(d_logits, d_w, d_i_vllm, d_tei, num_tokens, num_experts, topk, true,
                                                 1.5, nullptr, d_ids, d_t2e, vllm_dtype,
                                                 llm_kernels::nvidia::tensorrt_llm::dev::ScalarType::Int, 0);
  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);
  for (int i = 0; i < iters; ++i)
    llm_kernels::nvidia::InvokeTopkSoftplusSqrt(d_logits, d_w, d_i_vllm, d_tei, num_tokens, num_experts, topk, true,
                                                 1.5, nullptr, d_ids, d_t2e, vllm_dtype,
                                                 llm_kernels::nvidia::tensorrt_llm::dev::ScalarType::Int, 0);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float vllm_ms = 0;
  cudaEventElapsedTime(&vllm_ms, start, stop);

  // --- sglang ---
  for (int i = 0; i < warmup; ++i)
    llm_kernels::nvidia::sglang_gate::InvokeTopkSoftplusSqrtSglang(d_logits, GateInputDtype::kBFloat16, d_w, d_i_sg,
                                                                    num_tokens, num_experts, topk, true, 1.5f, nullptr,
                                                                    d_ids, d_t2e, 0);
  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);
  for (int i = 0; i < iters; ++i)
    llm_kernels::nvidia::sglang_gate::InvokeTopkSoftplusSqrtSglang(d_logits, GateInputDtype::kBFloat16, d_w, d_i_sg,
                                                                    num_tokens, num_experts, topk, true, 1.5f, nullptr,
                                                                    d_ids, d_t2e, 0);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float sglang_ms = 0;
  cudaEventElapsedTime(&sglang_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_logits);
  cudaFree(d_ids);
  cudaFree(d_t2e);
  cudaFree(d_w);
  cudaFree(d_i_vllm);
  cudaFree(d_i_sg);
  cudaFree(d_tei);

  return {vllm_ms * 1000.0f / iters, sglang_ms * 1000.0f / iters};
}

}  // namespace

TEST(SglangTopkSoftplusSqrt, Benchmark_Score_BF16) {
  constexpr int kWarmup = 20;
  constexpr int kIters = 200;
  const int batch_sizes[] = {1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096};
  const std::pair<int, int> configs[] = {{256, 6}, {384, 6}};

  for (auto [E, K] : configs) {
    printf("\n=== Score Path Benchmark: E=%d, K=%d, BF16 ===\n", E, K);
    printf("%-12s %12s %12s %12s\n", "num_tokens", "vllm(µs)", "sglang(µs)", "speedup");
    printf("%-12s %12s %12s %12s\n", "----------", "--------", "---------", "-------");
    for (int n : batch_sizes) {
      auto r = BenchScorePath(n, E, K, kWarmup, kIters);
      printf("%-12d %12.2f %12.2f %11.2fx\n", n, r.vllm_us, r.sglang_us, r.vllm_us / r.sglang_us);
    }
  }
}

TEST(SglangTopkSoftplusSqrt, Benchmark_Hash_BF16) {
  constexpr int kWarmup = 20;
  constexpr int kIters = 200;
  constexpr int kVocab = 163840;
  const int batch_sizes[] = {1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096};
  const std::pair<int, int> configs[] = {{256, 6}, {384, 6}};

  for (auto [E, K] : configs) {
    printf("\n=== Hash Path Benchmark: E=%d, K=%d, BF16, vocab=%d ===\n", E, K, kVocab);
    printf("%-12s %12s %12s %12s\n", "num_tokens", "vllm(µs)", "sglang(µs)", "speedup");
    printf("%-12s %12s %12s %12s\n", "----------", "--------", "---------", "-------");
    for (int n : batch_sizes) {
      auto r = BenchHashPath(n, E, K, kVocab, kWarmup, kIters);
      printf("%-12d %12.2f %12.2f %11.2fx\n", n, r.vllm_us, r.sglang_us, r.vllm_us / r.sglang_us);
    }
  }
}
