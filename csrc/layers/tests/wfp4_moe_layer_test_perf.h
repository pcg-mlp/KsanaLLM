/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

// FP4 MoE test — performance measurement: BenchResult, FLOPs/bytes computation,
// CUDA-event timing harness, benchmark-run template (templated on layer type),
// and TFLOPS/bandwidth converters.
//
// Expected to be included after the layer/kernel/torch headers from
// wfp4_moe_layer_test.cpp's ENABLE_CUDA block.
#include <torch/torch.h>

#include <functional>
#include <utility>

#include "wfp4_moe_layer_test_config.h"

namespace ksana_llm {

// ===========================================================================
// Trait: true for both Humming FP4 MoE layer types (A16 and A8).
// Used to select the deferred-perf-flush path in RunBenchmark.
// Specializations are guarded by ENABLE_CUDA because the layer classes are
// only defined under that guard.
// ===========================================================================
template <typename LayerT>
struct IsHummingFp4MoeLayer : std::false_type {};

#ifdef ENABLE_CUDA
template <>
struct IsHummingFp4MoeLayer<HummingFusedWfp4A16MoeLayer> : std::true_type {};
template <>
struct IsHummingFp4MoeLayer<HummingFusedWfp4Afp8MoeLayer> : std::true_type {};
#endif

template <typename LayerT>
inline constexpr bool IsHummingFp4MoeLayerV = IsHummingFp4MoeLayer<LayerT>::value;

// ===========================================================================
// Benchmark result for one (layer, token_count) pair
// ===========================================================================
struct BenchResult {
  float total_ms;
  float gemm1_ms;
  float gemm2_ms;
  float nogemm_ms;  // total - gemm1 - gemm2 (driver-computed)
  // Humming-only sub-phase breakdown (0 for marlin)
  float gating_ms = 0.f;
  float align_ms = 0.f;
  float quant_ms = 0.f;
  float silu_ms = 0.f;
  float mulsum_ms = 0.f;
  // Humming-only: selected JIT kernel meta (copied by value; the shared static
  // jit_compiler_'s kernel_table_ vector may reallocate when a later backend
  // compiles its kernels, invalidating raw pointers captured by an earlier backend).
  bool has_gemm1_meta = false;
  llm_kernels::nvidia::humming::KernelMeta gemm1_meta{};
  bool has_gemm2_meta = false;
  llm_kernels::nvidia::humming::KernelMeta gemm2_meta{};
  // Cost model: dual-track effective (no padding) vs real (with block-size
  // padding) FLOPs/bytes, plus routing summary. Filled by RunBenchmark via
  // ComputeCostModel — default-initialized so a BenchResult is well-formed
  // before that.
  // Block sizes captured from layer (via MoePerfStats)
  int32_t moe_block_size_1 = 0;
  int32_t moe_block_size_2 = 0;
  // Effective (no padding)
  double eff_gemm1_flops = 0.0;
  double eff_gemm2_flops = 0.0;
  double eff_total_flops = 0.0;
  double eff_gemm1_bytes = 0.0;
  double eff_gemm2_bytes = 0.0;
  double eff_total_bytes = 0.0;
  // Real (with block-size padding)
  double real_gemm1_flops = 0.0;
  double real_gemm2_flops = 0.0;
  double real_total_flops = 0.0;
  double real_gemm1_bytes = 0.0;
  double real_gemm2_bytes = 0.0;
  double real_total_bytes = 0.0;
  // Routing summary (for routing table)
  size_t experts_accessed = 0;
  size_t total_pairs = 0;
  size_t real_pairs_g1 = 0;
  size_t real_pairs_g2 = 0;
  double waste_rate_g1 = 0.0;
  double waste_rate_g2 = 0.0;
};

// ===========================================================================
// Expert routing analysis for one (config, num_tokens) pair.
// Backend-independent: computed once per token size, shared across all backends.
// ===========================================================================
struct RoutingStats {
  size_t num_tokens;
  size_t topk;                  // always 6
  size_t experts_total;         // E_local (experts per rank)
  size_t total_pairs;           // M*topk (TP) or local_pairs (EP)
  size_t experts_accessed;      // distinct experts with >= 1 hit
  std::vector<int> hit_counts;  // [experts_total], per-expert unpadded hit count
  bool is_ep;
};

// ===========================================================================
// Activation byte configuration per backend (fixes W4A8 bug).
// GEMM input activation: FP8 (1B) for W4A8, BF16 (2B) for W4A16/marlin.
// GEMM output: always BF16 (2B).
// ===========================================================================
struct ActivationByteConfig {
  double act_in_bytes;   // GEMM input: 1 (FP8) or 2 (BF16)
  double act_out_bytes;  // GEMM output: 2 (BF16)
};

inline ActivationByteConfig GetActivationByteConfig(Fp4Backend backend) {
  constexpr double kBF16 = 2.0;
  constexpr double kFP8 = 1.0;
  if (backend == Fp4Backend::kHummingW4A8) {
    return {kFP8, kBF16};
  }
  return {kBF16, kBF16};  // marlin, humming_w4a16
}

// ===========================================================================
// Compute expert routing stats by replicating the layer's gating logic in torch.
//
// Gating: scores = sqrt(softplus(gate)), topk=6, largest=true.
// This matches TopKSoftplusSqrt (validated by accuracy verification — cosine
// similarity passes against the same logic in wfp4_moe_layer_test_ref.h).
//
// TP: hit_counts[e] = count of all topk_ids == e across all M*topk pairs.
// EP: only pairs where expert ∈ [0, E_local) are counted (rank 0's local
//     experts). The layer's ExpertMap remaps non-local to -1; InvokeMoeAlignBlockSize
//     with EP flag drops -1 entries. total_pairs reflects rank 0's actual work.
// ===========================================================================
inline RoutingStats ComputeRoutingStats(const Fp4MoeTestConfig& cfg, size_t num_tokens,
                                        const torch::Tensor& gate_full) {
  RoutingStats rs;
  rs.num_tokens = num_tokens;
  rs.topk = cfg.topk();
  rs.experts_total = ComputeExpertsPerRank(cfg);
  rs.is_ep = (cfg.expert_para_size > 1);

  size_t E_local = rs.experts_total;
  int64_t M = static_cast<int64_t>(num_tokens);
  int64_t topk = static_cast<int64_t>(cfg.topk());

  // Slice gate to [0, M) and compute gating (same as ref.h GpuRefFp4MoeForward)
  auto gate_m = gate_full.slice(0, 0, M);
  auto gate_f32 = gate_m.to(torch::kFloat32);
  torch::Tensor scores;
  if (cfg.is_k3()) {
    scores = torch::sigmoid(gate_f32);
  } else {
    scores = torch::sqrt(torch::nn::functional::softplus(gate_f32));
  }
  auto topk_result = torch::topk(scores, topk, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
  auto topk_ids = std::get<1>(topk_result).to(torch::kInt64);  // [M, topk]
  auto flat_ids = topk_ids.reshape(-1);                        // [M*topk]

  rs.hit_counts.assign(E_local, 0);

  if (!rs.is_ep) {
    // TP: count all pairs
    rs.total_pairs = static_cast<size_t>(M * topk);
    for (size_t e = 0; e < E_local; ++e) {
      rs.hit_counts[e] = static_cast<int>((flat_ids == static_cast<int64_t>(e)).sum().item<int64_t>());
    }
  } else {
    // EP: only count pairs where expert ∈ [0, E_local) (rank 0's local experts)
    auto local_mask = flat_ids < static_cast<int64_t>(E_local);
    rs.total_pairs = static_cast<size_t>(local_mask.sum().item<int64_t>());
    auto local_ids = flat_ids.masked_select(local_mask);  // [local_pairs]
    for (size_t e = 0; e < E_local; ++e) {
      rs.hit_counts[e] = static_cast<int>((local_ids == static_cast<int64_t>(e)).sum().item<int64_t>());
    }
  }

  rs.experts_accessed = 0;
  for (size_t e = 0; e < E_local; ++e) {
    if (rs.hit_counts[e] > 0) ++rs.experts_accessed;
  }

  return rs;
}

// ===========================================================================
// Compute effective + real FLOPs and bytes for one (config, token_size, backend).
//
// Effective = without padding (actual useful computation).
// Real = with block-size padding (what the GPU actually computes).
//
// FLOPS:
//   GEMM1: 2 * pairs * K * (inter_per_rank * 2)   [fused gate+up]
//   GEMM2: 2 * pairs * inter_per_rank * K
//
// Bytes:
//   Weights: experts_accessed * dims * (kFp4Bytes + kScaleBytes) — same eff/real
//   Activations: pairs * dims * act_bytes — eff uses eff_pairs, real uses real_pairs
//   FP8 per-token scale (W4A8 only): pairs * 4 bytes
// ===========================================================================
inline void ComputeCostModel(const Fp4MoeTestConfig& cfg, const RoutingStats& rs, Fp4Backend backend, int32_t bs1,
                             int32_t bs2, BenchResult& br) {
  size_t inter_per_rank = ComputeInterPerRank(cfg);
  size_t K = cfg.hidden_size();
  double N1 = static_cast<double>(inter_per_rank) * 2.0;  // GEMM1 N (fused gate+up)
  double N2 = static_cast<double>(inter_per_rank);        // GEMM2 K
  double Kd = static_cast<double>(K);

  constexpr double kFp4Bytes = 0.5;         // 4 bits per element
  constexpr double kScaleBytes = 1.0 / 32;  // 1 byte (E8M0) per 32 elements

  bool is_w4a8 = (backend == Fp4Backend::kHummingW4A8);
  ActivationByteConfig act_cfg = GetActivationByteConfig(backend);

  // --- Per-expert padded counts ---
  double eff_pairs = static_cast<double>(rs.total_pairs);
  double real_pairs_g1 = 0.0;
  double real_pairs_g2 = 0.0;
  for (size_t e = 0; e < rs.experts_total; ++e) {
    int hit = rs.hit_counts[e];
    if (hit == 0) continue;
    if (bs1 > 0) {
      real_pairs_g1 += static_cast<double>(((hit + bs1 - 1) / bs1) * bs1);
    } else {
      real_pairs_g1 += static_cast<double>(hit);  // fallback: no padding info
    }
    if (bs2 > 0) {
      real_pairs_g2 += static_cast<double>(((hit + bs2 - 1) / bs2) * bs2);
    } else {
      real_pairs_g2 += static_cast<double>(hit);
    }
  }

  // --- FLOPS ---
  // GEMM1: 2 * pairs * K * N1
  br.eff_gemm1_flops = 2.0 * eff_pairs * Kd * N1;
  br.real_gemm1_flops = 2.0 * real_pairs_g1 * Kd * N1;
  // GEMM2: 2 * pairs * N2 * K
  br.eff_gemm2_flops = 2.0 * eff_pairs * N2 * Kd;
  br.real_gemm2_flops = 2.0 * real_pairs_g2 * N2 * Kd;
  br.eff_total_flops = br.eff_gemm1_flops + br.eff_gemm2_flops;
  br.real_total_flops = br.real_gemm1_flops + br.real_gemm2_flops;

  // --- Bytes ---
  double experts_accessed = static_cast<double>(rs.experts_accessed);
  // Weights (same for effective and real — loaded once per accessed expert)
  double gemm1_wt = experts_accessed * N1 * Kd * (kFp4Bytes + kScaleBytes);
  double gemm2_wt = experts_accessed * Kd * N2 * (kFp4Bytes + kScaleBytes);

  // Activations: GEMM1 in=[pairs,K] out=[pairs,N1]; GEMM2 in=[pairs,N2] out=[pairs,K]
  double eff_g1_act = eff_pairs * Kd * act_cfg.act_in_bytes + eff_pairs * N1 * act_cfg.act_out_bytes;
  double real_g1_act = real_pairs_g1 * Kd * act_cfg.act_in_bytes + real_pairs_g1 * N1 * act_cfg.act_out_bytes;
  double eff_g2_act = eff_pairs * N2 * act_cfg.act_in_bytes + eff_pairs * Kd * act_cfg.act_out_bytes;
  double real_g2_act = real_pairs_g2 * N2 * act_cfg.act_in_bytes + real_pairs_g2 * Kd * act_cfg.act_out_bytes;

  // FP8 per-token scale (W4A8 only): one float per token-expert pair, inlined
  // into each GEMM's bytes (GEMM1 reads [pairs] scales, GEMM2 reads [pairs] scales).
  // GEMM1 FP8 scale: approximated as pairs*4 (layer quantizes M tokens before topk expansion; ~0.1% overcount,
  // spec-blessed)
  br.eff_gemm1_bytes = gemm1_wt + eff_g1_act + (is_w4a8 ? eff_pairs * 4.0 : 0.0);
  br.eff_gemm2_bytes = gemm2_wt + eff_g2_act + (is_w4a8 ? eff_pairs * 4.0 : 0.0);
  br.real_gemm1_bytes = gemm1_wt + real_g1_act + (is_w4a8 ? real_pairs_g1 * 4.0 : 0.0);
  br.real_gemm2_bytes = gemm2_wt + real_g2_act + (is_w4a8 ? real_pairs_g2 * 4.0 : 0.0);
  br.eff_total_bytes = br.eff_gemm1_bytes + br.eff_gemm2_bytes;
  br.real_total_bytes = br.real_gemm1_bytes + br.real_gemm2_bytes;

  // --- Routing summary fields ---
  br.experts_accessed = rs.experts_accessed;
  br.total_pairs = rs.total_pairs;
  br.real_pairs_g1 = static_cast<size_t>(real_pairs_g1);
  br.real_pairs_g2 = static_cast<size_t>(real_pairs_g2);
  br.waste_rate_g1 = real_pairs_g1 > 0 ? (real_pairs_g1 - eff_pairs) / real_pairs_g1 : 0.0;
  br.waste_rate_g2 = real_pairs_g2 > 0 ? (real_pairs_g2 - eff_pairs) / real_pairs_g2 : 0.0;
}

// ===========================================================================
// Measure one (num_tokens, layer) configuration.
// Encapsulates warmup + benchmark + per-GEMM timing, similar in spirit to
// MeasureCudaExecutionTime but also harvests MoePerfStats from the layer.
// Returns a BenchResult with timing fields filled; FLOPs/bytes are left at 0
// for the caller (RunBenchmark) to fill — timing and cost-model accounting are
// separate concerns.
// ===========================================================================
template <typename Func>
BenchResult MeasureMoeCudaExecutionTime(Func&& forward, MoePerfStats* perf_stats, cudaStream_t stream, int warmups,
                                        int iterations, std::function<void(size_t)> flush = nullptr) {
  // Warmup
  for (int i = 0; i < warmups; ++i) forward();
  cudaDeviceSynchronize();

  // Reset perf stats after warmup so only benchmark iterations count
  perf_stats->Reset();

  // Benchmark: measure total GPU time with stream events
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, stream);
  for (int i = 0; i < iterations; ++i) forward();
  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);

  float total_ms;
  cudaEventElapsedTime(&total_ms, begin, end);
  cudaEventDestroy(begin);
  cudaEventDestroy(end);

  // Layer-specific deferred flush: read the last iteration's per-phase event markers
  // and populate perf_stats as an accumulated sum. No-op for layers that don't defer.
  if (flush != nullptr) {
    flush(static_cast<size_t>(iterations));
  }

  BenchResult br;
  br.total_ms = total_ms / iterations;
  br.gemm1_ms = perf_stats->gemm1_time_ms / iterations;
  br.gemm2_ms = perf_stats->gemm2_time_ms / iterations;
  br.nogemm_ms = br.total_ms - br.gemm1_ms - br.gemm2_ms;
  // Humming-only sub-phase times (0 for marlin — fields stay at default)
  br.gating_ms = perf_stats->gating_time_ms / iterations;
  br.align_ms = perf_stats->align_time_ms / iterations;
  br.quant_ms = perf_stats->quant_time_ms / iterations;
  br.silu_ms = perf_stats->silu_time_ms / iterations;
  br.mulsum_ms = perf_stats->mulsum_time_ms / iterations;
  // Humming-only: selected kernel meta (nullptr for marlin — same meta across all iters).
  // Copy by value: the meta pointer points into the static jit_compiler_'s
  // kernel_table_, which may be reallocated when a subsequent backend compiles.
  if (perf_stats->last_gemm1_meta != nullptr) {
    br.gemm1_meta = *perf_stats->last_gemm1_meta;
    br.has_gemm1_meta = true;
  }
  if (perf_stats->last_gemm2_meta != nullptr) {
    br.gemm2_meta = *perf_stats->last_gemm2_meta;
    br.has_gemm2_meta = true;
  }

  return br;
}

// ===========================================================================
// Run benchmark across all token sizes for one (layer, inputs) pair.
// Templated on the concrete layer type so we can call SetPerfStats directly.
// Also captures the last forward output per token size for accuracy checking.
// ===========================================================================
template <typename LayerT>
std::pair<std::vector<BenchResult>, std::vector<torch::Tensor>> RunBenchmark(
    LayerT* layer, std::vector<Tensor>& persistent_inputs, const Fp4MoeTestConfig& cfg, cudaStream_t stream,
    const void* hs_full, const void* gate_full, const std::vector<RoutingStats>& routing_stats, Fp4Backend backend,
    const torch::Tensor& e_score_bias = {}) {
  static constexpr size_t kBytesPerBF16 = 2;
  size_t K = cfg.hidden_size();
  size_t E = cfg.num_experts();

  MoePerfStats perf_stats;
  layer->SetPerfStats(&perf_stats);

  std::vector<BenchResult> results;
  results.reserve(kTokenSizes.size());
  std::vector<torch::Tensor> outputs;
  outputs.reserve(kTokenSizes.size());

  // Wrap e_score_bias into a ksana Tensor for Forward's 8th argument.
  Tensor e_bias_tensor;
  if (e_score_bias.defined()) {
    e_bias_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {E});
    cudaMemcpyAsync(e_bias_tensor.GetPtr<void>(), e_score_bias.data_ptr(), E * sizeof(float), cudaMemcpyDeviceToDevice,
                    stream);
  }

  for (size_t idx = 0; idx < kTokenSizes.size(); ++idx) {
    size_t num_tokens = kTokenSizes[idx];
    persistent_inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, K});
    persistent_inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, E});
    cudaMemcpyAsync(persistent_inputs[0].GetPtr<void>(), hs_full, num_tokens * K * kBytesPerBF16,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(persistent_inputs[1].GetPtr<void>(), gate_full, num_tokens * E * kBytesPerBF16,
                    cudaMemcpyDeviceToDevice, stream);

    Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, K});
    Tensor eplb_map;
    Tensor workspace;

    auto forward = [&]() {
      layer->Forward(&persistent_inputs[0], &persistent_inputs[1], &persistent_inputs[2], &persistent_inputs[3],
                     &eplb_map, &output, &workspace, e_score_bias.defined() ? &e_bias_tensor : nullptr);
    };

    if constexpr (IsHummingFp4MoeLayerV<LayerT>) {
      // Deferred flush: Forward records events but skips per-call sync; we read the
      // last iteration's markers once here, scaling by kBenchIters.
      layer->SetDeferPerfFlush(true);
      auto flush_fn = [layer](size_t iters) { layer->FlushPerfStats(iters); };
      BenchResult br = MeasureMoeCudaExecutionTime(forward, &perf_stats, stream, kWarmupIters, kBenchIters, flush_fn);
      br.moe_block_size_1 = perf_stats.last_moe_block_size_1;
      br.moe_block_size_2 = perf_stats.last_moe_block_size_2;
      ComputeCostModel(cfg, routing_stats[idx], backend, br.moe_block_size_1, br.moe_block_size_2, br);
      results.push_back(std::move(br));
    } else {
      // Marlin (or other): legacy per-Forward flush, no callback.
      BenchResult br = MeasureMoeCudaExecutionTime(forward, &perf_stats, stream, kWarmupIters, kBenchIters);
      br.moe_block_size_1 = perf_stats.last_moe_block_size_1;
      br.moe_block_size_2 = perf_stats.last_moe_block_size_2;
      ComputeCostModel(cfg, routing_stats[idx], backend, br.moe_block_size_1, br.moe_block_size_2, br);
      results.push_back(std::move(br));
    }

    // Capture output as FP32 on GPU for accuracy verification
    cudaStreamSynchronize(stream);
    auto gpu_out = torch::from_blob(output.GetPtr<void>(), {static_cast<int64_t>(num_tokens), static_cast<int64_t>(K)},
                                    torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBFloat16))
                       .to(torch::kFloat32);
    outputs.push_back(std::move(gpu_out));
  }

  layer->SetPerfStats(nullptr);
  return {results, outputs};
}

// ===========================================================================
// Compute TFLOPS from FLOPs and time
// ===========================================================================
inline double ComputeTFLOPS(double flops, float time_ms) {
  return time_ms > 0.f ? (flops / (time_ms / 1e3)) / 1e12 : 0.;
}

// ===========================================================================
// Compute bandwidth (GB/s) from bytes and time
// ===========================================================================
inline double ComputeBandwidthGBps(double bytes, float time_ms) {
  return time_ms > 0.f ? bytes / (time_ms / 1e3) / 1e9 : 0.;
}

}  // namespace ksana_llm
