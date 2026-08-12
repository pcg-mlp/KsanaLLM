/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "layers/layer_test_base.h"

#ifdef ENABLE_CUDA

#  include "configure/schedule_config_parser.h"
#  include "csrc/kernels/nvidia/others/humming/preprocess/humming_preprocess.h"
#  include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/marlin_moe_wna16_preprocess.h"
#  include "csrc/utils/nvidia/workspace.h"
#  include "device/nvidia/cuda_utils.h"
#  include "layers/base_moe_layer.h"
#  include "layers/humming_fused_wfp4a16_moe_layer.h"
#  include "layers/humming_fused_wfp4afp8_moe_layer.h"
#  include "layers/marlin_fused_wfp4a16_moe_layer.h"
#  include "layers/moe_layer.h"
#  include "layers/moe_layer_factory.h"
#  include "layers/moe_perf_stats.h"
#  include "utils/tensor.h"

#  include "wfp4_moe_layer_test_common.h"
#  include "wfp4_moe_layer_test_config.h"
#  include "wfp4_moe_layer_test_perf.h"
#  include "wfp4_moe_layer_test_print.h"
#  include "wfp4_moe_layer_test_ref.h"

namespace ksana_llm {

// ===========================================================================
// Full benchmark + accuracy driver for one config.
//
// Serialized design: base weights + max-size input data are generated once,
// then each backend is tested independently and its resources freed before
// the next backend starts.  After perf measurement, dequantized FP32 weights
// are computed for accuracy reference (TP configs only).
//
// Uses accuracy-friendly data (constrained E8M0 scales [119,123), swiglu_limit=10)
// so that GPU reference comparison can run alongside performance measurement.
// Accuracy verification is skipped for EP configs (expert_map remapping not
// modeled in the GPU reference).
// ===========================================================================
void RunBenchmarkForConfig(const Fp4MoeTestConfig& cfg, std::shared_ptr<Context> context,
                           RuntimeConfig& base_runtime_config) {
  if (context->device->ext->GetComputeCapacity() != 90) {
    GTEST_SKIP() << "Requires SM90+";
  }

  // FP4_COMPILE_EXTEND controls whether the humming JIT compiler compiles
  // extend/variant kernel configs (beyond the always-compiled heuristic set).
  // If the user has set it externally, respect their choice.  Otherwise default
  // to 0 (extend configs skipped) so the test exercises the profiling path
  // (QUANT_PROFILE defaults to 15 inside the layer) without paying the
  // multi-minute JIT compilation cost of extend/variant cubins.
  // This keeps CI/CD fast while still covering Preprocess/ProfileAllBuckets.
  if (std::getenv("FP4_COMPILE_EXTEND") == nullptr) {
    setenv("FP4_COMPILE_EXTEND", "0", 1);
  }

  // K3 uses sigmoid scoring + SiTU activation, which only Marlin supports.
  // Humming backends require sqrtsoftplus scoring.  Filter out humming
  // backends for K3 WITHOUT modifying FP4_BACKENDS — the env var must stay
  // unchanged so subsequent tests in the same process see the user's original
  // selection (gtest runs all tests in one process; setenv would leak).
  const bool is_k3 = cfg.is_k3();
  std::vector<BackendSpec> selected_backends = ParseRequestedBackends();
  if (is_k3) {
    std::vector<BackendSpec> k3_backends;
    k3_backends.reserve(selected_backends.size());
    for (const auto& spec : selected_backends) {
      if (spec.is_humming) {
        KLLM_LOG_WARNING << fmt::format("K3 config '{}': skipping '{}' (K3 sigmoid+SiTU only supported by Marlin)",
                                        cfg.name, spec.name);
        continue;
      }
      k3_backends.push_back(spec);
    }
    if (k3_backends.empty()) {
      KLLM_LOG_WARNING << fmt::format("K3 config '{}': no Marlin in FP4_BACKENDS, falling back to Marlin", cfg.name);
      for (const auto& spec : MasterBackendSpecs()) {
        if (spec.backend == Fp4Backend::kMarlin) {
          k3_backends.push_back(spec);
          break;
        }
      }
    }
    selected_backends = std::move(k3_backends);
  }
  KLLM_LOG_INFO << fmt::format("FP4_BACKENDS: running {} backends", selected_backends.size());

  constexpr int kDeviceRank = 0;
  cudaStream_t stream = GetStream(context);
  size_t K = cfg.hidden_size();
  size_t E = cfg.num_experts();
  size_t max_ntok = GetTokenSizes().back();
  const bool is_ep = (cfg.expert_para_size > 1);

  // Accuracy constants (applied to all configs; EP skips comparison)
  // Backend cosine-sim thresholds now live in MasterBackendSpecs() (BackendSpec::cos_threshold).
  // Activation params (swiglu_limit / situ_beta / situ_linear_beta) come from cfg (ModelSpec).
  constexpr int kScaleLo = 119;
  constexpr int kScaleHi = 123;

  // Configure EP so layers properly simulate expert parallelism:
  // sets global_expert_para_size_ = ep_size * expert_world_size, triggers EP
  // routing (ExpertMap + InvokeMoeAlignBlockSize with EP flag) in Forward.
  base_runtime_config.parallel_basic_config.expert_parallel_size = cfg.expert_para_size;

  // =========================================================================
  // 1. Generate base weights once (seed 777, constrained scales for accuracy)
  // =========================================================================
  torch::manual_seed(777);
  torch::cuda::manual_seed(777);
  ProcessedWeights base = GenerateBaseWeights(cfg, kScaleLo, kScaleHi);

  // =========================================================================
  // 2. Pre-generate max-size input data (seed 888).
  //    All backends slice from these, guaranteeing identical inputs per M.
  // =========================================================================
  torch::manual_seed(888);
  torch::cuda::manual_seed(888);
  auto bf16_opt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBFloat16);
  torch::Tensor hs_full = torch::randn({static_cast<int64_t>(max_ntok), static_cast<int64_t>(K)}, bf16_opt);
  torch::Tensor gate_full = torch::randn({static_cast<int64_t>(max_ntok), static_cast<int64_t>(E)}, bf16_opt);

  // e_score_correction_bias: 1-D float32 [num_experts], generated with a distinct
  // seed (999) so it is independent of gate_full (seed 888). All models use
  // e_score_correction_bias (noaux_tc routing), so always generate it.
  torch::manual_seed(999);
  torch::cuda::manual_seed(999);
  auto f32_opt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat32);
  torch::Tensor e_score_bias = torch::randn({static_cast<int64_t>(E)}, f32_opt);

  const void* hs_ptr = hs_full.data_ptr();
  const void* gate_ptr = gate_full.data_ptr();

  // =========================================================================
  // 3. Shared params (swiglu_limit for accuracy; perf unaffected)
  // =========================================================================
  auto params = BuildMoeParams(cfg, max_ntok);

  // Pre-compute expert routing stats for all token sizes (shared across backends).
  // Uses the same gate_full tensor that all backends slice from.
  std::vector<RoutingStats> routing_stats;
  routing_stats.reserve(GetTokenSizes().size());
  for (size_t num_tokens : GetTokenSizes()) {
    routing_stats.push_back(ComputeRoutingStats(cfg, num_tokens, gate_full));
  }

  std::vector<std::vector<BenchResult>> all_results;
  std::vector<std::vector<torch::Tensor>> all_outputs;  // [backend][token_size]

  // =========================================================================
  // 4-6. Run each backend (serialized)
  // =========================================================================
  auto run_backend = [&](Fp4Backend backend) {
    ProcessedWeights pw;
    switch (backend) {
      case Fp4Backend::kMarlin:
        pw = ProcessWeightsMarlin(base, cfg, stream);
        break;
      case Fp4Backend::kHummingW4A16:
        pw = ProcessWeightsHummingW4A16(base, cfg, stream);
        break;
      case Fp4Backend::kHummingW4A8:
        pw = ProcessWeightsHummingW4A8(base, cfg, stream);
        break;
      default:
        KLLM_THROW(fmt::format("Unsupported backend: {}", static_cast<int>(backend)));
    }
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    RuntimeConfig rc = base_runtime_config;
    rc.inter_data_type = TYPE_BF16;
    auto layer = CreateLayer(backend, params, rc, context, kDeviceRank);

    auto inputs = SetupPersistentInputs(pw, backend, stream);
    cudaStreamSynchronize(stream);

    if (backend == Fp4Backend::kMarlin) {
      auto [results, outputs] = RunBenchmark(static_cast<MarlinFusedWfp4A16MoeLayer*>(layer.get()), inputs, cfg, stream,
                                             hs_ptr, gate_ptr, routing_stats, backend, e_score_bias);
      all_results.push_back(std::move(results));
      all_outputs.push_back(std::move(outputs));
    } else if (backend == Fp4Backend::kHummingW4A16) {
      auto [results, outputs] = RunBenchmark(static_cast<HummingFusedWfp4A16MoeLayer*>(layer.get()), inputs, cfg,
                                             stream, hs_ptr, gate_ptr, routing_stats, backend, e_score_bias);
      all_results.push_back(std::move(results));
      all_outputs.push_back(std::move(outputs));
    } else if (backend == Fp4Backend::kHummingW4A8) {
      auto [results, outputs] = RunBenchmark(static_cast<HummingFusedWfp4Afp8MoeLayer*>(layer.get()), inputs, cfg,
                                             stream, hs_ptr, gate_ptr, routing_stats, backend, e_score_bias);
      all_results.push_back(std::move(results));
      all_outputs.push_back(std::move(outputs));
    }
  };

  for (const auto& spec : selected_backends) {
    run_backend(spec.backend);
  }

  // =========================================================================
  // 7. Print performance results
  // =========================================================================
  PrintLatencyTable(cfg.name, all_results, selected_backends);
  PrintExpertRoutingSummary(cfg.name, cfg, all_results, selected_backends);
  PrintEffectiveTFLOPS(cfg.name, all_results, selected_backends);
  PrintRealTFLOPS(cfg.name, all_results, selected_backends);
  PrintEffectiveBandwidth(cfg.name, all_results, selected_backends);
  PrintRealBandwidth(cfg.name, all_results, selected_backends);

  // Humming-only detailed analysis (printed for all configs, TP and EP)
  PrintJitKernelTable(cfg.name, cfg);
  PrintSelectedKernels(cfg.name, all_results, selected_backends);
  PrintHummingNonGemmBreakdown(cfg.name, all_results, selected_backends);

  // Write CSV files if FP4_PERF_CSV=1 (terminal printing is unaffected).
  WritePerfCsv(cfg.name, cfg, all_results, selected_backends);

  // =========================================================================
  // 8. Accuracy verification (TP configs only)
  //    Each selected backend independently compared against GPU torch reference.
  // =========================================================================
  if (is_ep) {
    return;
  }

  auto accuracy_results = VerifyAccuracy(base, cfg, all_outputs, selected_backends, hs_full, gate_full, e_score_bias);
  for (const auto& r : accuracy_results) {
    EXPECT_TRUE(r.passed) << cfg.name << " " << r.spec->name << " M=" << r.num_tokens << " cos_sim=" << r.cos_sim
                          << " (threshold " << r.threshold << ")";
  }
}

// NOTE(jinxcwu) 完整性能测试时可以打开所有情况的测试，目前为了节省测试时间只打开一个
TEST_F(LayerTest, V4Flash_TP4) { RunBenchmarkForConfig(kV4FlashTP4, context_, runtime_config); }
TEST_F(LayerTest, K3_TP8) { RunBenchmarkForConfig(kK3TP8, context_, runtime_config); }
TEST_F(LayerTest, DISABLED_V4Flash_EP4) { RunBenchmarkForConfig(kV4FlashEP4, context_, runtime_config); }
TEST_F(LayerTest, DISABLED_V4Pro_TP8) { RunBenchmarkForConfig(kV4ProTP8, context_, runtime_config); }
TEST_F(LayerTest, DISABLED_V4Pro_EP8) { RunBenchmarkForConfig(kV4ProEP8, context_, runtime_config); }

}  // namespace ksana_llm

#endif  // ENABLE_CUDA
