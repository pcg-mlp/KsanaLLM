/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include "flash_sparse_mla_test_common.h"
#include "flash_sparse_mla_test_config.h"
#include "flash_sparse_mla_test_perf.h"
#include "flash_sparse_mla_test_ref.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

void RunMlaBenchmark(LlamaNvidiaFlashSparseMlaTestSuit& fx, ModelType type, const MlaTestRange& range) {
  std::vector<std::vector<MlaPerfRow>> all_results(fx.selected_backends.size());

  for (const auto& cfg : kModelConfigs) {
    if (cfg.type != type) continue;
    for (size_t num_heads_q : range.h_q_cases) {
      for (auto& v : all_results) v.clear();

      for (size_t batch_size : range.batch_sizes) {
        for (size_t seqlen_k : range.seqlen_ks) {
          for (size_t seqlen_q : range.seqlen_q_cases) {
            for (IndicesStrategy idx_strategy : range.idx_strategies) {
              SCOPED_TRACE(std::string(cfg.name) + " h_q=" + std::to_string(num_heads_q) +
                           " s_q=" + std::to_string(seqlen_q) + " s_k=" + std::to_string(seqlen_k) +
                           " idx=" + std::to_string((int)idx_strategy));
              fx.d_kcache.reset();
              fx.d_kcache_fp8.reset();
              if (cfg.IsModel1()) {
                fx.d_extra_kcache.reset();
                fx.d_extra_kcache_fp8.reset();
              }
              c10::cuda::CUDACachingAllocator::emptyCache();
              fx.PrepareData(cfg, batch_size, seqlen_q, seqlen_k, num_heads_q, idx_strategy);

              for (size_t b = 0; b < fx.selected_backends.size(); ++b) {
                const auto& spec = fx.selected_backends[b];
                if (spec.backend == MlaBackend::kFp8DsMla && num_heads_q == 128) {
                  continue;
                }
                if (range.run_perf) {
                  MeasureAndSave(fx, cfg, batch_size, seqlen_q, seqlen_k, num_heads_q, idx_strategy, range, b,
                                 all_results);
                }
                if (range.run_accuracy &&
                    (spec.backend == MlaBackend::kFp8E4M3 ||
                     (spec.backend == MlaBackend::kFp8E4M3SwapAB && num_heads_q == 16)) &&
                    seqlen_k >= cfg.topk * kBlockSize) {
                  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(fx.stream));
                  auto acc =
                      VerifyAccuracy(fx, cfg, batch_size, seqlen_q, num_heads_q, seqlen_k, idx_strategy, fx.d_out);
                  EXPECT_TRUE(acc.passed)
                      << cfg.name << " h_q=" << num_heads_q << " s_q=" << seqlen_q << " s_k=" << seqlen_k
                      << " idx=" << (int)idx_strategy << " max_abs_err=" << acc.max_abs_err
                      << " ref_abs=" << acc.ref_abs << " has_nan=" << acc.has_nan;
                }
              }
            }
          }
        }
      }

      if (range.run_perf) {
        PrintMlaLatencyTable(cfg, num_heads_q, all_results, fx.selected_backends, range.idx_strategies);
      }
    }
  }
}

TEST_F(LlamaNvidiaFlashSparseMlaTestSuit, FlashSparseMlaKernelV32CICITest) {
  RunMlaBenchmark(*this, ModelType::kV32, kCICIRange);
}

TEST_F(LlamaNvidiaFlashSparseMlaTestSuit, FlashSparseMlaKernelModel1CICITest) {
  RunMlaBenchmark(*this, ModelType::kMODEL1, kCICIRange);
}

TEST_F(LlamaNvidiaFlashSparseMlaTestSuit, DISABLED_FlashSparseMlaKernelV32FullTest) {
  RunMlaBenchmark(*this, ModelType::kV32, kFullRange);
}

TEST_F(LlamaNvidiaFlashSparseMlaTestSuit, DISABLED_FlashSparseMlaKernelModel1FullTest) {
  RunMlaBenchmark(*this, ModelType::kMODEL1, kFullRange);
}

TEST_F(LlamaNvidiaFlashSparseMlaTestSuit, DISABLED_FlashSparseMlaKernelV32FastTest) {
  RunMlaBenchmark(*this, ModelType::kV32, kFastRange);
}

TEST_F(LlamaNvidiaFlashSparseMlaTestSuit, DISABLED_FlashSparseMlaKernelModel1FastTest) {
  RunMlaBenchmark(*this, ModelType::kMODEL1, kFastRange);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
