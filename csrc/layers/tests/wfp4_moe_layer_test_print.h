/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

// FP4 MoE test — result printing: latency, TFLOPS, bandwidth, JIT kernel table,
// selected-kernel schedule, and humming non-GEMM breakdown tables.
//
// Expected to be included after the layer/kernel/torch headers from
// wfp4_moe_layer_test.cpp's ENABLE_CUDA block.

#include <sys/stat.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "wfp4_moe_layer_test_config.h"
#include "wfp4_moe_layer_test_perf.h"

namespace ksana_llm {

// ===========================================================================
// Print latency table.  Cell format: "g1/g2/nogemm/total" (ms, 4 decimals).
// nogemm = total - gemm1 - gemm2 (driver-computed; works for all backends).
// ===========================================================================
void PrintLatencyTable(const std::string& config_name, const std::vector<std::vector<BenchResult>>& all_results,
                       const std::vector<BackendSpec>& backends) {
  static constexpr int kCellW = 32;  // "0.0123/0.0456/0.0789/0.1368" fits in 32 chars

  std::cout << "\n=== " << config_name << " Latency (ms, GEMM1 / GEMM2 / NoGEMM / Total) ===" << std::endl;
  std::cout << " " << std::setw(10) << "tokens";
  for (const auto& b : backends) {
    std::cout << std::setw(kCellW) << b.name;
  }
  std::cout << std::endl;

  for (size_t i = 0; i < GetTokenSizes().size(); ++i) {
    std::cout << " " << std::setw(10) << GetTokenSizes()[i];
    for (size_t b = 0; b < all_results.size(); ++b) {
      if (i < all_results[b].size()) {
        const auto& r = all_results[b][i];
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << r.gemm1_ms << "/" << r.gemm2_ms << "/" << r.nogemm_ms << "/"
            << r.total_ms;
        std::cout << std::setw(kCellW) << oss.str();
      } else {
        std::cout << std::setw(kCellW) << "N/A";
      }
    }
    std::cout << std::endl;
  }
}

// ===========================================================================
// Generic TFLOPS/Bandwidth table printer.
// field_selector extracts the relevant (g1, g2, total) triple from BenchResult.
// unit_label is "TFLOPS" or "Bandwidth (GB/s)".
// ===========================================================================
template <typename Selector>
void PrintPerfTable(const std::string& config_name, const std::string& table_label, const std::string& unit_label,
                    const std::vector<std::vector<BenchResult>>& all_results, const std::vector<BackendSpec>& backends,
                    Selector selector) {
  static constexpr int kCellW = 24;

  std::cout << "\n=== " << config_name << " " << table_label << " (" << unit_label
            << ", GEMM1 / GEMM2 / Total) ===" << std::endl;
  std::cout << " " << std::setw(10) << "tokens";
  for (const auto& b : backends) {
    std::cout << std::setw(kCellW) << b.name;
  }
  std::cout << std::endl;

  for (size_t i = 0; i < GetTokenSizes().size(); ++i) {
    std::cout << " " << std::setw(10) << GetTokenSizes()[i];
    for (size_t b = 0; b < all_results.size(); ++b) {
      if (i < all_results[b].size()) {
        const auto& r = all_results[b][i];
        auto [f1, f2, ft] = selector(r);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << f1 << "/" << f2 << "/" << ft;
        std::cout << std::setw(kCellW) << oss.str();
      } else {
        std::cout << std::setw(kCellW) << "N/A";
      }
    }
    std::cout << std::endl;
  }
}

void PrintEffectiveTFLOPS(const std::string& config_name, const std::vector<std::vector<BenchResult>>& all_results,
                          const std::vector<BackendSpec>& backends) {
  PrintPerfTable(config_name, "Effective TFLOPS", "TFLOPS", all_results, backends, [](const BenchResult& r) {
    return std::make_tuple(ComputeTFLOPS(r.eff_gemm1_flops, r.gemm1_ms), ComputeTFLOPS(r.eff_gemm2_flops, r.gemm2_ms),
                           ComputeTFLOPS(r.eff_total_flops, r.total_ms));
  });
}

void PrintRealTFLOPS(const std::string& config_name, const std::vector<std::vector<BenchResult>>& all_results,
                     const std::vector<BackendSpec>& backends) {
  PrintPerfTable(config_name, "Real TFLOPS", "TFLOPS", all_results, backends, [](const BenchResult& r) {
    return std::make_tuple(ComputeTFLOPS(r.real_gemm1_flops, r.gemm1_ms), ComputeTFLOPS(r.real_gemm2_flops, r.gemm2_ms),
                           ComputeTFLOPS(r.real_total_flops, r.total_ms));
  });
}

void PrintEffectiveBandwidth(const std::string& config_name, const std::vector<std::vector<BenchResult>>& all_results,
                             const std::vector<BackendSpec>& backends) {
  PrintPerfTable(config_name, "Effective Bandwidth", "GB/s", all_results, backends, [](const BenchResult& r) {
    return std::make_tuple(ComputeBandwidthGBps(r.eff_gemm1_bytes, r.gemm1_ms),
                           ComputeBandwidthGBps(r.eff_gemm2_bytes, r.gemm2_ms),
                           ComputeBandwidthGBps(r.eff_total_bytes, r.total_ms));
  });
}

void PrintRealBandwidth(const std::string& config_name, const std::vector<std::vector<BenchResult>>& all_results,
                        const std::vector<BackendSpec>& backends) {
  PrintPerfTable(config_name, "Real Bandwidth", "GB/s", all_results, backends, [](const BenchResult& r) {
    return std::make_tuple(ComputeBandwidthGBps(r.real_gemm1_bytes, r.gemm1_ms),
                           ComputeBandwidthGBps(r.real_gemm2_bytes, r.gemm2_ms),
                           ComputeBandwidthGBps(r.real_total_bytes, r.total_ms));
  });
}

// ===========================================================================
// Print expert routing summary: hits, padding, waste rate per token size.
// One sub-table per backend — block_size / real_pairs / waste are
// backend-dependent (each backend selects its own moe block size), so they
// must be shown for every backend that ran. Routing columns (total_pairs,
// experts_accessed) are backend-independent and repeat identically across
// sub-tables. Mirrors PrintHummingNonGemmBreakdown's per-backend structure.
// ===========================================================================
void PrintExpertRoutingSummary(const std::string& config_name, const Fp4MoeTestConfig& cfg,
                               const std::vector<std::vector<BenchResult>>& all_results,
                               const std::vector<BackendSpec>& backends) {
  for (size_t b = 0; b < backends.size(); ++b) {
    if (b >= all_results.size()) continue;

    std::cout << "\n=== " << config_name << " Expert Routing Summary — " << backends[b].name << " ===" << std::endl;
    std::cout << "  " << std::setw(10) << "tokens" << std::setw(6) << "topk" << std::setw(14) << "total_pairs"
              << std::setw(20) << "experts_accessed" << std::setw(16) << "experts_total" << std::setw(10) << "blk_sz1"
              << std::setw(10) << "blk_sz2" << std::setw(18) << "real_pairs_g1" << std::setw(18) << "real_pairs_g2"
              << std::setw(12) << "waste_g1%" << std::setw(12) << "waste_g2%" << std::endl;

    for (size_t i = 0; i < GetTokenSizes().size(); ++i) {
      if (i >= all_results[b].size()) break;
      const auto& r = all_results[b][i];
      std::cout << "  " << std::setw(10) << GetTokenSizes()[i] << std::setw(6) << cfg.topk() << std::setw(14)
                << r.total_pairs << std::setw(20) << r.experts_accessed << std::setw(16) << ComputeExpertsPerRank(cfg)
                << std::setw(10) << r.moe_block_size_1 << std::setw(10) << r.moe_block_size_2 << std::setw(18)
                << r.real_pairs_g1 << std::setw(18) << r.real_pairs_g2 << std::fixed << std::setprecision(1)
                << std::setw(11) << (r.waste_rate_g1 * 100.0) << "%" << std::setw(11) << (r.waste_rate_g2 * 100.0)
                << "%" << std::endl;
    }
  }
}

// ===========================================================================
// Print JIT-compiled kernel table (humming only).
// Lists all kernel variants compiled by HummingJitCompiler for this process.
// Called once per config, after the performance tables.
// ===========================================================================
void PrintJitKernelTable(const std::string& config_name, const Fp4MoeTestConfig& cfg) {
  const auto* table = HummingFusedWfp4A16MoeLayer::GetJitKernelTable();
  if (table == nullptr || table->empty()) {
    std::cout << "\n=== " << config_name << " JIT Kernel Table: EMPTY (no JIT compiler) ===" << std::endl;
    return;
  }

  // Problem shape header
  size_t E = ComputeExpertsPerRank(cfg);
  size_t N = ComputeInterPerRank(cfg);
  size_t K = cfg.hidden_size();
  std::cout << "\n=== " << config_name << " Problem Shapes ===" << std::endl;
  std::cout << "  experts_per_rank=" << E << "  topk=" << cfg.topk() << "  hidden=" << K << "  inter_per_rank=" << N
            << std::endl;
  std::cout << "  GEMM1: N=" << N * 2 << "  K=" << K << std::endl;
  std::cout << "  GEMM2: N=" << K << "  K=" << N << std::endl;

  std::cout << "\n=== " << config_name << " JIT Kernel Table ===" << std::endl;
  // Header
  std::cout << " " << std::setw(6) << "dtype" << std::setw(8) << "N" << std::setw(8) << "K" << std::setw(16)
            << "blkMxNxK" << std::setw(6) << "stg" << std::setw(6) << "cta" << std::setw(14) << "TMA(ABCS)"
            << std::setw(6) << "SK" << std::setw(6) << "WS" << std::setw(6) << "E8M0" << std::setw(22)
            << "shapeM[min..max]" << std::endl;

  // Sort by (compute_dtype, size_n) for readable output
  std::vector<llm_kernels::nvidia::humming::KernelMeta> sorted_table(table->begin(), table->end());
  std::sort(sorted_table.begin(), sorted_table.end(),
            [](const llm_kernels::nvidia::humming::KernelMeta& a, const llm_kernels::nvidia::humming::KernelMeta& b) {
              if (a.compute_dtype != b.compute_dtype) return a.compute_dtype < b.compute_dtype;
              return a.size_n < b.size_n;
            });

  for (const auto& m : sorted_table) {
    std::string dtype_str = (m.compute_dtype == llm_kernels::nvidia::humming::ComputeDtype::FP8) ? "FP8" : "BF16";
    std::string blk_str = std::to_string(m.block_m) + "x" + std::to_string(m.block_n) + "x" + std::to_string(m.block_k);
    std::string tma_str = std::string(m.use_tma_a ? "1" : "0") + std::string(m.use_tma_b ? "1" : "0") +
                          std::string(m.use_tma_c ? "1" : "0") + std::string(m.use_tma_bs ? "1" : "0");
    std::string shape_str = "[" + std::to_string(m.min_shape_m) + ".." + std::to_string(m.max_shape_m) + "]";

    std::cout << " " << std::setw(6) << dtype_str << std::setw(8) << m.size_n << std::setw(8) << m.size_k
              << std::setw(16) << blk_str << std::setw(6) << m.num_stages << std::setw(6) << m.num_ctas_per_sm
              << std::setw(14) << tma_str << std::setw(6) << (m.use_stream_k ? "Y" : "N") << std::setw(6)
              << (m.use_warp_spec ? "Y" : "N") << std::setw(6) << (m.use_fused_e8m0_scale ? "Y" : "N") << std::setw(22)
              << shape_str << std::endl;
  }
}

// ===========================================================================
// Print selected kernel schedule (humming only).
// For each token size, shows the actual KernelMeta selected at runtime for
// GEMM1 and GEMM2 by SelectGemmMeta (profiled right-align or heuristic fallback).
// ===========================================================================
void PrintSelectedKernels(const std::string& config_name, const std::vector<std::vector<BenchResult>>& all_results,
                          const std::vector<BackendSpec>& backends) {
  for (size_t b = 0; b < backends.size(); ++b) {
    if (!backends[b].is_humming) continue;
    if (b >= all_results.size()) continue;

    std::cout << "\n=== " << config_name << " Selected Kernels — " << backends[b].name << " ===" << std::endl;
    std::cout << " " << std::setw(10) << "tokens" << std::setw(36) << "GEMM1" << std::setw(36) << "GEMM2" << std::endl;

    for (size_t i = 0; i < GetTokenSizes().size(); ++i) {
      if (i >= all_results[b].size()) break;
      const auto& r = all_results[b][i];

      auto format_meta = [](const llm_kernels::nvidia::humming::KernelMeta* m) -> std::string {
        if (m == nullptr) return "nullptr(heuristic)";
        return llm_kernels::nvidia::humming::FormatKernelMeta(*m);
      };

      std::cout << " " << std::setw(10) << GetTokenSizes()[i] << std::setw(36)
                << format_meta(r.has_gemm1_meta ? &r.gemm1_meta : nullptr) << std::setw(36)
                << format_meta(r.has_gemm2_meta ? &r.gemm2_meta : nullptr) << std::endl;
    }
  }
}

// ===========================================================================
// Print non-GEMM kernel breakdown (humming only).
// For each token size, shows per-phase times: gating, align, quant, silu,
// mulsum, other (residual), nogemm (total-gemm1-gemm2), nogemm%.
// other = nogemm - (gating + align + quant + silu + mulsum) — captures memsets + gaps.
// ===========================================================================
void PrintHummingNonGemmBreakdown(const std::string& config_name,
                                  const std::vector<std::vector<BenchResult>>& all_results,
                                  const std::vector<BackendSpec>& backends) {
  for (size_t b = 0; b < backends.size(); ++b) {
    if (!backends[b].is_humming) continue;
    if (b >= all_results.size()) continue;

    std::cout << "\n=== " << config_name << " Non-GEMM Breakdown (ms) — " << backends[b].name << " ===" << std::endl;
    std::cout << " " << std::setw(10) << "tokens" << std::setw(10) << "gating" << std::setw(10) << "align"
              << std::setw(10) << "quant" << std::setw(10) << "silu" << std::setw(10) << "mulsum" << std::setw(10)
              << "other" << std::setw(10) << "nogemm" << std::setw(10) << "nogemm%" << std::endl;

    for (size_t i = 0; i < GetTokenSizes().size(); ++i) {
      if (i >= all_results[b].size()) break;
      const auto& r = all_results[b][i];

      float measured = r.gating_ms + r.align_ms + r.quant_ms + r.silu_ms + r.mulsum_ms;
      float other = r.nogemm_ms - measured;
      float nogemm_pct = (r.total_ms > 0.f) ? (r.nogemm_ms / r.total_ms * 100.f) : 0.f;

      std::cout << " " << std::setw(10) << GetTokenSizes()[i] << std::fixed << std::setprecision(4) << std::setw(10)
                << r.gating_ms << std::setw(10) << r.align_ms << std::setw(10) << r.quant_ms << std::setw(10)
                << r.silu_ms << std::setw(10) << r.mulsum_ms << std::setw(10) << other << std::setw(10) << r.nogemm_ms
                << std::setw(9) << std::setprecision(1) << nogemm_pct << "%" << std::endl;
    }
  }
}

// ===========================================================================
// CSV output: writes performance tables to ./{test_target}/{config}_*.csv
// Enabled when FP4_PERF_CSV env var is set to "1".
//
// Writes 5 standard CSV files (one per table type), each with a uniform schema
// so pandas/Excel can read them directly. Composite "g1/g2/total" cells are
// split into separate columns for easy analysis.
//
// Files (per config):
//   {config}_latency.csv    — per (tokens, backend) latency breakdown
//   {config}_routing.csv    — per (tokens, backend) expert routing + padding
//   {config}_tflops.csv      — per (tokens, backend) effective + real TFLOPS
//   {config}_bandwidth.csv   — per (tokens, backend) effective + real bandwidth
//   {config}_nongemm.csv     — per (tokens, backend) humming non-GEMM breakdown
// ===========================================================================

// Write all CSV files for one config. Called by the driver after terminal printing.
// Output directory: ./wfp4_moe_layers_test/ (created if missing).
void WritePerfCsv(const std::string& config_name, const Fp4MoeTestConfig& cfg,
                  const std::vector<std::vector<BenchResult>>& all_results, const std::vector<BackendSpec>& backends) {
  const char* env = std::getenv("FP4_PERF_CSV");
  if (env == nullptr || std::string(env) != "1") return;

  // Create output directory (ignore EEXIST — dir may already exist)
  const std::string dir = "wfp4_moe_layers_test";
  mkdir(dir.c_str(), 0755);

  auto path = [&](const std::string& suffix) -> std::string { return dir + "/" + config_name + suffix; };

  // --- latency.csv ---
  {
    std::ofstream f(path("_latency.csv"));
    f << "tokens,backend,gemm1_ms,gemm2_ms,nogemm_ms,total_ms\n";
    for (size_t i = 0; i < GetTokenSizes().size(); ++i) {
      for (size_t b = 0; b < all_results.size(); ++b) {
        if (i >= all_results[b].size()) continue;
        const auto& r = all_results[b][i];
        f << GetTokenSizes()[i] << "," << backends[b].name << "," << std::fixed << std::setprecision(4) << r.gemm1_ms
          << "," << r.gemm2_ms << "," << r.nogemm_ms << "," << r.total_ms << "\n";
      }
    }
  }

  // --- routing.csv (all backends — block_size/padding is per-backend) ---
  {
    std::ofstream f(path("_routing.csv"));
    f << "tokens,backend,topk,total_pairs,experts_accessed,experts_total,blk_sz1,blk_sz2,"
         "real_pairs_g1,real_pairs_g2,waste_g1_pct,waste_g2_pct\n";
    size_t experts_total = ComputeExpertsPerRank(cfg);
    for (size_t i = 0; i < GetTokenSizes().size(); ++i) {
      for (size_t b = 0; b < all_results.size(); ++b) {
        if (i >= all_results[b].size()) continue;
        const auto& r = all_results[b][i];
        f << GetTokenSizes()[i] << "," << backends[b].name << ",6," << r.total_pairs << "," << r.experts_accessed << ","
          << experts_total << "," << r.moe_block_size_1 << "," << r.moe_block_size_2 << "," << r.real_pairs_g1 << ","
          << r.real_pairs_g2 << "," << std::fixed << std::setprecision(1) << (r.waste_rate_g1 * 100.0) << ","
          << (r.waste_rate_g2 * 100.0) << "\n";
      }
    }
  }

  // --- tflops.csv (effective + real merged) ---
  {
    std::ofstream f(path("_tflops.csv"));
    f << "tokens,backend,eff_gemm1,eff_gemm2,eff_total,real_gemm1,real_gemm2,real_total\n";
    for (size_t i = 0; i < GetTokenSizes().size(); ++i) {
      for (size_t b = 0; b < all_results.size(); ++b) {
        if (i >= all_results[b].size()) continue;
        const auto& r = all_results[b][i];
        f << GetTokenSizes()[i] << "," << backends[b].name << "," << std::fixed << std::setprecision(2)
          << ComputeTFLOPS(r.eff_gemm1_flops, r.gemm1_ms) << "," << ComputeTFLOPS(r.eff_gemm2_flops, r.gemm2_ms) << ","
          << ComputeTFLOPS(r.eff_total_flops, r.total_ms) << "," << ComputeTFLOPS(r.real_gemm1_flops, r.gemm1_ms) << ","
          << ComputeTFLOPS(r.real_gemm2_flops, r.gemm2_ms) << "," << ComputeTFLOPS(r.real_total_flops, r.total_ms)
          << "\n";
      }
    }
  }

  // --- bandwidth.csv (effective + real merged) ---
  {
    std::ofstream f(path("_bandwidth.csv"));
    f << "tokens,backend,eff_gemm1_gbps,eff_gemm2_gbps,eff_total_gbps,"
         "real_gemm1_gbps,real_gemm2_gbps,real_total_gbps\n";
    for (size_t i = 0; i < GetTokenSizes().size(); ++i) {
      for (size_t b = 0; b < all_results.size(); ++b) {
        if (i >= all_results[b].size()) continue;
        const auto& r = all_results[b][i];
        f << GetTokenSizes()[i] << "," << backends[b].name << "," << std::fixed << std::setprecision(1)
          << ComputeBandwidthGBps(r.eff_gemm1_bytes, r.gemm1_ms) << ","
          << ComputeBandwidthGBps(r.eff_gemm2_bytes, r.gemm2_ms) << ","
          << ComputeBandwidthGBps(r.eff_total_bytes, r.total_ms) << ","
          << ComputeBandwidthGBps(r.real_gemm1_bytes, r.gemm1_ms) << ","
          << ComputeBandwidthGBps(r.real_gemm2_bytes, r.gemm2_ms) << ","
          << ComputeBandwidthGBps(r.real_total_bytes, r.total_ms) << "\n";
      }
    }
  }

  // --- nongemm.csv (humming only: gating/align/quant/silu/mulsum/other/nogemm) ---
  {
    std::ofstream f(path("_nongemm.csv"));
    f << "tokens,backend,gating_ms,align_ms,quant_ms,silu_ms,mulsum_ms,other_ms,nogemm_ms,nogemm_pct\n";
    for (size_t b = 0; b < backends.size(); ++b) {
      if (!backends[b].is_humming) continue;
      if (b >= all_results.size()) continue;
      for (size_t i = 0; i < GetTokenSizes().size(); ++i) {
        if (i >= all_results[b].size()) break;
        const auto& r = all_results[b][i];
        float measured = r.gating_ms + r.align_ms + r.quant_ms + r.silu_ms + r.mulsum_ms;
        float other = r.nogemm_ms - measured;
        float nogemm_pct = (r.total_ms > 0.f) ? (r.nogemm_ms / r.total_ms * 100.f) : 0.f;
        f << GetTokenSizes()[i] << "," << backends[b].name << "," << std::fixed << std::setprecision(4) << r.gating_ms
          << "," << r.align_ms << "," << r.quant_ms << "," << r.silu_ms << "," << r.mulsum_ms << "," << other << ","
          << r.nogemm_ms << "," << std::setprecision(1) << nogemm_pct << "\n";
      }
    }
  }

  std::cout << "\n=== CSV files written to " << dir << "/" << config_name << "_*.csv ===" << std::endl;
}
}  // namespace ksana_llm
