/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "flash_sparse_mla_test_common.h"

#include <functional>
#include <iomanip>
#include <sstream>

namespace llm_kernels {
namespace nvidia {
namespace test {

struct MlaPerfRow {
  size_t batch_size;
  size_t seqlen_k;
  size_t seqlen_q;
  IndicesStrategy idx_strategy;
  float latency_ms;
};

inline double CalcMlaFlops(const ModelConfig& cfg, size_t batch_size, size_t seqlen_q, size_t num_heads_q,
                           size_t seqlen_k) {
  const size_t eff_topk = cfg.IsModel1() ? std::min(cfg.topk, seqlen_k) : cfg.topk;
  const size_t num_kv_tokens = eff_topk + (cfg.IsModel1() ? eff_topk : 0);
  return 2.0 * static_cast<double>(batch_size) * static_cast<double>(seqlen_q) * static_cast<double>(num_heads_q) *
         static_cast<double>(num_kv_tokens) * static_cast<double>(cfg.head_size_k + cfg.head_size_v);
}

inline double CalcMlaBytes(const ModelConfig& cfg, size_t batch_size, size_t seqlen_q, size_t num_heads_q,
                           size_t seqlen_k) {
  const size_t eff_topk = cfg.IsModel1() ? std::min(cfg.topk, seqlen_k) : cfg.topk;
  const size_t num_kv_tokens = eff_topk + (cfg.IsModel1() ? eff_topk : 0);
  const double q_bytes = static_cast<double>(batch_size) * static_cast<double>(seqlen_q) *
                         static_cast<double>(num_heads_q) * static_cast<double>(cfg.head_size_k) * 1.0;
  const double kv_bytes = static_cast<double>(batch_size) * static_cast<double>(seqlen_q) *
                          static_cast<double>(num_kv_tokens) * static_cast<double>(cfg.head_size_k) * 1.0;
  const double o_bytes = static_cast<double>(batch_size) * static_cast<double>(seqlen_q) *
                         static_cast<double>(num_heads_q) * static_cast<double>(cfg.head_size_v) * 2.0;
  return q_bytes + kv_bytes + o_bytes;
}

inline void PrintMlaLatencyTable(const ModelConfig& cfg, size_t h_q,
                                 const std::vector<std::vector<MlaPerfRow>>& all_results,
                                 const std::vector<MlaBackendSpec>& backends,
                                 const std::vector<IndicesStrategy>& idx_strategies) {
  const bool both_selected = backends.size() == 2;
  size_t swapab_b = backends.size();
  for (size_t b = 0; b < backends.size(); ++b) {
    if (backends[b].backend == MlaBackend::kFp8E4M3SwapAB) {
      swapab_b = b;
      break;
    }
  }
  const bool has_swapab = swapab_b < backends.size();
  const size_t n_idx = idx_strategies.size();
  constexpr int kBatchW = 8;
  constexpr int kSkW = 10;
  constexpr int kSqW = 6;
  constexpr int kLatW = 24;
  constexpr int kSpeedupW = 22;
  constexpr int kPerfW = 50;

  std::cout << "\n=== " << cfg.name << " h_q=" << h_q << " Latency (ms) ===" << std::endl;
  std::cout << " " << std::setw(kBatchW) << "batch" << std::setw(kSkW) << "s_k" << std::setw(kSqW) << "s_q";
  for (const auto& b : backends) {
    std::cout << std::setw(kLatW) << b.name;
  }
  if (both_selected) {
    std::cout << std::setw(kSpeedupW) << "speedup(e4m3/ds_mla)";
  }
  if (has_swapab) {
    std::cout << std::setw(kPerfW) << "fp8_e4m3_swapab_perf";
  }
  std::cout << std::endl;

  if (all_results.empty() || n_idx == 0) return;
  size_t num_combos = 0;
  for (const auto& v : all_results) {
    num_combos = std::max(num_combos, v.size());
  }
  const size_t num_groups = num_combos / n_idx;

  for (size_t g = 0; g < num_groups; ++g) {
    const size_t base = g * n_idx;
    const MlaPerfRow* label_row = nullptr;
    for (const auto& v : all_results) {
      if (base < v.size()) {
        label_row = &v[base];
        break;
      }
    }
    std::cout << " " << std::setw(kBatchW) << label_row->batch_size << std::setw(kSkW) << label_row->seqlen_k
              << std::setw(kSqW) << label_row->seqlen_q;
    for (size_t b = 0; b < backends.size(); ++b) {
      if (base + n_idx <= all_results[b].size()) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);
        for (size_t k = 0; k < n_idx; ++k) {
          if (k > 0) oss << "|";
          oss << all_results[b][base + k].latency_ms;
        }
        std::cout << std::setw(kLatW) << oss.str();
      } else {
        std::cout << std::setw(kLatW) << "N/A";
      }
    }
    if (both_selected) {
      if (base + n_idx <= all_results[0].size() && base + n_idx <= all_results[1].size()) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        for (size_t k = 0; k < n_idx; ++k) {
          if (k > 0) oss << "|";
          float d = all_results[0][base + k].latency_ms;
          float e = all_results[1][base + k].latency_ms;
          if (d > 0.f && e > 0.f) {
            oss << (d / e) << "x";
          } else {
            oss << "N/A";
          }
        }
        std::cout << std::setw(kSpeedupW) << oss.str();
      } else {
        std::cout << std::setw(kSpeedupW) << "N/A";
      }
    }
    if (has_swapab) {
      if (base + n_idx <= all_results[swapab_b].size()) {
        const double flops = CalcMlaFlops(cfg, label_row->batch_size, label_row->seqlen_q, h_q, label_row->seqlen_k);
        const double bytes = CalcMlaBytes(cfg, label_row->batch_size, label_row->seqlen_q, h_q, label_row->seqlen_k);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        for (size_t k = 0; k < n_idx; ++k) {
          if (k > 0) oss << "|";
          const float latency_ms = all_results[swapab_b][base + k].latency_ms;
          if (latency_ms > 0.f) {
            const double bw_gbps = bytes / (latency_ms * 1e-3) / 1e9;
            const double tflops = flops / (latency_ms * 1e-3) / 1e12;
            oss << bw_gbps << "/" << tflops;
          } else {
            oss << "N/A";
          }
        }
        std::cout << std::setw(kPerfW) << oss.str();
      } else {
        std::cout << std::setw(kPerfW) << "N/A";
      }
    }
    std::cout << std::endl;
  }
}

inline void MeasureAndSave(LlamaNvidiaFlashSparseMlaTestSuit& fx, const ModelConfig& cfg, size_t batch_size,
                           size_t seqlen_q, size_t seqlen_k, size_t num_heads_q, IndicesStrategy idx_strategy,
                           const MlaTestRange& range, size_t backend_idx,
                           std::vector<std::vector<MlaPerfRow>>& all_results) {
  const auto& spec = fx.selected_backends[backend_idx];
  int* topk_length_ptr = fx.GetTopkLengthPtr();
  float* attn_sink_ptr = fx.GetAttnSinkPtr();
  int* extra_indices_ptr = fx.GetExtraIndicesPtr();
  int* extra_topk_length_ptr = fx.GetExtraTopkLengthPtr();
  int cur_extra_num_blocks = cfg.IsModel1() ? static_cast<int>(fx.extra_num_blocks) : 0;
  int cur_extra_block_size = cfg.IsModel1() ? static_cast<int>(kBlockSize) : 0;
  int cur_extra_topk = cfg.IsModel1() ? static_cast<int>(cfg.topk) : 0;
  int cur_extra_kcache_batch_stride = cfg.IsModel1() ? static_cast<int>(cfg.kcache_batch_stride) : -1;
  int cur_extra_kcache_batch_stride_fp8 = cfg.IsModel1() ? static_cast<int>(fx.kcache_batch_stride_fp8) : -1;

  using T = __nv_bfloat16;

  const bool need_expand_bf16 = (spec.backend == MlaBackend::kFp8DsMla) && (num_heads_q < 64);
  torch::Tensor d_expanded_q;
  torch::Tensor d_expanded_out;
  if (need_expand_bf16) {
    d_expanded_q = torch::empty(
        {static_cast<int64_t>(batch_size), static_cast<int64_t>(seqlen_q), 64, static_cast<int64_t>(cfg.head_size_k)},
        torch::TensorOptions().device(torch::kCUDA, fx.device).dtype(torch::kBFloat16));
    d_expanded_out = torch::empty(
        {static_cast<int64_t>(batch_size), static_cast<int64_t>(seqlen_q), 64, static_cast<int64_t>(cfg.head_size_v)},
        torch::TensorOptions().device(torch::kCUDA, fx.device).dtype(torch::kBFloat16));
  }

  auto invoke_kernel_bf16 = [&](const bool sched_meta_initialized) {
    if (need_expand_bf16) {
      cudaMemcpy2DAsync(reinterpret_cast<__nv_bfloat16*>(d_expanded_q.data_ptr()), 64 * cfg.head_size_k * sizeof(T),
                        reinterpret_cast<__nv_bfloat16*>(fx.d_q.data_ptr()), num_heads_q * cfg.head_size_k * sizeof(T),
                        num_heads_q * cfg.head_size_k * sizeof(T), batch_size * seqlen_q, cudaMemcpyDeviceToDevice,
                        fx.stream);
    }
    InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8DsMla>(
        reinterpret_cast<__nv_bfloat16*>(need_expand_bf16 ? d_expanded_q.data_ptr() : fx.d_q.data_ptr()),
        reinterpret_cast<uint8_t*>(fx.d_kcache.data_ptr()), fx.d_indices.data_ptr<int>(),
        fx.d_tile_scheduler_metadata.data_ptr<int>(), fx.d_num_splits.data_ptr<int>(), sched_meta_initialized,
        batch_size, seqlen_q, need_expand_bf16 ? 64ul : num_heads_q, cfg.head_size_k, fx.num_blocks, kBlockSize,
        fx.num_heads_k, cfg.kcache_batch_stride, cfg.topk, cfg.head_size_v, fx.softmax_scale, fx.stream,
        fx.d_workspace.data_ptr<float>(),
        reinterpret_cast<T*>(need_expand_bf16 ? d_expanded_out.data_ptr() : fx.d_out.data_ptr()), topk_length_ptr,
        attn_sink_ptr, fx.GetExtraKcachePtr(), extra_indices_ptr, extra_topk_length_ptr, cur_extra_num_blocks,
        cur_extra_block_size, cur_extra_topk, cur_extra_kcache_batch_stride);
    if (need_expand_bf16) {
      cudaMemcpy2DAsync(
          reinterpret_cast<__nv_bfloat16*>(fx.d_out.data_ptr()), num_heads_q * cfg.head_size_v * sizeof(T),
          reinterpret_cast<__nv_bfloat16*>(d_expanded_out.data_ptr()), 64 * cfg.head_size_v * sizeof(T),
          num_heads_q * cfg.head_size_v * sizeof(T), batch_size * seqlen_q, cudaMemcpyDeviceToDevice, fx.stream);
    }
  };

  auto invoke_kernel_fp8 = [&](const bool sched_meta_initialized) {
    InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8E4M3>(
        reinterpret_cast<uint8_t*>(fx.d_q_fp8.data_ptr()), reinterpret_cast<uint8_t*>(fx.d_kcache_fp8.data_ptr()),
        fx.d_indices.data_ptr<int>(), fx.d_tile_scheduler_metadata.data_ptr<int>(), fx.d_num_splits.data_ptr<int>(),
        sched_meta_initialized, batch_size, seqlen_q, num_heads_q, cfg.head_size_k, fx.num_blocks, kBlockSize,
        fx.num_heads_k, fx.kcache_batch_stride_fp8, cfg.topk, cfg.head_size_v, fx.softmax_scale, fx.stream,
        fx.d_workspace.data_ptr<float>(), reinterpret_cast<T*>(fx.d_out.data_ptr()), topk_length_ptr, attn_sink_ptr,
        fx.GetExtraKcacheFp8Ptr(), extra_indices_ptr, extra_topk_length_ptr, cur_extra_num_blocks, cur_extra_block_size,
        cur_extra_topk, cur_extra_kcache_batch_stride_fp8);
  };

  auto invoke_kernel_fp8_swapab = [&](const bool sched_meta_initialized) {
    InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8E4M3, /*USE_SWAP_AB=*/true>(
        reinterpret_cast<uint8_t*>(fx.d_q_fp8.data_ptr()), reinterpret_cast<uint8_t*>(fx.d_kcache_fp8.data_ptr()),
        fx.d_indices.data_ptr<int>(), fx.d_tile_scheduler_metadata.data_ptr<int>(), fx.d_num_splits.data_ptr<int>(),
        sched_meta_initialized, batch_size, seqlen_q, num_heads_q, cfg.head_size_k, fx.num_blocks, kBlockSize,
        fx.num_heads_k, fx.kcache_batch_stride_fp8, cfg.topk, cfg.head_size_v, fx.softmax_scale, fx.stream,
        fx.d_workspace.data_ptr<float>(), reinterpret_cast<T*>(fx.d_out.data_ptr()), topk_length_ptr, attn_sink_ptr,
        fx.GetExtraKcacheFp8Ptr(), extra_indices_ptr, extra_topk_length_ptr, cur_extra_num_blocks, cur_extra_block_size,
        cur_extra_topk, cur_extra_kcache_batch_stride_fp8);
  };

  if (spec.backend == MlaBackend::kFp8DsMla) {
    invoke_kernel_bf16(false);
    auto cuda_run = [&]() { invoke_kernel_bf16(true); };
    const float elapsed_ms = MeasureCudaExecutionTime(cuda_run, fx.stream, range.warmups, range.iterations);
    all_results[backend_idx].push_back({batch_size, seqlen_k, seqlen_q, idx_strategy, elapsed_ms});
  } else if (spec.backend == MlaBackend::kFp8E4M3) {
    invoke_kernel_fp8(false);
    auto cuda_run = [&]() { invoke_kernel_fp8(true); };
    const float elapsed_ms = MeasureCudaExecutionTime(cuda_run, fx.stream, range.warmups, range.iterations);
    all_results[backend_idx].push_back({batch_size, seqlen_k, seqlen_q, idx_strategy, elapsed_ms});
  } else if (spec.backend == MlaBackend::kFp8E4M3SwapAB) {
    if (num_heads_q != 16) {
      return;
    }
    invoke_kernel_fp8_swapab(false);
    auto cuda_run = [&]() { invoke_kernel_fp8_swapab(true); };
    const float elapsed_ms = MeasureCudaExecutionTime(cuda_run, fx.stream, range.warmups, range.iterations);
    all_results[backend_idx].push_back({batch_size, seqlen_k, seqlen_q, idx_strategy, elapsed_ms});
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
