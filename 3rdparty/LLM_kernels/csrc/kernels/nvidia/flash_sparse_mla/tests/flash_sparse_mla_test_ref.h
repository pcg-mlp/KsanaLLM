/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "flash_sparse_mla_test_common.h"

#include <cstring>

namespace llm_kernels {
namespace nvidia {
namespace test {

struct GoldenResult {
  torch::Tensor output;
  torch::Tensor lse;
};

inline GoldenResult ComputeGoldenReference(const torch::Tensor& q_fp8, const torch::Tensor& kcache_fp8,
                                           const torch::Tensor& indices, const torch::Tensor& topk_length,
                                           const torch::Tensor& attn_sink, const torch::Tensor& extra_kcache,
                                           const torch::Tensor& extra_indices, const torch::Tensor& extra_topk_length,
                                           int batch_size, int seqlen_q, int num_heads_q, int head_size_k,
                                           int head_size_v, int topk, int block_size, int num_blocks,
                                           int kcache_batch_stride, int extra_num_blocks, int extra_topk,
                                           int extra_kcache_batch_stride, float sm_scale, int device,
                                           cudaStream_t stream) {
  const int d_qk = head_size_k;
  const int d_v = head_size_v;
  const int per_token_stride = kcache_batch_stride / block_size;

  auto q = q_fp8.to(torch::kFloat32);

  auto kcache_data = kcache_fp8.view({num_blocks, block_size, per_token_stride}).narrow(2, 0, d_qk).contiguous();
  auto kcache_flat = kcache_data.view({-1, d_qk});

  auto indices_3d = indices.view({batch_size, seqlen_q, topk});
  auto idx_safe = torch::clamp_min(indices_3d, 0);
  auto gathered_kv = kcache_flat.index_select(0, idx_safe.reshape({-1})).reshape({batch_size, seqlen_q, topk, d_qk});
  auto invalid_mask = (indices_3d == -1);

  if (topk_length.defined() && topk_length.numel() > 0) {
    auto ar = torch::arange(topk, indices.options()).view({1, 1, topk}).expand({batch_size, seqlen_q, topk});
    invalid_mask = invalid_mask | (ar >= topk_length.view({batch_size, seqlen_q, 1}));
  }

  int topk_total = topk;
  if (extra_kcache.defined() && extra_kcache.numel() > 0) {
    int extra_per_token = extra_kcache_batch_stride / block_size;
    auto extra_data = extra_kcache.view({extra_num_blocks, block_size, extra_per_token})
                          .narrow(2, 0, d_qk)
                          .contiguous()
                          .view({-1, d_qk});
    auto extra_indices_3d = extra_indices.view({batch_size, seqlen_q, extra_topk});
    auto extra_gathered = extra_data.index_select(0, torch::clamp_min(extra_indices_3d, 0).reshape({-1}))
                              .reshape({batch_size, seqlen_q, extra_topk, d_qk});
    auto extra_mask = (extra_indices_3d == -1);
    if (extra_topk_length.defined() && extra_topk_length.numel() > 0) {
      auto ear = torch::arange(extra_topk, extra_indices.options())
                     .view({1, 1, extra_topk})
                     .expand({batch_size, seqlen_q, extra_topk});
      extra_mask = extra_mask | (ear >= extra_topk_length.view({batch_size, seqlen_q, 1}));
    }
    gathered_kv = torch::cat({gathered_kv, extra_gathered}, 2);
    invalid_mask = torch::cat({invalid_mask, extra_mask}, 2);
    topk_total = topk + extra_topk;
  }

  gathered_kv = gathered_kv.to(torch::kFloat32).nan_to_num_(0.0);

  auto q_flat = q.view({batch_size * seqlen_q, num_heads_q, d_qk});
  auto kv_flat = gathered_kv.view({batch_size * seqlen_q, topk_total, d_qk});
  auto attn = q_flat.matmul(kv_flat.transpose(1, 2)) * sm_scale;

  auto mask_flat =
      invalid_mask.view({batch_size * seqlen_q, 1, -1}).expand({batch_size * seqlen_q, num_heads_q, topk_total});
  attn.masked_fill_(mask_flat, -INFINITY);

  auto lse = torch::logsumexp(attn, -1);
  auto prob = (attn - lse.unsqueeze(-1)).exp();

  auto output = prob.matmul(kv_flat.narrow(2, 0, d_v));
  output = output.view({batch_size, seqlen_q, num_heads_q, d_v});
  lse = lse.view({batch_size, seqlen_q, num_heads_q});

  if (attn_sink.defined() && attn_sink.numel() > 0) {
    auto attn_sink_narrowed = attn_sink.narrow(0, 0, num_heads_q).view({1, 1, num_heads_q});
    auto factor = (1.0 / (1.0 + (attn_sink_narrowed - lse).exp())).unsqueeze(-1);
    output = output * factor;
  }

  auto lonely = (lse == -INFINITY);
  output.masked_fill_(lonely.unsqueeze(-1).expand({batch_size, seqlen_q, num_heads_q, d_v}), 0.0);
  lse.masked_fill_(lonely, INFINITY);

  return {output.to(torch::kBFloat16), lse.transpose(1, 2)};
}

struct AccuracyResult {
  bool passed;
  float max_abs_err;
  float ref_abs;
  bool has_nan;
};

inline bool IsBf16NaN(__nv_bfloat16 x) {
  uint16_t u16 = 0;
  static_assert(sizeof(uint16_t) == sizeof(__nv_bfloat16));
  std::memcpy(&u16, &x, sizeof(u16));
  constexpr uint16_t kExpMask = 0x7F80u;
  constexpr uint16_t kMantMask = 0x007Fu;
  uint16_t exp = u16 & kExpMask;
  uint16_t mant = u16 & kMantMask;
  return (exp == kExpMask) && (mant != 0);
}

inline AccuracyResult VerifyAccuracy(const LlamaNvidiaFlashSparseMlaTestSuit& fx, const ModelConfig& cfg,
                                     size_t batch_size, size_t seqlen_q, size_t num_heads_q, size_t seqlen_k,
                                     IndicesStrategy idx_strategy, const torch::Tensor& d_out) {
  auto golden = ComputeGoldenReference(
      fx.d_q_fp8, fx.d_kcache_fp8, fx.d_indices, cfg.IsModel1() ? fx.d_topk_length : torch::Tensor(),
      cfg.IsModel1() ? fx.d_attn_sink : torch::Tensor(), cfg.IsModel1() ? fx.d_extra_kcache_fp8 : torch::Tensor(),
      cfg.IsModel1() ? fx.d_extra_indices : torch::Tensor(), cfg.IsModel1() ? fx.d_extra_topk_length : torch::Tensor(),
      static_cast<int>(batch_size), static_cast<int>(seqlen_q), static_cast<int>(num_heads_q),
      static_cast<int>(cfg.head_size_k), static_cast<int>(cfg.head_size_v), static_cast<int>(cfg.topk),
      static_cast<int>(kBlockSize), static_cast<int>(fx.num_blocks), static_cast<int>(fx.kcache_batch_stride_fp8),
      cfg.IsModel1() ? static_cast<int>(fx.extra_num_blocks) : 0, cfg.IsModel1() ? static_cast<int>(cfg.topk) : 0,
      cfg.IsModel1() ? static_cast<int>(fx.kcache_batch_stride_fp8) : 0, fx.softmax_scale, fx.device, fx.stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(fx.stream));

  auto kernel_out_f32 = d_out.to(torch::kFloat32);
  auto golden_f32 = golden.output.to(torch::kFloat32);
  auto max_abs_err = (kernel_out_f32 - golden_f32).abs().max().item<float>();
  auto ref_abs = golden_f32.abs().max().item<float>();
  constexpr float atol = 0.5f;
  constexpr float rtol = 0.05f;
  bool passed = max_abs_err <= atol || max_abs_err <= rtol * ref_abs;

  auto h_out = d_out.to(torch::kCPU);
  bool has_nan = false;
  for (size_t i = 0; i < batch_size * seqlen_q * num_heads_q * cfg.head_size_v; i++) {
    if (IsBf16NaN(reinterpret_cast<const __nv_bfloat16*>(h_out.data_ptr())[i])) {
      has_nan = true;
      break;
    }
  }
  passed = passed && !has_nan;

  return {passed, max_abs_err, ref_abs, has_nan};
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
