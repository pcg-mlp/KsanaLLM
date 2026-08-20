/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "kernels/kunlun/basic_kernel_wrapper.h"

namespace ksana_llm {

// Template naming convention: see KUNLUN_TEMPLATE_NAMING.md for the shared glossary
// (T, TID, ...). This file additionally uses:
//   TS — score/scale dtype (softmax-topk output)
// (The .cpp also uses TI/TO and TWEIGHT internally; see the glossary.)

// ── MoE element-wise ops ─────────────────────────────────────────────────────

// sigmoid: y[i] = 1 / (1 + exp(-x[i]))
template <typename T>
int XpuSigmoid(xdnn::Context* ctx, const T* x, T* y, int64_t len);

// element-wise mul: z[i] = x[i] * y[i]
template <typename T>
int XpuMul(xdnn::Context* ctx, const T* x, const T* y, T* z, int64_t len);

// broadcast_mul: z = x * y with numpy-style broadcasting
// x_shape and y_shape define the broadcast semantics (e.g., [m,n] * [m,1])
template <typename T>
int XpuBroadcastMul(xdnn::Context* ctx, const T* x, const T* y, T* z, const std::vector<int64_t>& x_shape,
                    const std::vector<int64_t>& y_shape);

// ── MoE routing / dispatch ───────────────────────────────────────────────────

// moe_softmax_topk_norm: 路由 + TopK + 归一化 (Qwen3-MoE 等需要 norm)
// gating [m, expert_num] → topk_score [m, topk], topk_idx [m, topk],
// block_statistic [block_num * expert_num]
template <typename T, typename TS, typename TID>
int XpuMoeSoftmaxTopkNorm(xdnn::Context* ctx, const T* gating, TS* topk_score, TID* topk_idx, int32_t* block_statistic,
                          int64_t m, int64_t expert_num, int64_t topk, int64_t block_num, bool stable = true);

// moe_softmax_topk: 路由 + TopK 不带归一化 (Qwen2-MoE 等不需要 norm)
template <typename T, typename TS, typename TID>
int XpuMoeSoftmaxTopk(xdnn::Context* ctx, const T* gating, TS* topk_score, TID* topk_idx, int32_t* block_statistic,
                      int64_t m, int64_t expert_num, int64_t topk, int64_t block_num);

// gen_block_statistic: 生成block统计信息
// topk_ids [m, topk] → block_statistic [block_num * expert_num]
template <typename TID>
int XpuGenBlockStatistic(xdnn::Context* ctx, const TID* topk_ids, int32_t* block_statistic, int64_t seq_num,
                         int64_t expert_num, int64_t topk, int64_t block_num);

// moe_ffn_pre_sorted: token排序/分发
// x [m, n], index [m, topk] → sorted_x [m*topk, n], moe_index [m*topk],
// expert_m [expert_num], sorted_tokens_num_lod [expert_num+1]
template <typename T, typename TID>
int XpuMoeFfnPreSorted(xdnn::Context* ctx, const T* x, const TID* index, const int32_t* block_statistic, T* sorted_x,
                       TID* moe_index, TID* expert_m, TID* sorted_tokens_num_lod, int64_t m, int64_t n,
                       int64_t expert_num, int64_t topk, int64_t block_num);

// swiglu: SiLU(gate) * up 融合激活
// x [m, 2*n] → z [m, n]  (turn=true 表示 x 的 layout 是 [gate|up] interleaved)
template <typename T>
int XpuSwiglu(xdnn::Context* ctx, const T* x, T* z, int64_t m, int64_t n);

// fast_swiglu: 使用 xdnn::fast_swiglu 优化版本 (对应 XSGL_FAST_SWIGLU=1)
// x [m, 2*n] → z [m, n]
template <typename T>
int XpuFastSwiglu(xdnn::Context* ctx, const T* x, T* z, int64_t m, int64_t n);

// fast_sigmoid: 快速 Sigmoid (对应 xsglang: sigmoid_sdnn, 用于 MoE gating scoring_func="sigmoid")
// x [len] → y [len]
template <typename T>
int XpuFastSigmoid(xdnn::Context* ctx, const T* x, T* y, int64_t len);

// moe_fused_gate_kimi_k2: Sigmoid + TopK 融合 gate (Kimi-K2.6 专用)
// 对应 xsglang: infer_ops::moe_fused_gate_kimi_k2
// flow: input + bias → sigmoid → topk → normalize (optional)
// input [m, num_experts], bias [num_experts] → topk_score [m, topk], topk_idx [m, topk]
template <typename T, typename TS, typename TID>
int XpuMoeFusedGateKimiK2(xdnn::Context* ctx,
                          const T* input,  // [m, num_experts] gate logits
                          const T* bias,   // [num_experts] e_score_correction_bias (nullable)
                          TS* topk_score,  // [m, topk] output scores
                          TID* topk_idx,   // [m, topk] output indices
                          int64_t m, int64_t num_experts, int64_t topk, bool renormalize, float routed_scaling_factor,
                          int32_t* block_statistic,  // workspace [ncluster * num_experts] (nullable)
                          int64_t block_num);

// moe_sigmoid_topk_norm_fusion: Sigmoid + Group TopK + Norm (通用版本, 含 group 分组)
// 对应 xsglang: infer_ops::moe_sigmoid_topk_norm_fusion
template <typename T, typename TS, typename TID>
int XpuMoeSigmoidTopkNorm(xdnn::Context* ctx,
                          const T* input,            // [m, num_experts]
                          const float* bias,         // [num_experts] (nullable, fp32)
                          float scale,               // routed_scaling_factor
                          TS* topk_score,            // [m, topk]
                          TID* topk_idx,             // [m, topk]
                          int32_t* block_statistic,  // workspace (nullable)
                          int64_t m, int64_t num_experts, int64_t num_expert_group, int64_t topk_group, int64_t topk,
                          int64_t block_num);

// moe_ffn_post_fusion: 加权求和还原
// expert_hidden [m*topk, n], moe_index [m*topk], topk_score [m, topk] → y [m, n]
template <typename T, typename TID>
int XpuMoeFfnPostFusion(xdnn::Context* ctx, const T* expert_hidden, const TID* moe_index, const T* topk_score, T* y,
                        int64_t m, int64_t n, int64_t topk);

// moe_ffn_post_fusion with float topk_score (不要求 score 与 data 类型一致)
// 内部自动处理 float → T 的 cast（对齐 xsglang 的 klx_fused_moe 逻辑）
template <typename T, typename TID>
int XpuMoeFfnPostFusionF32Score(xdnn::Context* ctx, const T* expert_hidden, const TID* moe_index,
                                const float* topk_score, T* y, int64_t m, int64_t n, int64_t topk);

// ── xblas MoE FC wrapper ─────────────────────────────────────────────────────

// MoE FC fusion: 批量专家投影 (封装 xblas::moe_fc_fusion deprecated API)
// sorted_x [m*topk, k], weight [num_experts, n, k] (trans_b=true) → out [m*topk, n]
// sorted_tokens_ids = moe_index, sorted_tokens_num_lod 从 pre_sorted 获取
template <typename T, typename TID>
int XpuMoeFcFusion(xdnn::Context* ctx, const T* sorted_x, const T* weight, T* out, int64_t tokens, int64_t topk,
                   int64_t m_total, int64_t n, int64_t k, int64_t num_experts, const TID* sorted_tokens_ids,
                   const TID* sorted_tokens_num_lod);

// ── EP (Expert Parallel) MoE wrappers ───────────────────────────────────────

// moe_ep_ffn_pre_sorted: EP-aware token sorting (BF16 path)
// Same as XpuMoeFfnPreSorted but accepts ep_size/ep_rank for expert-parallel dispatch.
// Wraps infer_ops::moe_ep_ffn_pre_sorted.
template <typename T, typename TID>
int XpuMoeEpFfnPreSorted(xdnn::Context* ctx, const T* x, const TID* index, const int32_t* block_statistic, T* sorted_x,
                         TID* moe_index, TID* expert_m, TID* sorted_tokens_num_lod, int64_t m, int64_t n,
                         int64_t expert_num, int64_t topk, int64_t block_num, int64_t ep_size, int64_t ep_rank);

// moe_ep_pre_sorted_quant2d: EP-aware fused sort + INT8 quantization (W4A8 path)
// Wraps infer_ops::moe_ep_pre_sorted_quant2d.
// x [m, n] + index [m, topk] → quanted_y [m*topk, n, int8] + quant_scale [m*topk]
template <typename T, typename TID>
int XpuMoeEpPreSortedQuant2d(xdnn::Context* ctx, const T* x, const TID* index, const int32_t* block_statistic,
                             int8_t* quanted_y, TID* moe_index, TID* expert_m, TID* sorted_tokens_num_lod,
                             float* quant_scale, int64_t m, int64_t n, int64_t expert_num, int64_t topk,
                             int64_t block_num, int64_t ep_size, int64_t ep_rank);

// W4A8 MoE FC fusion: INT4 weight × INT8 activation expert projection
// Wraps xblas::moe_fc_fusion with int4_wo_int8 GEMM type.
// input [m_total, k, int8], weight [num_experts, n, k, int4] → output [m_total, n, T]
// act_scale [m_total] = per-token activation scale, weight_scale = per-channel weight scale
template <typename T, typename TID>
int XpuMoeW4A8FcFusion(xdnn::Context* ctx, const int8_t* input, const void* weight, T* output, const float* act_scale,
                       const float* weight_scale, int64_t tokens, int64_t topk, int64_t m_total, int64_t n, int64_t k,
                       int64_t num_experts, const TID* sorted_tokens_ids, const TID* sorted_tokens_num_lod);

// silu_and_mul_quant_unt: Fused SwiGLU + per-token INT8 quantization
// x [m, n] (where n = 2*ffn_hidden) → y [m, n/2, int8] + scale [m]
template <typename T>
int XpuSiluAndMulQuantUnt(xdnn::Context* ctx, const T* x, int8_t* y, float* scale, int64_t m, int64_t n, int64_t max_m,
                          const int32_t* recv_counts);

// moe_ep_ffn_post_fusion with float topk_score: EP-aware weighted reduce
// Wraps infer_ops::moe_ep_ffn_post_fusion (handles float → T score cast internally).
template <typename T, typename TID>
int XpuMoeEpFfnPostFusion(xdnn::Context* ctx, const T* expert_hidden, const TID* moe_index, const float* topk_score,
                          T* y, int64_t m, int64_t n, int64_t topk);

}  // namespace ksana_llm
