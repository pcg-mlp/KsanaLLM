/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernels/kunlun/moe_kernel_wrapper.h"

#include <infer_ops.h>         // infer_ops::moe_softmax_topk, moe_ffn_pre_sorted, etc.
#include <xblas_legacy_api.h>  // xblas::moe_fc_fusion
#include <xpu/kernel/cluster3_type.h>
#include <xpu/refactor/util/int4.h>          // ::int4_t
#include <xpu/refactor/util/int4_wo_int8.h>  // ::int4_wo_int8
#include <xpu/xdnn.h>

#include <type_traits>

namespace xblas = baidu::xpu::xblas;

namespace ksana_llm {

// ── sigmoid ──────────────────────────────────────────────────────────────────

template <typename T>
int XpuSigmoid(xdnn::Context* ctx, const T* x, T* y, int64_t len) {
  return xdnn::sigmoid<xdnn_t<T>>(ctx, xcast(x), xcast_m(y), len);
}

template int XpuSigmoid<float>(xdnn::Context*, const float*, float*, int64_t);
template int XpuSigmoid<float16>(xdnn::Context*, const float16*, float16*, int64_t);
template int XpuSigmoid<bfloat16>(xdnn::Context*, const bfloat16*, bfloat16*, int64_t);

// ── element-wise mul ─────────────────────────────────────────────────────────

template <typename T>
int XpuMul(xdnn::Context* ctx, const T* x, const T* y, T* z, int64_t len) {
  return xdnn::mul<xdnn_t<T>>(ctx, xcast(x), xcast(y), xcast_m(z), len);
}

template int XpuMul<float>(xdnn::Context*, const float*, const float*, float*, int64_t);
template int XpuMul<float16>(xdnn::Context*, const float16*, const float16*, float16*, int64_t);
template int XpuMul<bfloat16>(xdnn::Context*, const bfloat16*, const bfloat16*, bfloat16*, int64_t);

// ── broadcast mul ────────────────────────────────────────────────────────────

template <typename T>
int XpuBroadcastMul(xdnn::Context* ctx, const T* x, const T* y, T* z, const std::vector<int64_t>& x_shape,
                    const std::vector<int64_t>& y_shape) {
  return xdnn::broadcast_mul<xdnn_t<T>>(ctx, xcast(x), xcast(y), xcast_m(z), x_shape, y_shape);
}

template int XpuBroadcastMul<float>(xdnn::Context*, const float*, const float*, float*, const std::vector<int64_t>&,
                                    const std::vector<int64_t>&);
template int XpuBroadcastMul<float16>(xdnn::Context*, const float16*, const float16*, float16*,
                                      const std::vector<int64_t>&, const std::vector<int64_t>&);
template int XpuBroadcastMul<bfloat16>(xdnn::Context*, const bfloat16*, const bfloat16*, bfloat16*,
                                       const std::vector<int64_t>&, const std::vector<int64_t>&);

// ── fused swiglu (row-based, for MoE) ────────────────────────────────────────

template <typename T>
int XpuSwiglu(xdnn::Context* ctx, const T* x, T* z, int64_t m, int64_t n) {
  // swiglu: input x [m, 2*n], output z [m, n]
  // turn=true means layout is [gate|up] (first half is gate, second half is up)
  // NOTE: infer_ops::swiglu dispatches swiglu_fwd (not swish_glu_sdnn).
  // When KunlunXin provides a force_sdnn overload for the non-EP swiglu,
  // switch to that to align with XSGLANG's swish_glu_sdnn kernel.
  std::vector<int64_t> xshape = {m, 2 * n};
  return infer_ops::swiglu<xdnn_t<T>>(ctx, xcast(x), (const xdnn_t<T>*)nullptr, xcast_m(z), xshape, /*axis=*/1,
                                      /*turn=*/true);
}

template int XpuSwiglu<float>(xdnn::Context*, const float*, float*, int64_t, int64_t);
template int XpuSwiglu<float16>(xdnn::Context*, const float16*, float16*, int64_t, int64_t);
template int XpuSwiglu<bfloat16>(xdnn::Context*, const bfloat16*, bfloat16*, int64_t, int64_t);

// ── fast_swiglu (uses xdnn::fast_swiglu, enabled by XSGL_FAST_SWIGLU=1) ─────

template <typename T>
int XpuFastSwiglu(xdnn::Context* ctx, const T* x, T* z, int64_t m, int64_t n) {
  // fast_swiglu: input x [m, 2*n], output z [m, n]
  // turn=true: first half is gate (SiLU applied), second half is up (element-wise mul)
  std::vector<int64_t> xshape = {m, 2 * n};
  return xdnn::fast_swiglu<xdnn_t<T>>(ctx, xcast(x), xcast_m(z), xshape, /*axis=*/1, /*turn=*/true);
}

template int XpuFastSwiglu<float>(xdnn::Context*, const float*, float*, int64_t, int64_t);
template int XpuFastSwiglu<float16>(xdnn::Context*, const float16*, float16*, int64_t, int64_t);
template int XpuFastSwiglu<bfloat16>(xdnn::Context*, const bfloat16*, bfloat16*, int64_t, int64_t);

// ── fast_sigmoid (对应 xsglang: sigmoid_sdnn, 用于 Kimi-K2.6 MoE scoring) ────

template <typename T>
int XpuFastSigmoid(xdnn::Context* ctx, const T* x, T* y, int64_t len) {
  // xpytorch::xpu::api::fast_sigmoid 在 xsglang 中通过 xpukernel context 调用
  // 对于 KsanaLLM 通过 xdnn::sigmoid 路由, 性能已优化
  return xdnn::sigmoid<xdnn_t<T>>(ctx, xcast(x), xcast_m(y), len);
}

template int XpuFastSigmoid<float>(xdnn::Context*, const float*, float*, int64_t);
template int XpuFastSigmoid<float16>(xdnn::Context*, const float16*, float16*, int64_t);
template int XpuFastSigmoid<bfloat16>(xdnn::Context*, const bfloat16*, bfloat16*, int64_t);

// ── moe_fused_gate_kimi_k2: Sigmoid + TopK (Kimi-K2.6 专用) ─────────────────

template <typename T, typename TS, typename TID>
int XpuMoeFusedGateKimiK2(xdnn::Context* ctx, const T* input, const T* bias, TS* topk_score, TID* topk_idx, int64_t m,
                          int64_t num_experts, int64_t topk, bool renormalize, float routed_scaling_factor,
                          int32_t* block_statistic, int64_t block_num) {
  // 对应 xsglang: infer_ops::moe_fused_gate_kimi_k2
  // input [m, num_experts] + bias [num_experts] → sigmoid → topk → score
  return infer_ops::moe_fused_gate_kimi_k2<xdnn_t<T>, xdnn_t<TS>, TID>(
      ctx,
      xcast(input),                               // input_ptr
      xcast(bias),                                // bias_ptr (nullable)
      reinterpret_cast<xdnn_t<TS>*>(topk_score),  // output_ptr
      topk_idx,                                   // indices_ptr
      m,                                          // num_rows
      num_experts,                                // num_experts
      topk,                                       // topk
      renormalize,                                // renormalize
      routed_scaling_factor,                      // routed_scaling_factor
      true,                                       // apply_routed_scaling_factor_on_output
      block_statistic,                            // block_statistic
      block_num);                                 // block_num
}

template int XpuMoeFusedGateKimiK2<bfloat16, float, int>(xdnn::Context*, const bfloat16*, const bfloat16*, float*, int*,
                                                         int64_t, int64_t, int64_t, bool, float, int32_t*, int64_t);
template int XpuMoeFusedGateKimiK2<float16, float, int>(xdnn::Context*, const float16*, const float16*, float*, int*,
                                                        int64_t, int64_t, int64_t, bool, float, int32_t*, int64_t);
template int XpuMoeFusedGateKimiK2<float, float, int>(xdnn::Context*, const float*, const float*, float*, int*, int64_t,
                                                      int64_t, int64_t, bool, float, int32_t*, int64_t);

// ── moe_sigmoid_topk_norm_fusion: Sigmoid + Group TopK + Norm ────────────────

template <typename T, typename TS, typename TID>
int XpuMoeSigmoidTopkNorm(xdnn::Context* ctx, const T* input, const float* bias, float scale, TS* topk_score,
                          TID* topk_idx, int32_t* block_statistic, int64_t m, int64_t num_experts,
                          int64_t num_expert_group, int64_t topk_group, int64_t topk, int64_t block_num) {
  // 对应 xsglang: infer_ops::moe_sigmoid_topk_norm_fusion
  // Note: infer_ops API takes non-const float* bias (legacy API design), but does not modify it
  return infer_ops::moe_sigmoid_topk_norm_fusion<xdnn_t<T>, xdnn_t<TS>, TID>(
      ctx,
      xcast(input),                               // x
      const_cast<float*>(bias),                   // bias (fp32, nullable) — API takes non-const but doesn't modify
      scale,                                      // scale (routed_scaling_factor)
      reinterpret_cast<xdnn_t<TS>*>(topk_score),  // score
      topk_idx,                                   // index
      block_statistic,                            // block_statistic
      m,                                          // m
      num_experts,                                // n (num_experts)
      num_expert_group,                           // n_group
      topk_group,                                 // topk_group
      topk,                                       // top_k
      block_num);                                 // block_num
}

template int XpuMoeSigmoidTopkNorm<bfloat16, float, int>(xdnn::Context*, const bfloat16*, const float*, float, float*,
                                                         int*, int32_t*, int64_t, int64_t, int64_t, int64_t, int64_t,
                                                         int64_t);
template int XpuMoeSigmoidTopkNorm<float16, float, int>(xdnn::Context*, const float16*, const float*, float, float*,
                                                        int*, int32_t*, int64_t, int64_t, int64_t, int64_t, int64_t,
                                                        int64_t);
template int XpuMoeSigmoidTopkNorm<float, float, int>(xdnn::Context*, const float*, const float*, float, float*, int*,
                                                      int32_t*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

// ── MoE infer_ops wrappers ───────────────────────────────────────────────────

template <typename TI, typename TO, typename TID>
int XpuMoeSoftmaxTopk(xdnn::Context* ctx, const TI* gate, TO* topk_score, TID* topk_index, int32_t* block_statistic,
                      int64_t m, int64_t expert_num, int64_t topk, int64_t block_num) {
  // softmax + topk WITHOUT normalization (for models with norm_topk_prob=false, e.g. Qwen2-MoE)
  return infer_ops::moe_softmax_topk<xdnn_t<TI>, xdnn_t<TO>, TID>(ctx, xcast(gate),
                                                                  reinterpret_cast<xdnn_t<TO>*>(topk_score), topk_index,
                                                                  block_statistic, m, expert_num, topk, block_num);
}

template int XpuMoeSoftmaxTopk<float, float, int>(xdnn::Context*, const float*, float*, int*, int32_t*, int64_t,
                                                  int64_t, int64_t, int64_t);
template int XpuMoeSoftmaxTopk<float16, float, int>(xdnn::Context*, const float16*, float*, int*, int32_t*, int64_t,
                                                    int64_t, int64_t, int64_t);
template int XpuMoeSoftmaxTopk<bfloat16, float, int>(xdnn::Context*, const bfloat16*, float*, int*, int32_t*, int64_t,
                                                     int64_t, int64_t, int64_t);

template <typename TI, typename TO, typename TID>
int XpuMoeSoftmaxTopkNorm(xdnn::Context* ctx, const TI* gate, TO* norm_score, TID* topk_index, int32_t* block_statistic,
                          int64_t m, int64_t expert_num, int64_t topk, int64_t block_num, bool stable) {
  // softmax + topk WITH normalization (for models with norm_topk_prob=true, e.g. Qwen3-MoE)
  return infer_ops::moe_softmax_topk_norm_fusion<xdnn_t<TI>, xdnn_t<TO>, TID>(
      ctx, xcast(gate), reinterpret_cast<xdnn_t<TO>*>(norm_score), topk_index, block_statistic, m, expert_num, topk,
      block_num, nullptr, stable);
}

template int XpuMoeSoftmaxTopkNorm<float, float, int>(xdnn::Context*, const float*, float*, int*, int32_t*, int64_t,
                                                      int64_t, int64_t, int64_t, bool);
template int XpuMoeSoftmaxTopkNorm<float16, float, int>(xdnn::Context*, const float16*, float*, int*, int32_t*, int64_t,
                                                        int64_t, int64_t, int64_t, bool);
template int XpuMoeSoftmaxTopkNorm<bfloat16, float, int>(xdnn::Context*, const bfloat16*, float*, int*, int32_t*,
                                                         int64_t, int64_t, int64_t, int64_t, bool);

template <typename TID>
int XpuGenBlockStatistic(xdnn::Context* ctx, const TID* topk_ids, int32_t* block_statistic, int64_t seq_num,
                         int64_t expert_num, int64_t topk, int64_t block_num) {
  return infer_ops::gen_block_statistic<TID>(ctx, topk_ids, block_statistic, seq_num, expert_num, topk, block_num);
}

template int XpuGenBlockStatistic<int>(xdnn::Context*, const int*, int32_t*, int64_t, int64_t, int64_t, int64_t);

template <typename T, typename TID>
int XpuMoeFfnPreSorted(xdnn::Context* ctx, const T* x, const TID* index, const int32_t* block_statistic, T* sorted_x,
                       TID* moe_index, TID* expert_m, TID* sorted_tokens_num_lod, int64_t m, int64_t n,
                       int64_t expert_num, int64_t topk, int64_t block_num) {
  return infer_ops::moe_ffn_pre_sorted<xdnn_t<T>, TID>(ctx, xcast(x), index, block_statistic, xcast_m(sorted_x),
                                                       moe_index, expert_m, sorted_tokens_num_lod, m, n, expert_num,
                                                       topk, block_num);
}

template int XpuMoeFfnPreSorted<float, int>(xdnn::Context*, const float*, const int*, const int32_t*, float*, int*,
                                            int*, int*, int64_t, int64_t, int64_t, int64_t, int64_t);
template int XpuMoeFfnPreSorted<float16, int>(xdnn::Context*, const float16*, const int*, const int32_t*, float16*,
                                              int*, int*, int*, int64_t, int64_t, int64_t, int64_t, int64_t);
template int XpuMoeFfnPreSorted<bfloat16, int>(xdnn::Context*, const bfloat16*, const int*, const int32_t*, bfloat16*,
                                               int*, int*, int*, int64_t, int64_t, int64_t, int64_t, int64_t);

template <typename T, typename TID>
int XpuMoeFfnPostFusion(xdnn::Context* ctx, const T* expert_hidden, const TID* moe_index, const T* topk_score, T* y,
                        int64_t m, int64_t n, int64_t topk) {
  return infer_ops::moe_ffn_post_fusion<xdnn_t<T>, TID, float>(ctx, xcast(expert_hidden), moe_index, xcast(topk_score),
                                                               /*dequant_score=*/(const float*)nullptr, xcast_m(y), m,
                                                               n, topk);
}

template int XpuMoeFfnPostFusion<float, int>(xdnn::Context*, const float*, const int*, const float*, float*, int64_t,
                                             int64_t, int64_t);
template int XpuMoeFfnPostFusion<float16, int>(xdnn::Context*, const float16*, const int*, const float16*, float16*,
                                               int64_t, int64_t, int64_t);
template int XpuMoeFfnPostFusion<bfloat16, int>(xdnn::Context*, const bfloat16*, const int*, const bfloat16*, bfloat16*,
                                                int64_t, int64_t, int64_t);

// ── XpuMoeFfnPostFusionF32Score: float score → cast → post_fusion ───────────
// 对齐 xsglang klx_fused_moe.cc: 当 TD != TWEIGHT 时，先 cast topk_weight 到 TD。

template <typename T, typename TID>
int XpuMoeFfnPostFusionF32Score(xdnn::Context* ctx, const T* expert_hidden, const TID* moe_index,
                                const float* topk_score, T* y, int64_t m, int64_t n, int64_t topk) {
  if constexpr (std::is_same_v<T, float>) {
    // float data + float score: 直接转发到 existing wrapper
    return XpuMoeFfnPostFusion<float, TID>(ctx, expert_hidden, moe_index, topk_score, y, m, n, topk);
  } else {
    // bf16/fp16 data + float score: 需要先 cast score 到 T
    using XT = xdnn_t<T>;
    int64_t score_len = m * topk;

    // 使用 ctx_guard 分配临时 L3/GM 内存
    xdnn::ctx_guard guard(ctx);
    XT* score_cast = guard.alloc_l3_or_gm<XT>(score_len);
    if (!score_cast) return -1;

    // cast float → T
    int ret = xdnn::cast_v2<float, XT>(ctx, topk_score, score_cast, score_len);
    if (ret != 0) return ret;

    // 调用 infer_ops::moe_ffn_post_fusion
    ret =
        infer_ops::moe_ffn_post_fusion<XT, TID, float>(ctx, xcast(expert_hidden), moe_index, score_cast,
                                                       /*dequant_score=*/(const float*)nullptr, xcast_m(y), m, n, topk);
    // 同步：确保 kernel 完成后再让 ctx_guard 析构释放 workspace
    xpu_wait(ctx->xpu_stream);
    return ret;
  }
}

template int XpuMoeFfnPostFusionF32Score<float, int>(xdnn::Context*, const float*, const int*, const float*, float*,
                                                     int64_t, int64_t, int64_t);
template int XpuMoeFfnPostFusionF32Score<float16, int>(xdnn::Context*, const float16*, const int*, const float*,
                                                       float16*, int64_t, int64_t, int64_t);
template int XpuMoeFfnPostFusionF32Score<bfloat16, int>(xdnn::Context*, const bfloat16*, const int*, const float*,
                                                        bfloat16*, int64_t, int64_t, int64_t);

// ── xblas MoE FC wrapper ─────────────────────────────────────────────────────

// Type traits for MoE FC fusion — determines internal compute types per data type.
// Following xsglang's MOE_FC_T pattern (klx_fused_moe.cc lines 8-53).
// These must match the template instantiations available in libxpu_blas.so.
template <typename T>
struct MoeFcTypeTraits;

template <>
struct MoeFcTypeTraits<float16> {
  // float16 model: GEMM in float16, intermediate/weight in float16
  // Matches symbol: moe_fc_fusion<float16,float16,float16,float16, float,float,float16,float16, int,float,float>
  using TGEMM = float;
  using TGEMM_O = float;
  using TINTER_RES = xdnn_fp16;
  using TWEIGHT = xdnn_fp16;
};

template <>
struct MoeFcTypeTraits<bfloat16> {
  // bfloat16 model: GEMM in float32 for numerical stability
  // Matches symbol: moe_fc_fusion<bfloat16,bfloat16,bfloat16,bfloat16, float,float,float,float, int,float,float>
  using TGEMM = float;
  using TGEMM_O = float;
  using TINTER_RES = float;
  using TWEIGHT = float;
};

template <typename T, typename TID>
int XpuMoeFcFusion(xdnn::Context* ctx, const T* sorted_x, const T* weight, T* out, int64_t tokens, int64_t topk,
                   int64_t m_total, int64_t n, int64_t k, int64_t num_experts, const TID* sorted_tokens_ids,
                   const TID* sorted_tokens_num_lod) {
  using XT = xdnn_t<T>;
  using Traits = MoeFcTypeTraits<T>;
  using TGEMM = typename Traits::TGEMM;
  using TGEMM_O = typename Traits::TGEMM_O;
  using TINTER_RES = typename Traits::TINTER_RES;
  using TWEIGHT = typename Traits::TWEIGHT;
  using TBIAS = float;
  using TSCALE = float;

  xdnn::ctx_guard guard(ctx);

  // MoeDesc (deprecated API, same as xsglang uses)
  xblas::MoeDesc<TGEMM, TGEMM_O, TINTER_RES, TWEIGHT, TID> desc{
      1.0f,                             // alpha
      0.0f,                             // beta
      tokens,                           // tokens
      topk,                             // topk
      (const TWEIGHT*)nullptr,          // topk_weights (not needed here)
      sorted_tokens_ids,                // topk_ids (= moe_index from pre_sorted)
      sorted_tokens_num_lod,            // sorted_tokens_num_lod
      sorted_tokens_ids,                // sorted_tokens_ids (= moe_index)
      xblas::SortMode::SortedByExpert,  // a_sort_mode
      xblas::SortMode::Normal,          // c_sort_mode
      xblas::SortMode::SortedByExpert,  // d_sort_mode
      xblas::SortMode::SortedByExpert   // topk_weights_sort_mode
  };

  xblas::MoeEpilogue<TBIAS, TSCALE> epilogue{
      xdnn::Activation_t::LINEAR,  // act
      nullptr,                     // bias
      nullptr,                     // scale_a
      nullptr,                     // scale_b
      nullptr,                     // smooth
      0,                           // bias_mode
      0,                           // scale_a_mode
      0,                           // scale_b_mode
      nullptr                      // res_maxptr
  };

  // a: input [m_total, k] (flattened sorted tokens)
  xblas::FcFusionTensor<const XT> a{xcast(sorted_x), nullptr, m_total, k, k, false};

  // b: weight [num_experts, n, k] (trans_b=true, so shape is [n, k] per expert)
  xblas::FcFusionBatchTensor<const XT> b{
      xcast(weight),  // ptr
      nullptr,        // maxptr
      num_experts,    // batch_count
      n,              // rows (output dim)
      k,              // cols (input dim)
      n * k,          // batch_offset
      k,              // ld
      true            // trans
  };

  // c: not used
  xblas::FcFusionBatchTensor<const XT> c{nullptr, nullptr, 0, 0, 0, 0, 0, false};

  // d: output [m_total, n]
  xblas::FcFusionBatchTensor<XT> d{xcast_m(out), nullptr, m_total, 1, n, n, n, false};

  int ret = xblas::moe_fc_fusion<XT, XT, XT, XT, TGEMM, TGEMM_O, TINTER_RES, TWEIGHT, TID, TBIAS, TSCALE>(
      ctx, a, b, c, d, desc, epilogue);
  // 同步：moe_fc_fusion 内部可能使用 context workspace，确保完成后再被下一个 kernel 复用
  xpu_wait(ctx->xpu_stream);
  return ret;
}

// float16 instantiation — matches libxpu_blas.so symbol
template int XpuMoeFcFusion<float16, int>(xdnn::Context*, const float16*, const float16*, float16*, int64_t, int64_t,
                                          int64_t, int64_t, int64_t, int64_t, const int*, const int*);

// bfloat16 instantiation — matches libxpu_blas.so symbol
template int XpuMoeFcFusion<bfloat16, int>(xdnn::Context*, const bfloat16*, const bfloat16*, bfloat16*, int64_t,
                                           int64_t, int64_t, int64_t, int64_t, int64_t, const int*, const int*);

// float32: no xblas::moe_fc_fusion<float,...> exists in libxpu_blas.so.
// Use per-expert xdnn::fc loop as fallback (Plan C, implemented in test code).
template <>
int XpuMoeFcFusion<float, int>(xdnn::Context* /*ctx*/, const float* /*sorted_x*/, const float* /*weight*/,
                               float* /*out*/, int64_t /*tokens*/, int64_t /*topk*/, int64_t /*m_total*/, int64_t /*n*/,
                               int64_t /*k*/, int64_t /*num_experts*/, const int* /*sorted_tokens_ids*/,
                               const int* /*sorted_tokens_num_lod*/) {
  return -1;  // unsupported: xblas::moe_fc_fusion has no fp32 instantiation in SDK
}

// ═══════════════════════════════════════════════════════════════════════════════
// EP (Expert Parallel) MoE kernel wrappers
// Aligned with xsglang/sgl-kernel/csrc/klx/klx_ep_moe.cc
// ═══════════════════════════════════════════════════════════════════════════════

// ── EP pre-sorted (BF16 path) ────────────────────────────────────────────────

template <typename T, typename TID>
int XpuMoeEpFfnPreSorted(xdnn::Context* ctx, const T* x, const TID* index, const int32_t* block_statistic, T* sorted_x,
                         TID* moe_index, TID* expert_m, TID* sorted_tokens_num_lod, int64_t m, int64_t n,
                         int64_t expert_num, int64_t topk, int64_t block_num, int64_t ep_size, int64_t ep_rank) {
  return infer_ops::moe_ep_ffn_pre_sorted<xdnn_t<T>, TID>(ctx, xcast(x), index, block_statistic, xcast_m(sorted_x),
                                                          moe_index, expert_m, sorted_tokens_num_lod, m, n, expert_num,
                                                          topk, block_num, ep_size, ep_rank);
}

template <>
int XpuMoeEpFfnPreSorted<float, int>(xdnn::Context*, const float*, const int*, const int32_t*, float*, int*, int*, int*,
                                     int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {
  return -1;
}

template int XpuMoeEpFfnPreSorted<float16, int>(xdnn::Context*, const float16*, const int*, const int32_t*, float16*,
                                                int*, int*, int*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                                int64_t);
template int XpuMoeEpFfnPreSorted<bfloat16, int>(xdnn::Context*, const bfloat16*, const int*, const int32_t*, bfloat16*,
                                                 int*, int*, int*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                                 int64_t);

// ── EP pre-sorted + quant2d (W4A8 path) ─────────────────────────────────────

template <typename T, typename TID>
int XpuMoeEpPreSortedQuant2d(xdnn::Context* ctx, const T* x, const TID* index, const int32_t* block_statistic,
                             int8_t* quanted_y, TID* moe_index, TID* expert_m, TID* sorted_tokens_num_lod,
                             float* quant_scale, int64_t m, int64_t n, int64_t expert_num, int64_t topk,
                             int64_t block_num, int64_t ep_size, int64_t ep_rank) {
  return infer_ops::moe_ep_pre_sorted_quant2d<xdnn_t<T>, TID>(ctx, xcast(x), index, block_statistic, quanted_y,
                                                              moe_index, expert_m, sorted_tokens_num_lod, quant_scale,
                                                              m, n, expert_num, topk, block_num, ep_size, ep_rank);
}

template int XpuMoeEpPreSortedQuant2d<float16, int>(xdnn::Context*, const float16*, const int*, const int32_t*, int8_t*,
                                                    int*, int*, int*, float*, int64_t, int64_t, int64_t, int64_t,
                                                    int64_t, int64_t, int64_t);
template int XpuMoeEpPreSortedQuant2d<bfloat16, int>(xdnn::Context*, const bfloat16*, const int*, const int32_t*,
                                                     int8_t*, int*, int*, int*, float*, int64_t, int64_t, int64_t,
                                                     int64_t, int64_t, int64_t, int64_t);

// ── W4A8 MoE FC fusion ──────────────────────────────────────────────────────

template <typename T, typename TID>
int XpuMoeW4A8FcFusion(xdnn::Context* ctx, const int8_t* input, const void* weight, T* output, const float* act_scale,
                       const float* weight_scale, int64_t tokens, int64_t topk, int64_t m_total, int64_t n, int64_t k,
                       int64_t num_experts, const TID* sorted_tokens_ids, const TID* sorted_tokens_num_lod) {
  if constexpr (std::is_same_v<T, float>) {
    return -1;  // W4A8 not supported for fp32
  } else {
    using XT = xdnn_t<T>;
    const int4_t* w_int4 = reinterpret_cast<const int4_t*>(weight);

    // MoeDesc: tokens/topk with SortedByExpert (matches klx_ep_moe.cc)
    xblas::MoeDesc<int4_wo_int8, float, XT, XT, TID> desc{1.0f,
                                                          0.0f,
                                                          tokens,
                                                          topk,
                                                          (const XT*)nullptr,
                                                          sorted_tokens_ids,
                                                          sorted_tokens_num_lod,
                                                          sorted_tokens_ids,
                                                          xblas::SortMode::SortedByExpert,
                                                          xblas::SortMode::Normal,
                                                          xblas::SortMode::SortedByExpert,
                                                          xblas::SortMode::SortedByExpert};

    // MoeEpilogue: scale_a_mode=1 (per-token act scale), scale_b_mode=2 (per-channel weight scale)
    xblas::MoeEpilogue<float, float> epilogue{xdnn::Activation_t::LINEAR,
                                              nullptr,       // bias
                                              act_scale,     // scale_a (per-token activation scale)
                                              weight_scale,  // scale_b (per-channel weight scale)
                                              nullptr,       // smooth
                                              0,             // bias_mode
                                              1,             // scale_a_mode (per-token)
                                              2,             // scale_b_mode (per-channel)
                                              nullptr};      // res_maxptr

    // a: quantized activation [m_total, k, int8]
    xblas::FcFusionTensor<const int8_t> a{input, nullptr, m_total, k, k, false};

    // b: INT4 weight [num_experts, n, k] (logical dims, SDK handles int4 packing)
    xblas::FcFusionBatchTensor<const int4_t, 0> b{w_int4,
                                                  nullptr,
                                                  static_cast<int>(num_experts),
                                                  static_cast<int>(n),
                                                  static_cast<int>(k),
                                                  static_cast<int>(n * k),
                                                  static_cast<int>(k),
                                                  true};

    // c: not used
    xblas::FcFusionBatchTensor<const XT, 0> c{nullptr, nullptr, 0, 0, 0, 0, 0, false};

    // d: output [m_total, n]
    xblas::FcFusionBatchTensor<XT, 0> d{
        xcast_m(output),     nullptr, static_cast<int>(m_total), 1, static_cast<int>(n), static_cast<int>(n),
        static_cast<int>(n), false};

    int ret = xblas::moe_fc_fusion(ctx, a, b, c, d, desc, epilogue);
    return ret;
  }
}

template int XpuMoeW4A8FcFusion<float16, int>(xdnn::Context*, const int8_t*, const void*, float16*, const float*,
                                              const float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                              const int*, const int*);
template int XpuMoeW4A8FcFusion<bfloat16, int>(xdnn::Context*, const int8_t*, const void*, bfloat16*, const float*,
                                               const float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                               const int*, const int*);

// ── Fused SwiGLU + INT8 quant ────────────────────────────────────────────────

template <typename T>
int XpuSiluAndMulQuantUnt(xdnn::Context* ctx, const T* x, int8_t* y, float* scale, int64_t m, int64_t n, int64_t max_m,
                          const int32_t* recv_counts) {
  return infer_ops::silu_and_mul_quant_unt<xdnn_t<T>>(ctx, xcast(x), y, scale, m, n,
                                                      /*scale_m=*/1,
                                                      /*scale_n=*/n / 2,
                                                      /*turn=*/true, max_m,
                                                      /*num_groups=*/1, const_cast<int32_t*>(recv_counts),
                                                      /*force_sdnn=*/true);
}

template int XpuSiluAndMulQuantUnt<float16>(xdnn::Context*, const float16*, int8_t*, float*, int64_t, int64_t, int64_t,
                                            const int32_t*);
template int XpuSiluAndMulQuantUnt<bfloat16>(xdnn::Context*, const bfloat16*, int8_t*, float*, int64_t, int64_t,
                                             int64_t, const int32_t*);

// ── EP post-fusion (with float score) ────────────────────────────────────────

template <typename T, typename TID>
int XpuMoeEpFfnPostFusion(xdnn::Context* ctx, const T* expert_hidden, const TID* moe_index, const float* topk_score,
                          T* y, int64_t m, int64_t n, int64_t topk) {
  using XT = xdnn_t<T>;
  return infer_ops::moe_ep_ffn_post_fusion<XT, TID, float, float>(ctx, xcast(expert_hidden), moe_index, topk_score,
                                                                  (const float*)nullptr, xcast_m(y), m, n, topk);
}

template <>
int XpuMoeEpFfnPostFusion<float, int>(xdnn::Context*, const float*, const int*, const float*, float*, int64_t, int64_t,
                                      int64_t) {
  return -1;
}

template int XpuMoeEpFfnPostFusion<float16, int>(xdnn::Context*, const float16*, const int*, const float*, float16*,
                                                 int64_t, int64_t, int64_t);
template int XpuMoeEpFfnPostFusion<bfloat16, int>(xdnn::Context*, const bfloat16*, const int*, const float*, bfloat16*,
                                                  int64_t, int64_t, int64_t);

}  // namespace ksana_llm
