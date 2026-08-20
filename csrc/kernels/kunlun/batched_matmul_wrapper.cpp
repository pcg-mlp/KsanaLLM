/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernels/kunlun/batched_matmul_wrapper.h"

#include <infer_ops.h>
#include <xblas_legacy_api.h>
#include <xpu/xdnn.h>

namespace xblas = baidu::xpu::xblas;

namespace ksana_llm {

// ── Type traits for fc_batch ─────────────────────────────────────────────────
// Mirrors xsglang mla_bfp16_bmm.cc FC_BATCH_T pattern.

template <typename T>
struct BmmTypeTraits;

template <>
struct BmmTypeTraits<bfloat16> {
  using TGEMM = float;  // fp32 accumulation for bf16 numerical stability
  using TGEMMO = float;
  using TINTER_RES = xdnn_bf16;
};

template <>
struct BmmTypeTraits<float16> {
  using TGEMM = xdnn_fp16;  // fp16 accumulation
  using TGEMMO = xdnn_fp16;
  using TINTER_RES = xdnn_fp16;
};

// INT8 path type traits (mirrors mla_int8_bmm.cc FC_BATCH_T)
template <typename TOutput>
struct BmmInt8TypeTraits;

template <>
struct BmmInt8TypeTraits<bfloat16> {
  using TGEMMO = float;
  using TINTER_RES = float;
};

template <>
struct BmmInt8TypeTraits<float16> {
  using TGEMMO = float;
  using TINTER_RES = xdnn_fp16;
};

// ── BF16/FP16 Batched MatMul (FcFusionBatchTensor 路径，参照 klx_t_fc_t.cc) ───

template <typename T>
int XpuBatchedMatMul(xdnn::Context* ctx, const T* a, const T* b, T* c, int64_t m, int64_t n, int64_t k, int64_t batch) {
  using XT = xdnn_t<T>;
  using Traits = BmmTypeTraits<T>;
  using TGEMM = typename Traits::TGEMM;
  using TGEMMO = typename Traits::TGEMMO;
  using TINTER_RES = typename Traits::TINTER_RES;

  // A 物理布局 [M, B, K] → 逻辑 [B, M, K]
  // lda=K (batch维步长), stride=B*K (M维步长)
  xblas::FcFusionBatchTensor<const XT> A{xcast(a),       nullptr, batch, m, k, /*lda=*/k, /*stride=*/batch * k,
                                         /*trans=*/false};

  // B 物理布局 [B, N, K]，连续，trans_b=true
  // lda=N*K (batch维步长), stride=K (N维步长)
  xblas::FcFusionBatchTensor<const XT> B{xcast(b), nullptr, batch, n, k, /*lda=*/n * k, /*stride=*/k, /*trans=*/true};

  // C placeholder (beta=0 不读取)
  xblas::FcFusionBatchTensor<const XT> C{xcast(a),       nullptr, batch, m, n, /*lda=*/n, /*stride=*/batch * n,
                                         /*trans=*/false};

  // D 物理布局 [M, B, N] → 逻辑 [B, M, N]
  // lda=N (batch维步长), stride=B*N (M维步长)
  xblas::FcFusionBatchTensor<XT> D{xcast_m(c), nullptr, batch, m, n, /*lda=*/n, /*stride=*/batch * n, /*trans=*/false};

  xblas::FcBatchDesc<TGEMM, TGEMMO, TINTER_RES> desc;
  xblas::FcBatchEpilogue<float> epilogue;

  return xblas::fc_batch(ctx, A, B, C, D, desc, epilogue);
}

template int XpuBatchedMatMul<bfloat16>(xdnn::Context*, const bfloat16*, const bfloat16*, bfloat16*, int64_t, int64_t,
                                        int64_t, int64_t);
template int XpuBatchedMatMul<float16>(xdnn::Context*, const float16*, const float16*, float16*, int64_t, int64_t,
                                       int64_t, int64_t);

// float specialization: not supported by xblas fc_batch
template <>
int XpuBatchedMatMul<float>(xdnn::Context* /*ctx*/, const float* /*a*/, const float* /*b*/, float* /*c*/, int64_t /*m*/,
                            int64_t /*n*/, int64_t /*k*/, int64_t /*batch*/) {
  return -1;
}

// ── [未使用] NdScaleTensor 路径，原 deep_gemm 风格实现，保留备用 ─────────────
template <typename T>
int XpuBatchedMatMulNdScale(xdnn::Context* ctx, const T* a, const T* b, T* c, int64_t m, int64_t n, int64_t k,
                            int64_t batch) {
  using XT = xdnn_t<T>;
  using Traits = BmmTypeTraits<T>;
  using TGEMM = typename Traits::TGEMM;
  using TGEMMO = typename Traits::TGEMMO;
  using TINTER_RES = typename Traits::TINTER_RES;

  // A physical layout: [M, B, K] contiguous, stride = {B*K, K, 1}
  // Reinterpret as batch-first: shape {batch, m, k}, stride {k, batch*k, 1}
  const xblas::NdScaleTensor<3, const XT, const float> A{
      xcast(a), nullptr, {batch, m, k}, {k, batch * k, 1}, nullptr, {1, 1, k}, {batch, m, 1}, {1, 1, 1},
  };

  // B physical layout: [B, N, K] contiguous, stride = {N*K, K, 1}
  // trans_b=true: xblas treats B as [N, K] per batch and transposes to [K, N] internally.
  const xblas::NdScaleTensor<3, const XT, const float> B{
      xcast(b), nullptr, {batch, n, k}, {n * k, k, 1}, nullptr, {1, 1, k}, {batch, n, 1}, {n, 1, 1},
  };

  // C (unused input for beta=0): placeholder
  const xblas::NdScaleTensor<3, const XT, const float> C{
      nullptr, nullptr, {batch, m, n}, {n, batch * n, 1}, nullptr, {1, 1, 1}, {batch, m, 1}, {1, 1, 1},
  };

  // D physical layout: [M, B, N] contiguous, stride = {B*N, N, 1}
  // Reinterpret as batch-first: shape {batch, m, n}, stride {n, batch*n, 1}
  const xblas::NdScaleTensor<3, XT, const float> D{
      xcast_m(c), nullptr, {batch, m, n}, {n, batch * n, 1}, nullptr, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
  };

  // trans_b=true: B is stored as [N, K] per batch, xblas transposes internally to get [K, N].
  const xblas::FcBatchDescV2<TGEMM, TGEMMO, TINTER_RES> desc{
      false,  // trans_a
      true,   // trans_b
      false,  // trans_c
      false,  // trans_d
      1.0f,   // alpha
      0.0f,   // beta
  };

  const xblas::FcBatchEpilogue<float> epilogue;

  return xblas::fc_batch<3, XT, float, 0, 3, XT, float, 0, 3, XT, float, 0, 3, XT, float, 0, TGEMM, TGEMMO, TINTER_RES,
                         float>(ctx, A, B, C, D, desc, epilogue);
}

template int XpuBatchedMatMulNdScale<bfloat16>(xdnn::Context*, const bfloat16*, const bfloat16*, bfloat16*, int64_t,
                                               int64_t, int64_t, int64_t);
template int XpuBatchedMatMulNdScale<float16>(xdnn::Context*, const float16*, const float16*, float16*, int64_t,
                                              int64_t, int64_t, int64_t);

// float specialization: not supported by xblas fc_batch
template <>
int XpuBatchedMatMulNdScale<float>(xdnn::Context* /*ctx*/, const float* /*a*/, const float* /*b*/, float* /*c*/,
                                   int64_t /*m*/, int64_t /*n*/, int64_t /*k*/, int64_t /*batch*/) {
  return -1;
}

// ── INT8 Batched MatMul ──────────────────────────────────────────────────────

template <typename T>
int XpuBatchedMatMulInt8(xdnn::Context* ctx, const T* a, const int8_t* b, const float* b_scales, T* c, int64_t m,
                         int64_t n, int64_t k, int64_t batch) {
  using XT = xdnn_t<T>;
  using Traits = BmmInt8TypeTraits<T>;
  using TGEMMO = typename Traits::TGEMMO;
  using TINTER_RES = typename Traits::TINTER_RES;

  xdnn::ctx_guard guard(ctx);

  // Step 1: Per-token INT8 quantization of A with scale transpose.
  // A is [M, B, K] physically. quant2d_with_scale_transpose treats it as:
  //   x[m, group_num, n] → y_int8[m, group_num, n] + scale[group_num, m, 1]
  // Here: m=M, group_num=B(batch), n=K
  int8_t* a_int8 = guard.alloc_l3_or_gm<int8_t>(m * batch * k);
  float* a_scale = guard.alloc_l3_or_gm<float>(batch * m);
  if (!a_int8 || !a_scale) return -1;

  int ret = infer_ops::quant2d_with_scale_transpose<XT>(ctx, xcast(a), a_int8, a_scale, m, batch, k,
                                                        /*ldn=*/-1, /*force_sdnn=*/false);
  if (ret != 0) return ret;

  // Step 2: INT8 fc_batch
  // A_int8: [M, B, K] → batch-first {batch, m, k}, stride {k, batch*k, 1}
  // a_scale: [B, M] (output of quant2d_with_scale_transpose) → scale for A
  const xblas::NdScaleTensor<3, const int8_t, const float> A_nd{
      a_int8, nullptr, {batch, m, k}, {k, batch * k, 1}, a_scale, {1, 1, k}, {batch, m, 1}, {m, 1, 1},
  };

  // B_int8: [B, N, K] contiguous, with per-channel scale [B, N]
  const xblas::NdScaleTensor<3, const int8_t, const float> B_nd{
      b, nullptr, {batch, n, k}, {n * k, k, 1}, const_cast<float*>(b_scales), {1, 1, k}, {batch, n, 1}, {n, 1, 1},
  };

  // C placeholder (beta=0)
  const xblas::NdScaleTensor<3, const XT, const float> C_nd{
      nullptr, nullptr, {batch, m, n}, {n, batch * n, 1}, nullptr, {1, 1, 1}, {batch, m, 1}, {1, 1, 1},
  };

  // D output: [M, B, N] → batch-first {batch, m, n}, stride {n, batch*n, 1}
  const xblas::NdScaleTensor<3, XT, const float> D_nd{
      xcast_m(c), nullptr, {batch, m, n}, {n, batch * n, 1}, nullptr, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
  };

  const xblas::FcBatchDescV2<int8_t, TGEMMO, TINTER_RES> desc{
      false,  // trans_a
      true,   // trans_b
      false,  // trans_c
      false,  // trans_d
      1.0f,   // alpha
      0.0f,   // beta
  };

  const xblas::FcBatchEpilogue<float> epilogue;

  return xblas::fc_batch<3, int8_t, float, 0, 3, int8_t, float, 0, 3, XT, float, 0, 3, XT, float, 0, int8_t, TGEMMO,
                         TINTER_RES, float>(ctx, A_nd, B_nd, C_nd, D_nd, desc, epilogue);
}

template int XpuBatchedMatMulInt8<bfloat16>(xdnn::Context*, const bfloat16*, const int8_t*, const float*, bfloat16*,
                                            int64_t, int64_t, int64_t, int64_t);
template int XpuBatchedMatMulInt8<float16>(xdnn::Context*, const float16*, const int8_t*, const float*, float16*,
                                           int64_t, int64_t, int64_t, int64_t);

// float specialization: not supported
template <>
int XpuBatchedMatMulInt8<float>(xdnn::Context* /*ctx*/, const float* /*a*/, const int8_t* /*b*/,
                                const float* /*b_scales*/, float* /*c*/, int64_t /*m*/, int64_t /*n*/, int64_t /*k*/,
                                int64_t /*batch*/) {
  return -1;
}

}  // namespace ksana_llm
