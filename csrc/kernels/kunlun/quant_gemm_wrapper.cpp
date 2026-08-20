/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernels/kunlun/quant_gemm_wrapper.h"

#include <infer_ops.h>
#include <xblas_legacy_api.h>
#include <xpu/refactor/util/int4.h>          // ::int4_t (global struct for INT4 packed type)
#include <xpu/refactor/util/int4_wo_int8.h>  // ::int4_wo_int8 (global struct for W4A8 GEMM type marker)
#include <xpu/xdnn.h>

namespace xblas = baidu::xpu::xblas;

// 静态变量统一管理 kernel 行为开关，便于后续按环境变量/配置切换，当前配置对齐xsgl最优配置
static const bool kIntertypeBfp16 = true;
static const bool kXdnnQuant = true;
static const bool kFastSwiglu = true;

namespace ksana_llm {

// ── Per-token INT8 quantization ──────────────────────────────────────────────

template <typename T>
int XpuPerTokenQuantInt8(xdnn::Context* ctx, const T* input, int8_t* output_q, float* output_scale, int64_t m,
                         int64_t n) {
  xdnn::ctx_guard guard(ctx);
  return infer_ops::quant2d<xdnn_t<T>>(ctx, xcast(input), output_q, output_scale, m, n,
                                       /*ldn=*/-1, /*force_sdnn=*/kXdnnQuant);
}

template int XpuPerTokenQuantInt8<bfloat16>(xdnn::Context*, const bfloat16*, int8_t*, float*, int64_t, int64_t);
template int XpuPerTokenQuantInt8<float16>(xdnn::Context*, const float16*, int8_t*, float*, int64_t, int64_t);
template int XpuPerTokenQuantInt8<float>(xdnn::Context*, const float*, int8_t*, float*, int64_t, int64_t);

// ── W8 INT8 per-channel GEMM (with dynamic activation quantization) ─────────
// 框架侧需准备:
//   w: INT8 weight, 物理 shape [N, K] (trans_b=true)
//   scale_w: FP32 per-channel scale, shape [N] (max_abs 格式, 即已乘以 127)

template <typename T>
int XpuInt8ScaledMM(xdnn::Context* ctx, const T* x, const int8_t* w, const float* scale_w, T* y, int64_t m, int64_t n,
                    int64_t k, const float* bias) {
  using XT = xdnn_t<T>;
  xdnn::ctx_guard guard(ctx);

  // 1. Allocate temp buffers for quantization
  int8_t* x_int8 = guard.alloc_l3_or_gm<int8_t>(m * k);
  float* scale_x = guard.alloc_l3_or_gm<float>(m);
  if (!x_int8 || !scale_x) return -1;

  // 2. Per-token dynamic INT8 quantization
  int ret = infer_ops::quant2d<XT>(ctx, xcast(x), x_int8, scale_x, m, k,
                                   /*ldn=*/-1, /*force_sdnn=*/kXdnnQuant);
  if (ret != 0) return ret;

  // 3. INT8 GEMM with scale dequantization (FcFusion V1 API)
  xblas::FcFusionTensor<const int8_t> a{x_int8, nullptr, m, k, k, false};
  xblas::FcFusionTensor<const int8_t> b{w, nullptr, n, k, k, true};
  xblas::FcFusionTensor<const XT> c{xcast(y), nullptr, m, n, n, false};
  xblas::FcFusionTensor<XT> d{xcast_m(y), nullptr, m, n, n, false};

  xblas::FcFusionEpilogue<float, float> epilogue{
      xdnn::Activation_t::LINEAR,
      bias,                         // [n] nullable bias
      scale_x,                      // scale_a: per-token [m]
      const_cast<float*>(scale_w),  // scale_b: per-channel [n]
      1,                            // scale_a_mode: 1 = per-row
      1,                            // scale_b_mode: 1 = per-column
      nullptr                       // res_maxptr
  };

  if (kIntertypeBfp16) {
    // Intermediate result in bf16/fp16 (same as output dtype)
    xblas::FcFusionDesc<int8_t, float, XT> desc{1.0f, 0.0f, nullptr};
    return xblas::fc_fusion(ctx, a, b, c, d, desc, epilogue);
  } else {
    // Intermediate result in fp32 (higher precision)
    xblas::FcFusionDesc<int8_t, float, float> desc{1.0f, 0.0f, nullptr};
    return xblas::fc_fusion(ctx, a, b, c, d, desc, epilogue);
  }
}

template int XpuInt8ScaledMM<bfloat16>(xdnn::Context*, const bfloat16*, const int8_t*, const float*, bfloat16*, int64_t,
                                       int64_t, int64_t, const float*);
template int XpuInt8ScaledMM<float16>(xdnn::Context*, const float16*, const int8_t*, const float*, float16*, int64_t,
                                      int64_t, int64_t, const float*);

// float specialization: INT8 GEMM not supported for fp32 output (SDK has no such instantiation)
template <>
int XpuInt8ScaledMM<float>(xdnn::Context* /*ctx*/, const float* /*x*/, const int8_t* /*w*/, const float* /*scale_w*/,
                           float* /*y*/, int64_t /*m*/, int64_t /*n*/, int64_t /*k*/, const float* /*bias*/) {
  return -1;  // unsupported: xblas has no fp32 INT8 GEMM instantiation
}

// ── W8 INT8 GEMM with pre-quantized activation ──────────────────────────────
// 框架侧需准备:
//   w: INT8 weight, 物理 shape [N, K] (trans_b=true)
//   scale_w: FP32 per-channel scale, shape [N] (max_abs 格式)
//   x_int8: 已量化的 INT8 activation, shape [M, K]
//   scale_x: FP32 per-token scale, shape [M] (max_abs 格式)

template <typename T>
int XpuInt8ScaledMMPreQuantized(xdnn::Context* ctx, const int8_t* x_int8, const int8_t* w, const float* scale_x,
                                const float* scale_w, T* y, int64_t m, int64_t n, int64_t k, const float* bias) {
  using XT = xdnn_t<T>;
  xdnn::ctx_guard guard(ctx);

  xblas::FcFusionTensor<const int8_t> a{x_int8, nullptr, m, k, k, false};
  xblas::FcFusionTensor<const int8_t> b{w, nullptr, n, k, k, true};
  xblas::FcFusionTensor<const XT> c{xcast(y), nullptr, m, n, n, false};
  xblas::FcFusionTensor<XT> d{xcast_m(y), nullptr, m, n, n, false};

  xblas::FcFusionEpilogue<float, float> epilogue{
      xdnn::Activation_t::LINEAR, bias, const_cast<float*>(scale_x), const_cast<float*>(scale_w), 1, 1, nullptr};

  if (kIntertypeBfp16) {
    xblas::FcFusionDesc<int8_t, float, XT> desc{1.0f, 0.0f, nullptr};
    return xblas::fc_fusion(ctx, a, b, c, d, desc, epilogue);
  } else {
    xblas::FcFusionDesc<int8_t, float, float> desc{1.0f, 0.0f, nullptr};
    return xblas::fc_fusion(ctx, a, b, c, d, desc, epilogue);
  }
}

template int XpuInt8ScaledMMPreQuantized<bfloat16>(xdnn::Context*, const int8_t*, const int8_t*, const float*,
                                                   const float*, bfloat16*, int64_t, int64_t, int64_t, const float*);
template int XpuInt8ScaledMMPreQuantized<float16>(xdnn::Context*, const int8_t*, const int8_t*, const float*,
                                                  const float*, float16*, int64_t, int64_t, int64_t, const float*);

// float specialization: not supported
template <>
int XpuInt8ScaledMMPreQuantized<float>(xdnn::Context* /*ctx*/, const int8_t* /*x_int8*/, const int8_t* /*w*/,
                                       const float* /*scale_x*/, const float* /*scale_w*/, float* /*y*/, int64_t /*m*/,
                                       int64_t /*n*/, int64_t /*k*/, const float* /*bias*/) {
  return -1;
}

// ── W4 INT4-packed per-channel GEMM ──────────────────────────────────────────
// 框架侧需准备:
//   w: INT8 buffer, 物理 shape [K/2, N] (每字节存 2 个 INT4 元素, 沿 K 维 pack)
//   scale_w: FP32 per-channel scale, shape [N] (已乘以 7)
//   逻辑语义: Y[M,N] = dequant(quant(X[M,K])) * dequant(W[K,N])
// 注意: w_trans=false, 与 XpuInt8ScaledMM 的 trans_b=true 不同

template <typename T>
int XpuW4A8PerChnScaledMM(xdnn::Context* ctx, const T* x, const int8_t* w, const float* scale_w, T* y, int64_t m,
                          int64_t n, int64_t k, const float* bias) {
  using XT = xdnn_t<T>;
  xdnn::ctx_guard guard(ctx);

  // 1. Per-token INT8 quantization of activation
  int8_t* x_int8 = guard.alloc_l3_or_gm<int8_t>(m * k);
  float* x_scale = guard.alloc_l3_or_gm<float>(m);
  if (!x_int8 || !x_scale) return -1;

  int ret = infer_ops::quant2d<XT>(ctx, xcast(x), x_int8, x_scale, m, k);
  if (ret != 0) return ret;

  // 2. INT4 GEMM via xblas V2 API (NdScaleTensor)
  // int4_t and int4_wo_int8 are global type markers for INT4 weight-only quantization
  // (defined in xpu/refactor/util/int4.h and xpu/refactor/util/int4_wo_int8.h)

  // Activation: per-token scale
  xblas::NdScaleTensor<2, const int8_t, const float> a{
      x_int8,  nullptr, {m, k}, {k, 1},  // logical shape & strides
      x_scale,                           // scale data
      {1, k},  {m, 1},  {1, 1}           // scale logical shape, data shape, strides
  };

  // Weight: per-channel scale (already multiplied by 7)
  xblas::NdScaleTensor<2, const int4_t, const float> b{
      reinterpret_cast<const int4_t*>(w),
      nullptr,
      {k, n},
      {n, 1},                       // logical shape (unpacked dims)
      const_cast<float*>(scale_w),  // scale data
      {k, 1},
      {1, n},
      {n, 1}  // scale broadcast semantics
  };

  xblas::FcFusionTensor<const XT> c{xcast(y), nullptr, m, n, n, false};
  xblas::FcFusionTensor<XT> d{xcast_m(y), nullptr, m, n, n, false};

  xblas::FcFusionDescV2<int4_wo_int8, float, XT> desc{false, false, 1.0f, 0.0f, nullptr};
  xblas::FcFusionEpilogueV2<float> epilogue{xdnn::Activation_t::LINEAR, bias, nullptr};

  return xblas::fc_fusion(ctx, a, b, c, d, desc, epilogue);
}

template int XpuW4A8PerChnScaledMM<bfloat16>(xdnn::Context*, const bfloat16*, const int8_t*, const float*, bfloat16*,
                                             int64_t, int64_t, int64_t, const float*);
template int XpuW4A8PerChnScaledMM<float16>(xdnn::Context*, const float16*, const int8_t*, const float*, float16*,
                                            int64_t, int64_t, int64_t, const float*);

// float specialization: not supported
template <>
int XpuW4A8PerChnScaledMM<float>(xdnn::Context* /*ctx*/, const float* /*x*/, const int8_t* /*w*/,
                                 const float* /*scale_w*/, float* /*y*/, int64_t /*m*/, int64_t /*n*/, int64_t /*k*/,
                                 const float* /*bias*/) {
  return -1;
}

// ── Fused W8A8 INT8 MLP ─────────────────────────────────────────────────────

template <typename T>
int XpuFusedInt8Mlp(xdnn::Context* ctx, const T* x, const int8_t* gate_up_weight, const float* gate_up_scale,
                    const int8_t* down_weight, const float* down_scale, T* output, void* workspace,
                    size_t workspace_size, int64_t m, int64_t hidden, int64_t inter) {
  using XT = xdnn_t<T>;

  // Workspace layout: inter_out [m, 2*inter] | act_out [m, inter]
  size_t inter_out_bytes = static_cast<size_t>(m) * 2 * inter * sizeof(XT);
  size_t act_out_bytes = static_cast<size_t>(m) * inter * sizeof(XT);
  size_t required = inter_out_bytes + act_out_bytes;
  if (workspace_size < required) return -2;  // insufficient workspace

  XT* inter_out = reinterpret_cast<XT*>(workspace);
  XT* act_out = reinterpret_cast<XT*>(reinterpret_cast<char*>(workspace) + inter_out_bytes);

  // Step 1: gate_up projection — INT8 GEMM
  // x [m, hidden] × gate_up_weight [hidden, 2*inter] → inter_out [m, 2*inter]
  int ret = XpuInt8ScaledMM<T>(ctx, x, gate_up_weight, gate_up_scale, reinterpret_cast<T*>(inter_out), m, 2 * inter,
                               hidden, nullptr);
  if (ret != 0) return ret;

  // Step 2: SwiGLU activation
  // inter_out [m, 2*inter] → act_out [m, inter]
  if (kFastSwiglu) {
    ret = xdnn::fast_swiglu<XT>(ctx, inter_out, act_out, {m, 2 * inter}, 1, true);
  } else {
    ret = xdnn::swiglu<XT>(ctx, inter_out, act_out, {m, 2 * inter}, 1, true);
  }
  if (ret != 0) return ret;

  // Step 3: down projection — INT8 GEMM
  // act_out [m, inter] × down_weight [inter, hidden] → output [m, hidden]
  ret = XpuInt8ScaledMM<T>(ctx, reinterpret_cast<T*>(act_out), down_weight, down_scale, output, m, hidden, inter,
                           nullptr);
  return ret;
}

template int XpuFusedInt8Mlp<bfloat16>(xdnn::Context*, const bfloat16*, const int8_t*, const float*, const int8_t*,
                                       const float*, bfloat16*, void*, size_t, int64_t, int64_t, int64_t);
template int XpuFusedInt8Mlp<float16>(xdnn::Context*, const float16*, const int8_t*, const float*, const int8_t*,
                                      const float*, float16*, void*, size_t, int64_t, int64_t, int64_t);

// float specialization: not supported
template <>
int XpuFusedInt8Mlp<float>(xdnn::Context* /*ctx*/, const float* /*x*/, const int8_t* /*gate_up_weight*/,
                           const float* /*gate_up_scale*/, const int8_t* /*down_weight*/, const float* /*down_scale*/,
                           float* /*output*/, void* /*workspace*/, size_t /*workspace_size*/, int64_t /*m*/,
                           int64_t /*hidden*/, int64_t /*inter*/) {
  return -1;
}

}  // namespace ksana_llm
