/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "kernels/kunlun/basic_kernel_wrapper.h"

namespace ksana_llm {

// Template naming convention: see KUNLUN_TEMPLATE_NAMING.md for the shared glossary.
// The public API here uses only T (bf16/fp16 activation); int8 weight/scale
// types are fixed, not templated.

// ── Per-token INT8 dynamic quantization ──────────────────────────────────────
// x[m, n] (bf16/fp16) → output_q[m, n] (int8) + output_scale[m] (fp32)
// 对应 xsglang: infer_ops::quant2d / sgl_per_token_quant_int8
template <typename T>
int XpuPerTokenQuantInt8(xdnn::Context* ctx, const T* input, int8_t* output_q, float* output_scale, int64_t m,
                         int64_t n);

// ── W8 INT8 per-channel GEMM (dynamic activation quantization) ───────────────
// y[m,n] = dequant(quant(x[m,k]) × w_int8[k,n], scale_x[m], scale_w[n]) + bias
// 对应 xsglang: klx_int8_scaled_mm
// 内部: x(bf16) → quant2d → x_int8 + scale_x → xblas::fc_fusion → y(bf16)
template <typename T>
int XpuInt8ScaledMM(xdnn::Context* ctx,
                    const T* x,            // [m, k] bf16/fp16 activation
                    const int8_t* w,       // [k, n] int8 weight (已转置存储)
                    const float* scale_w,  // [n] fp32 per-channel weight scale
                    T* y,                  // [m, n] output (bf16/fp16)
                    int64_t m, int64_t n, int64_t k,
                    const float* bias = nullptr);  // [n] optional fp32 bias

// ── W8 INT8 GEMM with pre-quantized activation ──────────────────────────────
// 输入已是 INT8 (从 fused_add_rmsnorm_quant 输出), 跳过动态量化步骤
// 对应 xsglang: klx_int8_scaled_mm_not_absorb_quant
template <typename T>
int XpuInt8ScaledMMPreQuantized(xdnn::Context* ctx,
                                const int8_t* x_int8,  // [m, k] int8 (已量化)
                                const int8_t* w,       // [k, n] int8 weight
                                const float* scale_x,  // [m] per-token activation scale
                                const float* scale_w,  // [n] per-channel weight scale
                                T* y,                  // [m, n] output
                                int64_t m, int64_t n, int64_t k, const float* bias = nullptr);

// ── W4 INT4-packed per-channel GEMM (dynamic activation quantization) ────────
// y[m,n] = dequant(quant(x[m,k]) × w_int4[k/2,n], scale_x[m], scale_w[n])
// 对应 xsglang: klx_w4a8_per_chn_scaled_mm
// 注意: scale_w 必须已乘以 7 (INT4 range [-8,7] denormalization)
// 使用 xblas V2 API (NdScaleTensor + FcFusionDescV2<int4_wo_int8>)
template <typename T>
int XpuW4A8PerChnScaledMM(xdnn::Context* ctx,
                          const T* x,            // [m, k] bf16/fp16
                          const int8_t* w,       // [k/2, n] int4 packed as int8
                          const float* scale_w,  // [n] per-channel (已乘7)
                          T* y,                  // [m, n] output
                          int64_t m, int64_t n, int64_t k, const float* bias = nullptr);

// ── Fused W8A8 INT8 Dense MLP ────────────────────────────────────────────────
// 融合 gate_up_proj(INT8 GEMM) → SwiGLU → down_proj(INT8 GEMM)
// 对应 xsglang: fused_w8a8_int8_mlp
// workspace 需要 m*(2*inter + inter) * sizeof(T) 字节
template <typename T>
int XpuFusedInt8Mlp(xdnn::Context* ctx,
                    const T* x,                    // [m, hidden] bf16
                    const int8_t* gate_up_weight,  // [hidden, 2*inter] int8 (已转置: [k, n])
                    const float* gate_up_scale,    // [2*inter] fp32
                    const int8_t* down_weight,     // [inter, hidden] int8 (已转置: [k, n])
                    const float* down_scale,       // [hidden] fp32
                    T* output,                     // [m, hidden] bf16
                    void* workspace,               // workspace buffer
                    size_t workspace_size,         // workspace buffer size
                    int64_t m, int64_t hidden, int64_t inter);

// Helper: compute workspace size for XpuFusedInt8Mlp
inline size_t XpuFusedInt8MlpWorkspaceSize(int64_t max_m, int64_t inter, size_t dtype_size) {
  // inter_out [max_m, 2*inter] + act_out [max_m, inter]
  return static_cast<size_t>(max_m) * (2 * inter + inter) * dtype_size;
}

}  // namespace ksana_llm
