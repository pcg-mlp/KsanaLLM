/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "kernels/kunlun/basic_kernel_wrapper.h"

namespace ksana_llm {

// Template naming convention: see KUNLUN_TEMPLATE_NAMING.md for the shared glossary.
// The public API here uses only T (in == out); the .cpp uses TOutput internally.

// ── BF16/FP16 Batched MatMul ─────────────────────────────────────────────────
// C[M, B, N] = A[M, B, K] × B[B, N, K]^T
// Weight B is stored in [B, N, K] layout; xblas::fc_batch uses trans_b=true.
// float specialization returns -1 (not supported by SDK).
template <typename T>
int XpuBatchedMatMul(xdnn::Context* ctx,
                     const T* a,  // [M, B, K]
                     const T* b,  // [B, N, K]
                     T* c,        // [M, B, N]
                     int64_t m, int64_t n, int64_t k, int64_t batch);

// ── [未使用] NdScaleTensor 路径，原 deep_gemm 风格实现，保留备用 ─────────────
// C[M, B, N] = A[M, B, K] × B[B, N, K]^T
// 与 XpuBatchedMatMul 功能相同，使用 NdScaleTensor + FcBatchDescV2 API。
template <typename T>
int XpuBatchedMatMulNdScale(xdnn::Context* ctx,
                            const T* a,  // [M, B, K]
                            const T* b,  // [B, N, K]
                            T* c,        // [M, B, N]
                            int64_t m, int64_t n, int64_t k, int64_t batch);

// ── INT8 Batched MatMul (dynamic activation quantization) ────────────────────
// C[M, B, N] = dequant(quant(A[M, B, K]) × B_int8[B, N, K]^T, a_scale, b_scale)
// Activation is bf16/fp16, internally quantized to INT8 via quant2d_with_scale_transpose.
// Weight B is stored in [B, N, K] layout; xblas::fc_batch uses trans_b=true.
// float specialization returns -1.
template <typename T>
int XpuBatchedMatMulInt8(xdnn::Context* ctx,
                         const T* a,             // [M, B, K] (bf16/fp16 activation)
                         const int8_t* b,        // [B, N, K] int8 weight
                         const float* b_scales,  // [B, N] per-channel weight scale
                         T* c,                   // [M, B, N] output
                         int64_t m, int64_t n, int64_t k, int64_t batch);

}  // namespace ksana_llm
