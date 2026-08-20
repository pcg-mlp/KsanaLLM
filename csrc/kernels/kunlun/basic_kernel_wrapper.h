/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <xpu/kernel/cluster3_type.h>
#include <xpu/runtime.h>
#include <xpu/xdnn.h>

#include <cstdint>
#include <vector>

#include "device/device_types.h"

// ── Type traits ───────────────────────────────────────────────────────────────

// KsanaLLM: typedef int16_t float16; typedef uint16_t bfloat16.
// xdnn .so: struct ::float16 / ::bfloat16 from cluster3_type.h.
// Both are 2 bytes with identical memory layout → reinterpret_cast is safe.
using xdnn_fp16 = ::float16;
using xdnn_bf16 = ::bfloat16;

template <typename T>
struct XdnnType {
  using type = T;
};
template <>
struct XdnnType<ksana_llm::float16> {
  using type = xdnn_fp16;
};
template <>
struct XdnnType<ksana_llm::bfloat16> {
  using type = xdnn_bf16;
};

template <typename T>
using xdnn_t = typename XdnnType<T>::type;

template <typename T>
const xdnn_t<T>* xcast(const T* p) {
  return reinterpret_cast<const xdnn_t<T>*>(p);
}

template <typename T>
xdnn_t<T>* xcast_m(T* p) {
  return reinterpret_cast<xdnn_t<T>*>(p);
}

// ── XdnnContext ───────────────────────────────────────────────────────────────

// Forward-declare xpytorch::xpu::api::Context for the bridge — avoids leaking
// xpytorch headers to every translation unit that includes this header.
// MUST be at global scope: xpytorch headers define this type in ::xpytorch, not
// ksana_llm::xpytorch.  A forward-decl inside ksana_llm would create a different
// incomplete type and break all callers.
namespace xpytorch {
namespace xpu {
namespace api {
class Context;
}  // namespace api
}  // namespace xpu
}  // namespace xpytorch

namespace ksana_llm {

// Template naming convention: see KUNLUN_TEMPLATE_NAMING.md for the shared glossary.
// This file is the home of the generic GEMM/element-wise vocabulary:
//   T                — main compute/activation dtype (in == out)
//   TW               — weight dtype (often `= float`)
//   TX / TY          — input / output dtype when they differ (cast, quant GEMM)
//   TGEMM            — GEMM internal quant/accum dtype (`= int16_t`)
//   TACC             — accumulator/scalar dtype (`= int64_t`)
//   TID              — index/id dtype (gather/scatter, token id)

// RAII wrapper: should only be used for test fixtures that need isolated context lifetime.
// Production code should use KunlunDeviceContextExtension::GetXdnnContext() instead.
struct XdnnContext {
  xdnn::Context* ctx = nullptr;

  explicit XdnnContext(XPUStream stream) {
    ctx = xdnn::create_context();
    ctx->set_stream(stream);
  }
  ~XdnnContext() {
    if (ctx) {
      xdnn::destroy_context(ctx);
      ctx = nullptr;
    }
  }
  XdnnContext(const XdnnContext&) = delete;
  XdnnContext& operator=(const XdnnContext&) = delete;
};

// ── Elementwise ops ───────────────────────────────────────────────────────────

// element-wise add: z[i] = x[i] + y[i]
template <typename T>
int XpuAdd(xpytorch::xpu::api::Context* ctx, const T* x, const T* y, T* z, int64_t len);

// type cast
template <typename TX, typename TY>
int XpuCast(xpytorch::xpu::api::Context* ctx, const TX* x, TY* y, int64_t len);

// silu: y[i] = x[i] * sigmoid(x[i])
template <typename T>
int XpuSilu(xpytorch::xpu::api::Context* ctx, const T* x, T* y, int64_t len);

// gated_silu: output = silu(input) * gate; input [m,n], gate [m,n], output [m,n]
template <typename T>
int XpuSiluMul(xpytorch::xpu::api::Context* ctx, const T* input, const T* gate, T* output, int64_t m, int64_t n);

// row_based gated_silu: input [m, 2n] → output [m, n]; output = silu(left_half) * right_half
template <typename T>
int XpuRowBasedSiluMul(xdnn::Context* ctx, const T* input, T* output, int64_t m, int64_t n);

// gated_gelu: output = gelu(input) * gated_weight (gated_weight may be nullptr)
template <typename T>
int XpuGatedGelu(xpytorch::xpu::api::Context* ctx, const T* input, const T* gated_weight, T* output, int64_t m,
                 int64_t n);

// embedding lookup: output[i] = weight[input_ids[i], :]
template <typename T>
int XpuEmbeddingLookup(xpytorch::xpu::api::Context* ctx, const int64_t* input_ids, const T* weight, T* output,
                       int64_t token_num, int64_t hidden_units, int64_t vocab_size);

// 2D strided copy: m rows of n elements; src row stride ldx, dst row stride ldy
template <typename T>
int XpuCopy2d(xpytorch::xpu::api::Context* ctx, const T* src, T* dst, int64_t m, int64_t ldx, int64_t ldy, int64_t n);

// ── Linear ops ────────────────────────────────────────────────────────────────

// fc (matmul): y[m,n] = x[m,k] * w[k,n]; x_trans/w_trans control transposition.
// maxptr nullptr means no quantization scaling.
// bf16 path routes through xblas::fc_fusion (xdnn::fc does not export bf16).
template <typename TX, typename TW, typename TY, typename TGEMM = int16_t>
int XpuFc(xpytorch::xpu::api::Context* ctx, const TX* x, const TW* w, TY* y, int64_t m, int64_t n, int64_t k,
          bool x_trans, bool w_trans, const float* x_maxptr, const float* w_maxptr, float* y_maxptr);

// TODO(zhongzhicao): XpuFc / XpuFcFusion 暂时并存，后续验证后只保留功能更强的一个。
// fc_fusion (xblas): higher-performance GEMM via xblas::fc_fusion with FcFusionTensor API.
// Supports fp16/bf16/fp32 with proper internal accumulation types.
template <typename TX, typename TW, typename TY>
int XpuFcFusion(xdnn::Context* ctx, const TX* x, const TW* w, TY* y, int64_t m, int64_t n, int64_t k, bool x_trans,
                bool w_trans, const float* x_maxptr, const float* w_maxptr, float* y_maxptr);

// ── Utility ops (XSGLANG-aligned) ─────────────────────────────────────────────

// arange: fill y with increasing sequence y[i] = begin + i * step
template <typename T, typename TACC = int64_t>
int XpuArange(xpytorch::xpu::api::Context* ctx, T* y, TACC begin, TACC step, int64_t len);

// constant: fill x with a constant value val
template <typename T>
int XpuConstant(xpytorch::xpu::api::Context* ctx, T* x, int64_t len, T val);

// neg: y[i] = -x[i]
template <typename T>
int XpuNeg(xpytorch::xpu::api::Context* ctx, const T* x, T* y, int64_t len);

// clamp: y[i] = clamp(x[i], min_val, max_val)
template <typename T>
int XpuClamp(xpytorch::xpu::api::Context* ctx, const T* x, T* y, int64_t len, T min_val, T max_val);

// add_scalar: y[i] = x[i] + other * alpha
template <typename T>
int XpuAddScalar(xpytorch::xpu::api::Context* ctx, const T* x, T* y, float alpha, float other, int64_t len);

// sub_scalar: y[i] = x[i] - other * alpha
template <typename T>
int XpuSubScalar(xpytorch::xpu::api::Context* ctx, const T* x, T* y, float alpha, float other, int64_t len);

// gather: y = x[..., index[idx], ...] along axis
template <typename T, typename TID>
int XpuGather(xpytorch::xpu::api::Context* ctx, const T* x, const TID* index, T* y, const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& idxshape, int64_t axis);

// gather_nd: gather elements from x according to N-dimensional index
template <typename T, typename TID>
int XpuGatherNd(xpytorch::xpu::api::Context* ctx, const T* x, const TID* index, T* y,
                const std::vector<int64_t>& xshape, const std::vector<int64_t>& index_shape);

// index_select: select along axis by 1D index array of length index_len
template <typename T, typename TID>
int XpuIndexSelect(xpytorch::xpu::api::Context* ctx, const T* x, const TID* index, T* y,
                   const std::vector<int64_t>& xshape, int64_t index_len, int64_t axis);

// scatter_element: scatter updates into y at index positions along axis
template <typename T, typename TID>
int XpuScatterElement(xpytorch::xpu::api::Context* ctx, const T* x, const T* updates, const TID* index, T* y,
                      const std::vector<int64_t>& xyshape, const std::vector<int64_t>& updshape,
                      const std::vector<int64_t>& idxshape, int64_t axis, int64_t reduction = 0);

// scatter_nd: scatter updates into y at N-dimensional index positions
template <typename T, typename TID>
int XpuScatterNd(xpytorch::xpu::api::Context* ctx, const T* x, const T* updates, T* y, const TID* index,
                 const std::vector<int64_t>& xshape, const std::vector<int64_t>& index_shape, bool is_overwrite = true);

// strided_slice: extract a slice from x with strides
template <typename T>
int XpuStridedSlice(xpytorch::xpu::api::Context* ctx, const T* x, T* y, const std::vector<int64_t>& xshape,
                    const std::vector<int64_t>& starts, const std::vector<int64_t>& ends,
                    const std::vector<int64_t>& strides, int64_t offset = 0);

// concat: concatenate a list of tensors along axis
template <typename T>
int XpuConcat(xpytorch::xpu::api::Context* ctx, const std::vector<const T*>& x_list, T* y,
              const std::vector<std::vector<int64_t>>& xshape_list, int64_t axis);

// split: split a tensor into multiple outputs along axis (inverse of concat)
template <typename T>
int XpuSplit(xpytorch::xpu::api::Context* ctx, const T* x, const std::vector<T*>& y_list,
             const std::vector<int64_t>& xshape, const std::vector<int64_t>& split_list, int64_t axis);

// cumsum: prefix sum along axis
template <typename T>
int XpuCumsum(xpytorch::xpu::api::Context* ctx, const T* x, T* y, const std::vector<int64_t>& xshape, bool reverse,
              bool exclusive, int64_t axis);

// sort: sort each row of x[m,n]; y = sorted values, index = sorted indices
template <typename T, typename TID = int64_t>
int XpuSort(xpytorch::xpu::api::Context* ctx, const T* x, T* y, TID* index, int64_t m, int64_t n, bool compare = false);

// compare_*: element-wise comparison with scalar, result is bool mask
template <typename T>
int XpuEqScalar(xpytorch::xpu::api::Context* ctx, const T* x, bool* y, T other, int64_t len);
template <typename T>
int XpuNeScalar(xpytorch::xpu::api::Context* ctx, const T* x, bool* y, T other, int64_t len);
template <typename T>
int XpuGtScalar(xpytorch::xpu::api::Context* ctx, const T* x, bool* y, T other, int64_t len);
template <typename T>
int XpuGeScalar(xpytorch::xpu::api::Context* ctx, const T* x, bool* y, T other, int64_t len);
template <typename T>
int XpuLtScalar(xpytorch::xpu::api::Context* ctx, const T* x, bool* y, T other, int64_t len);
template <typename T>
int XpuLeScalar(xpytorch::xpu::api::Context* ctx, const T* x, bool* y, T other, int64_t len);

// stable_sort: stable sort each row of x[m,n]; y = sorted values, index = sorted indices
template <typename T, typename TID = int64_t>
int XpuStableSort(xpytorch::xpu::api::Context* ctx, const T* x, T* y, TID* index, int64_t m, int64_t n,
                  bool compare = false);

// unique_count: count unique elements in array
template <typename T, typename TID = int64_t>
int XpuUniqueCount(xpytorch::xpu::api::Context* ctx, const T* x, int64_t* unique_len, int64_t x_len,
                   T* sorted_x = nullptr, TID* sorted_idx = nullptr, T* stage1_y = nullptr, TID* stage1_idx = nullptr,
                   int64_t* count_min_max = nullptr, bool is_ascend_sorted = false);

// copy: 1D device memory copy y[i] = x[i] (memcpy_1d equivalent)
template <typename T>
int XpuCopy(xpytorch::xpu::api::Context* ctx, const T* x, T* y, int64_t len);

// mul_tensor: element-wise multiply z[i] = x[i] * y[i]
template <typename T>
int XpuMulTensor(xpytorch::xpu::api::Context* ctx, const T* x, const T* y, T* z, const std::vector<int64_t>& xshape,
                 bool accumulate = false);

// mul_scalar: y[i] = x[i] * other
template <typename T>
int XpuMulScalar(xpytorch::xpu::api::Context* ctx, const T* x, T* y, float other, int64_t len);

// softmax: y = softmax(x) along axis
template <typename T>
int XpuSoftmax(xpytorch::xpu::api::Context* ctx, const T* x, T* y, const std::vector<int64_t>& xshape, int64_t axis);

// scatter: scatter updates into y along axis by 1D index
template <typename T, typename TID>
int XpuScatter(xpytorch::xpu::api::Context* ctx, const T* x, const T* updates, T* y, const TID* index,
               int64_t index_len, const std::vector<int64_t>& xshape, int64_t axis, bool is_overwrite = true);

// ── Build-ahead (depth=2 executor pipeline overlap) token-relay kernels ──────
// Phase 1 (dense / MTP-off / TP=1 / concurrency small). Not composed from xdnn
// ops: the semantics are a data-dependent placeholder guard + req_id linear
// scan with early-exit, for which there's no xdnn primitive, and going back to
// the host to do the scan would defeat the point of a device-resident token
// relay. Handwritten XPU kernels instead (custom_kernels.xpu +
// custom_kernels_launcher.cpp, mirroring vector_add_kernel's launch idiom).
// See build_ahead_kunlun_phase1_design.md §3 for the full protocol.
//
//   cur_pairs [2*batch_size] int64, interleaved (req_id, last_token_offset) per
//             request, written by ModelInput::PrepareInputIds each step.
//   ring      [1 + 2*ring_capacity] int64, layout
//             [count, req_id_0, token_0, req_id_1, token_1, ...]; count = the
//             batch_size written by the XpuWriteRing call that step.
//
// Replace input_ids[dst_off] == kFastPathPlaceholderTokenId (-1) placeholders
// with the token the *previous* step's sampling wrote into prev_ring for the
// same req_id (linear scan over prev_ring's `count` entries; batch_size/count
// are tiny in Phase 1, so this scan is negligible). On req_id miss (defensive
// only -- should not happen on the fast path) input_ids[dst_off] is left at
// -1. If prev_ring[0] == 0 (cold start), the scan is vacuous: natural no-op.
//   input_ids   [input_ids_len] int32 device buffer, persistent across steps.
//   cur_pairs   [2*batch_size]  int64 device buffer, this step's pairs.
//   prev_ring   [1 + 2*B_prev]  int64 device buffer, the ring slot the
//                               previous step's XpuWriteRing wrote.
int XpuReplaceLastToken(xpytorch::xpu::api::Context* ctx, int32_t* input_ids, const int64_t* cur_pairs,
                        const int64_t* prev_ring, int64_t batch_size, int64_t input_ids_len);

// Build the PERSISTENT per-req union of latest sampled tokens in the destination
// ring slot: carry forward prev_ring's (req_id, token) entries (dropping any in
// evict_ids), then upsert this step's batch. This keeps every active req's last
// token available across steps even when intervening steps ran a different batch
// (continuous batching + build-ahead), which a per-step-snapshot ring could not.
// consuming Sampler::device_output_tokens_ directly (no D2H round trip).
//   ring            [1 + 2*ring_capacity] int64 dst slot; ring[0] = entry count
//                   on return.
//   prev_ring       [1 + 2*ring_capacity] int64 src slot (the OTHER ping-pong
//                   slot, holding the previous step's union). Must not alias ring.
//   cur_pairs       [2*batch_size] int64, this step's (req_id, last_token_offset);
//                   only the req_id half (even indices) is read.
//   sampled_tokens  [batch_size] uint32 device buffer (vocab id, non-negative).
//   evict_ids       [evict_count] int64, req_ids that finished this step and must
//                   be dropped from the carried-forward union (may be nullptr when
//                   evict_count == 0).
//   ring_capacity   max entry count the slot holds (sized 2*max_batch_size).
int XpuWriteRing(xpytorch::xpu::api::Context* ctx, int64_t* ring, const int64_t* prev_ring, const int64_t* cur_pairs,
                 const uint32_t* sampled_tokens, const int64_t* evict_ids, int64_t evict_count, int64_t batch_size,
                 int64_t ring_capacity);

// masked_fill: y[i] = (mask[i] != 0) ? value : x[i]
template <typename T>
int XpuMaskedFill(xpytorch::xpu::api::Context* ctx, const T* x, const bool* mask, T* y, float value, int64_t len);

// multinomial: sample num_samples token indices from probability distribution
template <typename T>
int XpuMultinomial(xpytorch::xpu::api::Context* ctx, const T* probs, int64_t* indices, int64_t m, int64_t n,
                   int64_t num_samples, int64_t seed, bool replacement = true);

// ── Norm ops ──────────────────────────────────────────────────────────────────

// layer_norm: x[m,n] -> y[m,n]; scale/bias [n]; mean/var [m] (may be nullptr)
template <typename T, typename TW = float>
int XpuLayerNorm(xpytorch::xpu::api::Context* ctx, const T* x, T* y, int64_t m, int64_t n, float eps, const TW* scale,
                 const TW* bias, float* mean, float* var);

// rms_layer_norm: bias may be nullptr
template <typename T, typename TW = float, typename TY = T>
int XpuRmsLayerNorm(xdnn::Context* ctx, const T* x, TY* y, int64_t m, int64_t n, float eps, const TW* scale,
                    const TW* bias, float* var);

// fused_add_rmsnorm (in-place):
//   residual += input; input = rmsnorm(residual, weight)
template <typename T, typename TW = float>
int XpuFusedAddRmsNorm(xdnn::Context* ctx, T* input, T* residual, const TW* weight, int64_t m, int64_t n, float eps);

}  // namespace ksana_llm

// Custom XPU kernel launchers (defined in custom_kernels_launcher.cpp, compiled with xpu-clang).
// Declared with extern "C" linkage (no name mangling).
extern "C" int launch_vector_add_kernel(const float* a, const float* b, float* c, int n, void* stream);
extern "C" int launch_replace_last_token_kernel(int32_t* input_ids, const int64_t* cur_pairs, const int64_t* prev_ring,
                                                int64_t batch_size, int64_t input_ids_len, void* stream);
extern "C" int launch_write_ring_kernel(int64_t* ring, const int64_t* prev_ring, const int64_t* cur_pairs,
                                        const uint32_t* sampled_tokens, const int64_t* evict_ids, int64_t evict_count,
                                        int64_t batch_size, int64_t ring_capacity, void* stream);
