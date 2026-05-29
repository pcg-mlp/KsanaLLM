/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

// Templated, host-only C++ wrapper around the CUDA kernels defined in
// c4.cuh / c128.cuh / fused_norm_rope.cuh / plan.cuh.
//
// This header can be included from both .cu and .cpp translation units.
// The kernel template definitions live in compressor_wrapper.cu, and only
// explicitly instantiated (InT, OutT [, kHeadDim]) combinations are available
// at link time.
//
// Usage from a .cpp caller:
//   using Wrap = llm_kernels::nvidia::CompressorWrapper<
//       __nv_bfloat16, __nv_bfloat16>;
//   Wrap::FlashC4Prefill(...);
//   Wrap::FlashC128Prefill<128>(...);

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/forward_mode.h"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/plan.h"
#include "csrc/utils/quant_type.h"

namespace llm_kernels {
namespace nvidia {

// -----------------------------------------------------------------------------
// CompressorWrapper<InT, OutT>
//
// Supported instantiations (must match the explicit instantiations at the
// bottom of compressor_wrapper.cu):
//   <__nv_bfloat16, __nv_bfloat16>
//   <__half,        __half>
//   <float,         float>
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
class CompressorWrapper {
 public:
  // ---------------------------------------------------------------------------
  // plan_prefill_compress: builds compress_plan / write_plan for prefill.
  // Thin passthrough; independent of (InT, OutT) but kept as a class member so
  // callers only need to instantiate the wrapper once per dtype combo.
  // ---------------------------------------------------------------------------
  static void PlanPrefillCompress(PrefillPlan* compress_plan, PrefillPlan* write_plan, const IndicesT* seq_lens,
                                  const IndicesT* extend_lens, uint32_t batch_size, uint32_t num_tokens,
                                  uint32_t compress_ratio, bool is_overlap, cudaStream_t stream);

  // ---------------------------------------------------------------------------
  // flash_c4_prefill (head_dim is a *compile-time* template parameter).
  // Supported kHeadDim: 128, 512.
  // ---------------------------------------------------------------------------
  template <int64_t kHeadDim>
  static void FlashC4Prefill(void* kv_score_buffer, const void* kv_score_input, void* kv_compressed_output,
                             const void* score_bias, const IndicesT* indices, const PrefillPlan* compress_plan,
                             const PrefillPlan* write_plan, const void* extra, uint32_t num_compress,
                             uint32_t num_write, cudaStream_t stream);

  // ---------------------------------------------------------------------------
  // flash_c4_decode (head_dim is a *compile-time* template parameter).
  // Supported kHeadDim: 128, 512.
  // ---------------------------------------------------------------------------
  template <int64_t kHeadDim>
  static void FlashC4Decode(void* kv_score_buffer, const void* kv_score_input, void* kv_compressed_output,
                            const void* score_bias, const IndicesT* indices, const IndicesT* seq_lens,
                            const IndicesT* extra, uint32_t batch_size, cudaStream_t stream);

  // ---------------------------------------------------------------------------
  // flash_c128_prefill (head_dim is a *compile-time* template parameter).
  // Supported kHeadDim: 128, 512.
  // ---------------------------------------------------------------------------
  template <int64_t kHeadDim>
  static void FlashC128Prefill(void* kv_score_buffer, const void* kv_score_input, void* kv_compressed_output,
                               const void* score_bias, const IndicesT* indices, const IndicesT* load_indices,
                               const PrefillPlan* compress_plan, const PrefillPlan* write_plan, uint32_t num_compress,
                               uint32_t num_write, cudaStream_t stream);

  // ---------------------------------------------------------------------------
  // flash_c128_decode (head_dim is a *compile-time* template parameter).
  // Supported kHeadDim: 128, 512.
  // ---------------------------------------------------------------------------
  template <int64_t kHeadDim>
  static void FlashC128Decode(void* kv_score_buffer, const void* kv_score_input, void* kv_compressed_output,
                              const void* score_bias, const IndicesT* indices, const IndicesT* seq_lens,
                              uint32_t batch_size, cudaStream_t stream);

  // ---------------------------------------------------------------------------
  // compress_fused_norm_rope_inplace
  //
  // Only InT is used as the element type (OutT is ignored for this entry).
  // kHeadDim is a compile-time parameter; supported values: 128, 512.
  // ---------------------------------------------------------------------------
  template <int64_t kHeadDim>
  static void CompressFusedNormRopeInplace(void* input, const void* weight, const void* handle, const void* freqs_cis,
                                           uint32_t num_works, ForwardMode mode, float eps, uint32_t compress_ratio,
                                           cudaStream_t stream);

  // ---------------------------------------------------------------------------
  // compress_fused_norm_rope_store
  //
  // Only supports kHeadDim=512.
  // page_bits is a runtime parameter; supported values: 1, 6, 8.
  // KV_DTYPE controls kv cache format (kFp8DsMla: scales + bf16 rope, kFp8E4M3: direct fp8)
  // ---------------------------------------------------------------------------
  template <int64_t kHeadDim, llm_kernels::utils::KVCacheType KV_DTYPE = llm_kernels::utils::KVCacheType::kFp8DsMla>
  static void CompressFusedNormRopeStore(const void* input, const void* weight, const void* handle,
                                         const void* freqs_cis, uint32_t num_works, ForwardMode mode, float eps,
                                         uint32_t compress_ratio, void* cache, const int32_t* indices,
                                         int64_t page_stride, uint32_t page_bits, cudaStream_t stream);

  // ---------------------------------------------------------------------------
  // indexer_fused_norm_rope_store
  //
  // Only supports kHeadDim=128.
  // page_bits is a runtime parameter; supported values: 1, 6, 8.
  // ---------------------------------------------------------------------------
  template <int64_t kHeadDim>
  static void IndexerFusedNormRopeStore(const void* input, const void* weight, const void* handle,
                                        const void* freqs_cis, uint32_t num_works, ForwardMode mode, float eps,
                                        uint32_t compress_ratio, void* cache, const int32_t* indices,
                                        int64_t page_stride, uint32_t page_bits, cudaStream_t stream);
};

}  // namespace nvidia
}  // namespace llm_kernels
