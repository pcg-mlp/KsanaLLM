/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

// Template-based dispatch layer: every method on CompressorWrapper<InT, OutT>
// forwards to the underlying CUDA kernel template with the same dtype(s).
// The bottom of this file explicitly instantiates the supported (InT, OutT)
// combinations; only those instantiations are available at link time.

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/compressor_wrapper.h"

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/c128.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/c4.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/fused_norm_rope.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/fused_norm_rope_store.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/fused_norm_rope_store_indexer.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/plan.cuh"

namespace llm_kernels {
namespace nvidia {


// =============================================================================
// PlanPrefillCompress
// =============================================================================
template <typename InT, typename OutT>
void CompressorWrapper<InT, OutT>::PlanPrefillCompress(PrefillPlan* compress_plan, PrefillPlan* write_plan,
                                                       const IndicesT* seq_lens, const IndicesT* extend_lens,
                                                       uint32_t batch_size, uint32_t num_tokens,
                                                       uint32_t compress_ratio, bool is_overlap, cudaStream_t stream) {
  plan_prefill_compress(compress_plan, write_plan, seq_lens, extend_lens, batch_size, num_tokens, compress_ratio,
                        is_overlap, stream);
}

// =============================================================================
// FlashC4Prefill
// =============================================================================
template <typename InT, typename OutT>
template <int64_t kHeadDim>
void CompressorWrapper<InT, OutT>::FlashC4Prefill(void* kv_score_buffer, const void* kv_score_input,
                                                  void* kv_compressed_output, const void* score_bias,
                                                  const IndicesT* indices, const PrefillPlan* compress_plan,
                                                  const PrefillPlan* write_plan, const void* extra,
                                                  uint32_t num_compress, uint32_t num_write, cudaStream_t stream) {
  flash_c4_prefill<InT, OutT, kHeadDim>(kv_score_buffer, kv_score_input, kv_compressed_output, score_bias, indices,
                                        compress_plan, write_plan, extra, num_compress, num_write, stream);
}

// =============================================================================
// FlashC4Decode
// =============================================================================
template <typename InT, typename OutT>
template <int64_t kHeadDim>
void CompressorWrapper<InT, OutT>::FlashC4Decode(void* kv_score_buffer, const void* kv_score_input,
                                                 void* kv_compressed_output, const void* score_bias,
                                                 const IndicesT* indices, const IndicesT* seq_lens,
                                                 const IndicesT* extra, uint32_t batch_size, cudaStream_t stream) {
  flash_c4_decode<InT, OutT, kHeadDim>(kv_score_buffer, kv_score_input, kv_compressed_output, score_bias, indices,
                                       seq_lens, extra, batch_size, stream);
}

// =============================================================================
// FlashC128Prefill
// =============================================================================
template <typename InT, typename OutT>
template <int64_t kHeadDim>
void CompressorWrapper<InT, OutT>::FlashC128Prefill(void* kv_score_buffer, const void* kv_score_input,
                                                    void* kv_compressed_output, const void* score_bias,
                                                    const IndicesT* indices, const IndicesT* load_indices,
                                                    const PrefillPlan* compress_plan, const PrefillPlan* write_plan,
                                                    uint32_t num_compress, uint32_t num_write, cudaStream_t stream) {
  flash_c128_prefill<InT, OutT, kHeadDim>(kv_score_buffer, kv_score_input, kv_compressed_output, score_bias, indices,
                                          load_indices, compress_plan, write_plan, num_compress, num_write, stream);
}

// =============================================================================
// FlashC128Decode
// =============================================================================
template <typename InT, typename OutT>
template <int64_t kHeadDim>
void CompressorWrapper<InT, OutT>::FlashC128Decode(void* kv_score_buffer, const void* kv_score_input,
                                                   void* kv_compressed_output, const void* score_bias,
                                                   const IndicesT* indices, const IndicesT* seq_lens,
                                                   uint32_t batch_size, cudaStream_t stream) {
  flash_c128_decode<InT, OutT, kHeadDim>(kv_score_buffer, kv_score_input, kv_compressed_output, score_bias, indices,
                                         seq_lens, batch_size, stream);
}

// =============================================================================
// CompressFusedNormRopeInplace  (only InT is used as the element type)
// =============================================================================
template <typename InT, typename OutT>
template <int64_t kHeadDim>
void CompressorWrapper<InT, OutT>::CompressFusedNormRopeInplace(void* input, const void* weight, const void* handle,
                                                                const void* freqs_cis, uint32_t num_works,
                                                                ForwardMode mode, float eps, uint32_t compress_ratio,
                                                                cudaStream_t stream) {
  compress_fused_norm_rope_inplace<InT, kHeadDim>(input, weight, handle, freqs_cis, num_works, mode, eps,
                                                  compress_ratio, stream);
}

// =============================================================================
// CompressFusedNormRopeStore  (only InT is used as the element type)
// =============================================================================
template <typename InT, typename OutT>
template <int64_t kHeadDim>
void CompressorWrapper<InT, OutT>::CompressFusedNormRopeStore(const void* input, const void* weight, const void* handle,
                                                              const void* freqs_cis, uint32_t num_works,
                                                              ForwardMode mode, float eps, uint32_t compress_ratio,
                                                              void* cache, const int32_t* indices, int64_t page_stride,
                                                              uint32_t page_bits, cudaStream_t stream) {
  compress_fused_norm_rope_store<InT, kHeadDim>(input, weight, handle, freqs_cis, num_works, mode, eps, compress_ratio,
                                                cache, indices, page_stride, page_bits, stream);
}

// =============================================================================
// IndexerFusedNormRopeStore  (only InT is used as the element type)
// =============================================================================
template <typename InT, typename OutT>
template <int64_t kHeadDim>
void CompressorWrapper<InT, OutT>::IndexerFusedNormRopeStore(const void* input, const void* weight, const void* handle,
                                                             const void* freqs_cis, uint32_t num_works,
                                                             ForwardMode mode, float eps, uint32_t compress_ratio,
                                                             void* cache, const int32_t* indices, int64_t page_stride,
                                                             uint32_t page_bits, cudaStream_t stream) {
  indexer_fused_norm_rope_store<InT, kHeadDim>(input, weight, handle, freqs_cis, num_works, mode, eps, compress_ratio,
                                               cache, indices, page_stride, page_bits, stream);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
// Class-level instantiations bring in PlanPrefillCompress for each supported
// (InT, OutT) combo.
template class CompressorWrapper<float, __nv_bfloat16>;
template class CompressorWrapper<__nv_bfloat16, __nv_bfloat16>;

// Member-function templates that take kHeadDim must be instantiated separately
// for every (InT, OutT, kHeadDim) combination we need at link time.
#define INSTANTIATE_HD_MEMBERS(IN_T, OUT_T, HD)                                                                      \
  template void CompressorWrapper<IN_T, OUT_T>::FlashC4Prefill<HD>(                                                  \
      void*, const void*, void*, const void*, const IndicesT*, const PrefillPlan*, const PrefillPlan*, const void*,  \
      uint32_t, uint32_t, cudaStream_t);                                                                             \
  template void CompressorWrapper<IN_T, OUT_T>::FlashC4Decode<HD>(void*, const void*, void*, const void*,            \
                                                                  const IndicesT*, const IndicesT*, const IndicesT*, \
                                                                  uint32_t, cudaStream_t);                           \
  template void CompressorWrapper<IN_T, OUT_T>::FlashC128Prefill<HD>(                                                \
      void*, const void*, void*, const void*, const IndicesT*, const IndicesT*, const PrefillPlan*,                  \
      const PrefillPlan*, uint32_t, uint32_t, cudaStream_t);                                                         \
  template void CompressorWrapper<IN_T, OUT_T>::FlashC128Decode<HD>(                                                 \
      void*, const void*, void*, const void*, const IndicesT*, const IndicesT*, uint32_t, cudaStream_t);             \
  template void CompressorWrapper<OUT_T, OUT_T>::CompressFusedNormRopeInplace<HD>(                                   \
      void*, const void*, const void*, const void*, uint32_t, ForwardMode, float, uint32_t, cudaStream_t);
INSTANTIATE_HD_MEMBERS(float, __nv_bfloat16, 128);
INSTANTIATE_HD_MEMBERS(float, __nv_bfloat16, 512);
#undef INSTANTIATE_HD_MEMBERS

#define INSTANTIATE_HD_MEMBERS(IN_T, OUT_T, HD)                                                          \
  template void CompressorWrapper<OUT_T, OUT_T>::CompressFusedNormRopeStore<HD>(                         \
      const void*, const void*, const void*, const void*, uint32_t, ForwardMode, float, uint32_t, void*, \
      const int32_t*, int64_t, uint32_t, cudaStream_t)
INSTANTIATE_HD_MEMBERS(float, __nv_bfloat16, 512);
#undef INSTANTIATE_HD_MEMBERS

#define INSTANTIATE_HD_MEMBERS(IN_T, OUT_T, HD)                                                          \
  template void CompressorWrapper<OUT_T, OUT_T>::IndexerFusedNormRopeStore<HD>(                          \
      const void*, const void*, const void*, const void*, uint32_t, ForwardMode, float, uint32_t, void*, \
      const int32_t*, int64_t, uint32_t, cudaStream_t)
INSTANTIATE_HD_MEMBERS(float, __nv_bfloat16, 128);
#undef INSTANTIATE_HD_MEMBERS

}  // namespace nvidia
}  // namespace llm_kernels
