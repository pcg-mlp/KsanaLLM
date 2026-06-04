/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

// Template-based dispatch layer: every method on TopKWrapper<InT, OutT>
// forwards to the underlying CUDA launcher with matching dtypes. The bottom
// of this file explicitly instantiates the supported (InT, OutT) combination;
// only those instantiations are available at link time.
//
// Current kernels only support (float scores, int32_t indices). We still keep
// the (InT, OutT) template parameters to leave the door open for extending to
// half-precision scores (e.g. __half / __nv_bfloat16) without changing the
// API surface on the .cpp layer side.

#include "csrc/kernels/nvidia/others/sglang/main/topk/topk_wrapper.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#include "csrc/kernels/nvidia/others/sglang/main/topk/topk_v2.cuh"

namespace llm_kernels {
namespace nvidia {
namespace topk {

// =============================================================================
// TransformV2
// =============================================================================
template <typename InT, typename OutT>
void TopKWrapper<InT, OutT>::TransformV2(const void* seq_lens, const void* scores, const void* page_table,
                                         void* page_indices_out, void* workspace, void* metadata, int64_t score_stride,
                                         int64_t page_table_stride, int64_t workspace_stride_bytes, uint32_t batch_size,
                                         uint32_t page_bits, uint32_t max_seq_len, uint32_t static_cluster_threshold,
                                         const void* batch_indices, uint32_t extend_len, uint32_t k,
                                         cudaStream_t stream) {
  static_assert(std::is_same_v<InT, float>, "TopKWrapper currently only supports InT == float");
  static_assert(std::is_same_v<OutT, int32_t>, "TopKWrapper currently only supports OutT == int32_t");

  // Helper: dispatch launch_topk_v2 with compile-time K_VAL
  auto dispatch = [&]<uint32_t K_VAL>() {
    launch_topk_v2<K_VAL>(
        reinterpret_cast<const uint32_t*>(seq_lens), reinterpret_cast<const float*>(scores),
        reinterpret_cast<const int32_t*>(page_table), reinterpret_cast<int32_t*>(page_indices_out),
        score_stride, page_table_stride, reinterpret_cast<uint8_t*>(workspace),
        reinterpret_cast<Metadata*>(metadata), workspace_stride_bytes, batch_size, page_bits,
        max_seq_len, static_cluster_threshold, reinterpret_cast<const int32_t*>(batch_indices), extend_len, stream);
  };

  // Supported K values: 512, 1024 (V2 kernel supports up to kMaxTopK=1024)
  if (k == 1024) {
    dispatch.template operator()<1024>();
  } else {
    dispatch.template operator()<512>();
  }
}

// =============================================================================
// Explicit instantiations — keep in sync with the header doc.
// =============================================================================
template class TopKWrapper<float, int32_t>;

}  // namespace topk
}  // namespace nvidia
}  // namespace llm_kernels
