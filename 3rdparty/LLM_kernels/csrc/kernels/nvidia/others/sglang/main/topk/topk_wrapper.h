/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

// Templated, host-only C++ wrapper around the CUDA kernels defined in
// topk.cuh / topk_1024.cuh / topk_v2.cuh.
//
// This header can be included from both .cu and .cpp translation units.
// The kernel template definitions live in topk_wrapper.cu, and only
// explicitly instantiated (InT, OutT) combinations are available at link
// time. `InT` is the score dtype (always float for the current kernels);
// `OutT` is the index dtype (always int32_t for the current kernels); the
// template arguments are kept to match the skill's wrapper convention and to
// leave room for future dtype variants (e.g. half-precision scores).
//
// Usage from a .cpp caller:
//   using Wrap = llm_kernels::nvidia::topk::TopKWrapper<float, int32_t>;
//   Wrap::TransformV2(/* raw device ptrs */, stream);

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {
namespace topk {

// -----------------------------------------------------------------------------
// TopKWrapper<InT, OutT>
//
// Supported instantiations (must match the explicit instantiations at the
// bottom of topk_wrapper.cu):
//   <float, int32_t>
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
class TopKWrapper {
 public:
  // ---------------------------------------------------------------------------
  // TransformV2 — multi-path TopK using the topk_v2 algorithm (Sm=Register,
  // Medium=Streaming, Large=Cluster). Dispatches automatically based on
  // max_seq_len and batch_size. Raw device pointers are used throughout.
  //
  //   seq_lens           [batch_size]                   uint32_t
  //   scores             [batch_size * score_stride]    InT (float)
  //   page_table         [batch_size * page_table_stride] int32_t
  //   page_indices_out   [batch_size * K]               OutT (int32_t)
  //   workspace          scratch; >= (batch_size * workspace_stride_bytes)
  //   metadata           scratch; >= ((batch_size+1) * 4 * sizeof(int32_t))
  //   workspace_stride_bytes  bytes per batch row in workspace
  //   batch_size         number of sequences
  //   page_bits          log2(page_size)
  //   max_seq_len        maximum sequence length across the batch
  //   static_cluster_threshold  0 = auto-select
  //   stream             CUDA stream
  // ---------------------------------------------------------------------------
  static void TransformV2(const void* seq_lens, const void* scores, const void* page_table, void* page_indices_out,
                          void* workspace, void* metadata, int64_t score_stride, int64_t page_table_stride,
                          int64_t workspace_stride_bytes, uint32_t batch_size, uint32_t page_bits, uint32_t max_seq_len,
                          uint32_t static_cluster_threshold, const void* batch_indices, uint32_t extend_len,
                          uint32_t k, cudaStream_t stream);
};

}  // namespace topk
}  // namespace nvidia
}  // namespace llm_kernels
