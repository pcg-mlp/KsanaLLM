/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

// Host-only C++ declarations for the top-K transform kernels ported from
// SGLang DeepSeek-V4 (`topk.cuh` / `topk_1024.cuh`).
//
// This header is safe to include from both `.cu` and `.cpp` translation units.
// The full CUDA device kernels live in `topk.cuh` / `topk_1024.cuh`; here we
// only expose the host-side launchers, which operate on raw device pointers
// and accept a `cudaStream_t` — no sgl_kernel / tvm / dlpack dependency.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {
namespace topk {

// -----------------------------------------------------------------------------
// launch_topk512
// -----------------------------------------------------------------------------
// Select the top-512 scores per batch, then materialise the page indices using
// the supplied page_table. Equivalent (numerically and layout-wise) to calling
// SGLang's `TopK512Kernel<kUsePDL>::transform(...)` on the same tensors.
//
//   scores           : [batch_size, score_stride]  float   (device)
//                      per-batch row b uses the [row_starts[b], row_ends[b]) slice.
//   row_starts       : [num_works]                int32_t  (device)
//   row_ends         : [num_works]                int32_t  (device)
//   page_table       : [num_batches, page_table_stride] int32_t (device)
//   page_indices_out : [num_works, 512]           int32_t  (device, contiguous)
//   raw_indices_out  : [num_works, 512]           int32_t  (device, contiguous)
//                      Optional; pass nullptr to skip. When non-null, stores
//                      the raw absolute positions before page transform.
//   score_stride     : row stride (in elements) of `scores`
//   page_table_stride: row stride (in elements) of `page_table`
//   num_works        : total number of work items (grid size).
//   page_size        : must be a power of two (checked at runtime).
//   extend_lens      : [num_batches] int32_t (device), work count per batch (prefill).
//                      Pass nullptr for decode path.
//   extend_len       : scalar, uniform work count per batch (decode).
//                      Ignored when extend_lens != nullptr.
//   num_batches      : total number of batches.
//   stream           : CUDA stream on which to launch.
//
// Throws std::invalid_argument if `page_size` is not a power of two.
void launch_topk512(const float* scores, const int32_t* row_starts, const int32_t* row_ends, const int32_t* page_table,
                    int32_t* page_indices_out, int32_t* raw_indices_out, int64_t score_stride,
                    int64_t page_table_stride, uint32_t num_works, uint32_t page_bits, const int32_t* extend_lens,
                    uint32_t extend_len, uint32_t num_batches, cudaStream_t stream);

// -----------------------------------------------------------------------------
// launch_topk1024
// -----------------------------------------------------------------------------
// Same as `launch_topk512` but returns the top-1024 entries per batch; the
// last dim of both output tensors must therefore be 1024.
void launch_topk1024(const float* scores, const int32_t* row_starts, const int32_t* row_ends, const int32_t* page_table,
                     int32_t* page_indices_out, int32_t* raw_indices_out, int64_t score_stride,
                     int64_t page_table_stride, uint32_t num_works, uint32_t page_bits, const int32_t* extend_lens,
                     uint32_t extend_len, uint32_t num_batches, cudaStream_t stream);

// -----------------------------------------------------------------------------
// Compile-time constants exposed for callers that need to size output tensors.
// -----------------------------------------------------------------------------
inline constexpr uint32_t kTopK512 = 512;
inline constexpr uint32_t kTopK1024 = 1024;

}  // namespace topk
}  // namespace nvidia
}  // namespace llm_kernels
