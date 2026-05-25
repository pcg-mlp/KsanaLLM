/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/store.cuh
 *
 * Copyright 2023-2024 SGLang Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/store.h"

#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/deepseek_v4/fp8_utils.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/vec.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/warp.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

using deepseek_v4::fp8::pack_fp8;
using namespace llm_kernels::utils;

// Compile-time ceil(a / b) using integer math (replaces upstream host::div_ceil
// which is unavailable in this build environment).
template <int64_t A, int64_t B>
inline constexpr int64_t kCeilDiv = (A + B - 1) / B;

// =============================================================================
// fused_store_indexer_cache — kernel
// =============================================================================
struct FusedStoreCacheParam {
  const void* __restrict__ input;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  uint32_t num_tokens;
  int64_t page_stride;
};

template <typename Float, typename IndicesT, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_store_indexer_cache_kernel(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  /// NOTE: 132 = 128 + 4
  // constexpr int64_t kPageBytes = 132 << kPageBits;

  // each warp handles 128 elements, 1 warp, each block handles multiple rows
  const auto& [input, cache, indices, num_tokens, page_stride] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;

  if (global_wid >= num_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  // prefetch the index
  const auto index = static_cast<const IndicesT*>(indices)[global_wid];
  if (index == -1) return;  // TODO(yfnjin): avoid this branch?
  // always load the value from input (don't store if invalid)
  using Float2 = packed_t<Float>;
  using InStorage = AlignedVector<Float2, 2>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;
  const auto elems = static_cast<const InStorage*>(input)[global_tid];
  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);
  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);
  const auto local_max = fmaxf(fmaxf(fabs(x0), fabs(x1)), fmaxf(fabs(y0), fabs(y1)));
  const auto abs_max = warp::reduce_max(local_max);
  // use normal fp32 scale
  const auto scale = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
  const auto inv_scale = 1.0f / scale;
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  const auto page_ptr = pointer::offset(cache, page * /*kPageBytes*/ page_stride);
  const auto value_ptr = pointer::offset(page_ptr, offset * 128);
  const auto scale_ptr = pointer::offset(page_ptr, 128 << kPageBits, offset * 4);
  OutStorage result;
  result[0] = pack_fp8(x0 * inv_scale, x1 * inv_scale);
  result[1] = pack_fp8(y0 * inv_scale, y1 * inv_scale);
  static_cast<OutStorage*>(value_ptr)[lane_id] = result;
  static_cast<float*>(scale_ptr)[0] = scale;

  PDLTriggerSecondary<kUsePDL>();
}

// =============================================================================
// fused_store_indexer_cache — host launcher
// =============================================================================
// Pack and write input tokens into a paged indexer KV cache.
//
// Input  : [num_tokens, 128] Float
// Cache  : paged byte array; each page is 132 * page_size bytes.
// indices: [num_tokens] IndicesT
template <typename Float, typename IndicesT, uint32_t kPageBits>
inline void fused_store_indexer_cache(const void* input, void* cache, const void* indices, uint32_t num_tokens,
                                      int64_t page_stride, cudaStream_t stream) {
  const FusedStoreCacheParam params{input, cache, indices, num_tokens, page_stride};
  constexpr uint32_t kBlockSize = 128;
  const uint32_t num_blocks = device::div_ceil(num_tokens * 32u, kBlockSize);
  constexpr auto kernel = fused_store_indexer_cache_kernel<Float, IndicesT, kPageBits, /*kUsePDL=*/true>;
  LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kBlockSize, 0, stream, params);
}

// =============================================================================
// fused_gather_indexer_cache — kernel
// =============================================================================
struct FusedGatherCacheParam {
  void* __restrict__ k_fp8;                   // output: [total_tokens, 128] FP8
  void* __restrict__ k_scale;                 // output: [total_tokens] Float
  const void* __restrict__ cache;             // paged cache blob
  const void* __restrict__ block_table;       // [batch_size, max_num_blocks_per_seq]
  const size_t* __restrict__ seq_len_offset;  // [batch_size + 1] cumulative lengths (uncompressed)
  uint32_t batch_size;
  uint32_t total_tokens;  // total compressed entries to gather
  int64_t page_stride;    // bytes between consecutive blocks
  int32_t max_num_blocks_per_seq;
  int32_t compress_ratio;
  int32_t gather_offset;  // global compressed entry offset for this chunk
};

template <typename IndicesT, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_gather_indexer_cache_kernel(const __grid_constant__ FusedGatherCacheParam param) {
  using namespace device;

  // Each warp handles one output entry (128 FP8 bytes + 1 float32 scale)
  const auto& [k_fp8, k_scale, cache, block_table, seq_len_offset, batch_size, total_tokens, page_stride,
               max_num_blocks_per_seq, compress_ratio, gather_offset] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;

  if (global_wid >= total_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  // Map local entry to global entry using gather_offset
  const IndicesT global_entry = gather_offset + global_wid;

  // Determine which batch this entry belongs to via linear search on global entry.
  uint32_t batch_idx;
  for (batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    const IndicesT compressed_end = static_cast<IndicesT>(seq_len_offset[batch_idx + 1]) / compress_ratio;
    if (global_entry < compressed_end) {
      break;
    }
  }
  if (batch_idx >= batch_size) [[unlikely]]
    return;

  const IndicesT compressed_start = static_cast<IndicesT>(seq_len_offset[batch_idx]) / compress_ratio;
  const IndicesT entry_offset_in_req = global_entry - compressed_start;

  // Map to physical page and offset within page
  const IndicesT block_idx_in_req = entry_offset_in_req >> kPageBits;
  const IndicesT offset_in_block = entry_offset_in_req & ((1 << kPageBits) - 1);

  // Look up physical block from block_table
  const IndicesT physical_block =
      static_cast<const IndicesT*>(block_table)[batch_idx * max_num_blocks_per_seq + block_idx_in_req];

  // Compute addresses in the split-layout page
  const auto page_ptr = pointer::offset(cache, static_cast<int64_t>(physical_block) * page_stride);
  const auto value_ptr = pointer::offset(page_ptr, offset_in_block * 128);
  const auto scale_ptr = pointer::offset(page_ptr, 128 << kPageBits, offset_in_block * 4);

  // Read and write 128 bytes of FP8 data (32 lanes × 4 bytes each)
  using FP8Storage = AlignedVector<fp8x2_e4m3_t, 2>;  // 2 × fp8x2 = 4 bytes per lane
  const FP8Storage data = static_cast<const FP8Storage*>(value_ptr)[lane_id];
  static_cast<FP8Storage*>(k_fp8)[global_tid] = data;

  // Lane 0 reads and writes the float32 scale
  if (lane_id == 0) {
    static_cast<float*>(k_scale)[global_wid] = *static_cast<const float*>(scale_ptr);
  }

  PDLTriggerSecondary<kUsePDL>();
}

// =============================================================================
// fused_gather_indexer_cache — host launcher
// =============================================================================
// Gather FP8 values and per-token scales from a paged indexer KV cache.
//
// Output : k_fp8   [total_tokens, 128] Fp8E4M3
//          k_scale [total_tokens] Float
// Cache  : paged byte array; each page is 132 * page_size bytes.
// block_table    : [batch_size, max_num_blocks_per_seq] IndicesT
// seq_len_offset : [batch_size + 1] size_t
template <typename Float, typename IndicesT, uint32_t kPageBits>
inline void fused_gather_indexer_cache(void* k_fp8, void* k_scale, const void* cache, const void* block_table,
                                       const size_t* seq_len_offset, uint32_t batch_size, uint32_t total_tokens,
                                       int64_t page_stride, int32_t max_num_blocks_per_seq, int32_t compress_ratio,
                                       int32_t gather_offset, cudaStream_t stream) {
  const FusedGatherCacheParam params{k_fp8,          k_scale,        cache,
                                     block_table,    seq_len_offset, batch_size,
                                     total_tokens,   page_stride,    max_num_blocks_per_seq,
                                     compress_ratio, gather_offset};
  constexpr uint32_t kBlockSize = 128;
  const uint32_t num_blocks = device::div_ceil(total_tokens * 32u, kBlockSize);
  constexpr auto kernel = fused_gather_indexer_cache_kernel<IndicesT, kPageBits, /*kUsePDL=*/true>;
  LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kBlockSize, 0, stream, params);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE_STORE(Float, IndicesT, kPageBits)                                                              \
  template void fused_store_indexer_cache<Float, IndicesT, kPageBits>(const void*, void*, const void*, uint32_t,   \
                                                                      int64_t, cudaStream_t);                      \
  template void fused_gather_indexer_cache<Float, IndicesT, kPageBits>(void*, void*, const void*, const void*,     \
                                                                       const size_t*, uint32_t, uint32_t, int64_t, \
                                                                       int32_t, int32_t, int32_t, cudaStream_t)
INSTANTIATE_STORE(__nv_bfloat16, int32_t, 1);
INSTANTIATE_STORE(__nv_bfloat16, int32_t, 6);
INSTANTIATE_STORE(__nv_bfloat16, int32_t, 8);
#undef INSTANTIATE_STORE

}  // namespace nvidia
}  // namespace llm_kernels
