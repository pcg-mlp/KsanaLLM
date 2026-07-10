/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/c4.cuh
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

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/plan.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/tile.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/type.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/utils.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/vec.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

using namespace llm_kernels::utils;

/// \brief Each thread will handle this many elements (split along head_dim)
constexpr int kC4TileElements = 4;

/// \brief Need to improve register usage to reduce latency
#define C4_KERNEL __global__ __launch_bounds__(128, 4)

// -----------------------------------------------------------------------------
// PageMode
// -----------------------------------------------------------------------------
// Two page-organization strategies of the kv_score_buffer:
//   - Page4Align : each "page" holds compress_ratio(=4) ragged tokens; overlap
//                  tokens live on a separate (previous) page.
//   - RingBuffer : a single 8-slot ring buffer per compress slot.
enum class C4PageMode : int32_t {
  RingBuffer = 8,
  Page4Align = 4,
};

// Layout of the per-batch extra tensor for prefill (4 × int32 per batch).
// write_pages[k] is the page index to write when the token page is k+1 pages
// behind the last page; the last page uses `indices[bid]` directly.
// Supports MTP kv fallback for spans covering up to 3 pages (indices + 2 back).
struct alignas(16) C4IndexBundle {
  IndicesT load_first_page;   // page index to load overlap kv/score from
  IndicesT load_second_page;  // page index to load non-overlap kv/score from
  IndicesT write_pages[2];    // 1 / 2 pages back from current page
};

struct Compress4DecodeParams {
  /**
   * \brief Shape: `[num_indices, 8, head_dim * 4]` \n
   * last dimension layout:
   * | kv overlap | kv | score overlap | score |
   */
  void* __restrict__ kv_score_buffer;
  /** \brief Shape: `[batch_size, head_dim * 4]` */
  const void* __restrict__ kv_score_input;
  /** \brief Shape: `[batch_size, head_dim]` */
  void* __restrict__ kv_compressed_output;
  /** \brief Shape: `[8, head_dim]` (called `ape`) */
  const void* __restrict__ score_bias;
  /** \brief Shape: `[batch_size, ]`*/
  const IndicesT* __restrict__ indices;
  /** \brief Shape: `[batch_size, ]` */
  const IndicesT* __restrict__ seq_lens;
  /** \brief Shape: `[batch_size, 1]` */
  const IndicesT* __restrict__ extra;
  /** \NOTE: `batch_size` <= `num_indices` */
  uint32_t batch_size;
};

struct Compress4PrefillParams {
  /**
   * \brief Shape: `[num_indices, 8, head_dim * 4]` \n
   * last dimension layout:
   * | kv overlap | kv | score overlap | score |
   */
  void* __restrict__ kv_score_buffer;
  /** \brief Shape: `[num_q_tokens, head_dim * 4]` */
  const void* __restrict__ kv_score_input;
  /** \brief Shape: `[num_q_tokens, head_dim]` */
  void* __restrict__ kv_compressed_output;
  /** \brief Shape: `[8, head_dim]` (called `ape`) */
  const void* __restrict__ score_bias;
  /** \brief Shape: `[batch_size, ]`*/
  const IndicesT* __restrict__ indices;
  /** \brief Shape: `[batch_size, 4]` */
  const C4IndexBundle* __restrict__ extra;
  /** \brief The following part is plan info. */

  const PrefillPlan* __restrict__ compress_plan;
  const PrefillPlan* __restrict__ write_plan;
  uint32_t num_compress;
  uint32_t num_write;
};

template <typename T>
SGL_DEVICE void c4_write(T* kv_score_buf, const T* kv_score_src, const int64_t head_dim, const int32_t write_pos) {
  using namespace device;

  using Storage = AlignedVector<T, kC4TileElements>;
  const auto element_size = head_dim * 4;
  const auto gmem = tile::Memory<Storage>::warp();
  kv_score_buf += write_pos * element_size;

  /// NOTE: Layout | [0] = kv overlap | [1] = kv | [2] = score overlap | [3] = score |
  Storage kv_score[4];
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    kv_score[i] = gmem.load(kv_score_src + head_dim * i);
  }
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    gmem.store(kv_score_buf + head_dim * i, kv_score[i]);
  }
}

template <bool kPaged, typename InFloat, typename OutFloat>
SGL_DEVICE void c4_forward(const InFloat* kv_score_buf, const InFloat* kv_score_src, OutFloat* kv_out,
                           const float* score_bias, const int64_t head_dim, const IndicesT seq_len,
                           const int32_t window_len, [[maybe_unused]] const InFloat* kv_score_overlap_buf = nullptr) {
  using namespace device;

  const auto element_size = head_dim * 4;
  const auto score_offset = head_dim * 2;
  const auto overlap_stride = head_dim;

  /// NOTE: part 1: load kv + score
  using StorageIn = AlignedVector<InFloat, kC4TileElements>;
  using StorageBias = AlignedVector<float, kC4TileElements>;
  const auto gmem_in = tile::Memory<StorageIn>::warp();
  const auto gmem_bias = tile::Memory<StorageBias>::warp();
  StorageIn kv[8];
  StorageIn score[8];
  StorageBias bias[8];

#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    bias[i] = gmem_bias.load(score_bias + i * head_dim);
  }

#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    const bool is_overlap = i < 4;
    const InFloat* src;
    if (i < window_len) {
      /// NOTE: `seq_len` must be a multiple of 4 here
      if constexpr (kPaged) {
        const auto kv_score_ptr = is_overlap ? kv_score_overlap_buf : kv_score_buf;
        const int32_t k = i % 4;
        src = kv_score_ptr + k * element_size;
      } else {
        const int32_t k = (seq_len + i) % 8;
        src = kv_score_buf + k * element_size;
      }
    } else {
      /// NOTE: k in [-7, 0]. We'll load from the ragged `kv_score_src`
      const int32_t k = i - 7;
      src = kv_score_src + k * element_size;
    }
    src += (is_overlap ? 0 : overlap_stride);
    kv[i] = gmem_in.load(src);
    score[i] = gmem_in.load(src + score_offset);
  }

  // Edge case: seq_len == 4 means no history, only the first 4 slots exist.
  if (seq_len == 4) {
    [[unlikely]];
    constexpr float kFloatNegInf = -1e9f;
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      kv[i].fill(cast<InFloat>(0.0f));
      score[i].fill(cast<InFloat>(kFloatNegInf));
    }
  }

  /// NOTE: part 2: safe online softmax + weighted sum
  using StorageOut = AlignedVector<OutFloat, kC4TileElements>;
  const auto gmem_out = tile::Memory<StorageOut>::warp();
  StorageOut result;

#pragma unroll
  for (int32_t i = 0; i < kC4TileElements; ++i) {
    float score_fp32[8];

#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      score_fp32[j] = cast<float>(score[j][i]) + bias[j][i];
    }

    float max_value = score_fp32[0];
    float sum_exp_value = 0.0f;

#pragma unroll
    for (int32_t j = 1; j < 8; ++j) {
      const auto fp32_score = score_fp32[j];
      max_value = fmaxf(max_value, fp32_score);
    }

    float sum_product = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const auto fp32_score = score_fp32[j];
      const auto exp_score = expf(fp32_score - max_value);
      sum_product += cast<float>(kv[j][i]) * exp_score;
      sum_exp_value += exp_score;
    }

    result[i] = cast<OutFloat>(sum_product / sum_exp_value);
  }

  gmem_out.store(kv_out, result);
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, C4PageMode kMode>
C4_KERNEL void flash_c4_decode_kernel(const __grid_constant__ Compress4DecodeParams params) {
  using namespace device;

  constexpr int64_t kTileDim = kC4TileElements * kWarpThreads;  // 128
  constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  constexpr int64_t kElementSize = kHeadDim * 4;
  static_assert(kHeadDim % kTileDim == 0, "Head dim must be a multiple of tile dim");

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;
  const uint32_t global_bid = global_wid / kNumSplit;
  const uint32_t global_sid = global_wid % kNumSplit;

  if (global_bid >= params.batch_size) return;

  const int32_t index = params.indices[global_bid];
  const int32_t seq_len = params.seq_lens[global_bid];
  const int64_t split_offset = global_sid * kTileDim;

  // kv score
  auto* kv_score_buffer = static_cast<InFloat*>(params.kv_score_buffer);

  // kv input
  const auto* kv_score_input = static_cast<const InFloat*>(params.kv_score_input);
  const auto* kv_src = kv_score_input + global_bid * kElementSize + split_offset;

  // kv output
  auto* kv_compressed_output = static_cast<OutFloat*>(params.kv_compressed_output);
  auto* kv_out = kv_compressed_output + global_bid * kHeadDim + split_offset;

  const auto* score_bias = static_cast<const float*>(params.score_bias) + split_offset;

  PDLWaitPrimary</*kUsePDL*/ true>();

  /// NOTE: `position` = `seq_len - 1`. To avoid underflow, we use `seq_len + page_size - 1`
  if constexpr (kMode == C4PageMode::Page4Align) {
    const int32_t index_prev = params.extra[global_bid];
    auto* kv_buf = kv_score_buffer + index * (kElementSize * 4) + split_offset;
    c4_write(kv_buf, kv_src, kHeadDim, /*write_pos=*/(seq_len + 3) % 4);
    if (seq_len % 4 == 0) {
      const auto* kv_overlap = kv_buf + (index_prev - index) * (kElementSize * 4);
      c4_forward</*kPaged=*/true>(kv_buf, kv_src, kv_out, score_bias, kHeadDim, seq_len, 8, kv_overlap);
    }
  } else {
    static_assert(kMode == C4PageMode::RingBuffer, "Unsupported PageMode");
    auto* kv_buf = kv_score_buffer + index * (kElementSize * 8) + split_offset;
    c4_write(kv_buf, kv_src, kHeadDim, /*write_pos=*/(seq_len + 7) % 8);
    if (seq_len % 4 == 0) {
      c4_forward</*kPaged=*/false>(kv_buf, kv_src, kv_out, score_bias, kHeadDim, seq_len, /*window_len=*/8);
    }
  }

  PDLTriggerSecondary</*kUsePDL*/ true>();
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, C4PageMode kMode, bool kWrite>
C4_KERNEL void flash_c4_prefill_kernel(const __grid_constant__ Compress4PrefillParams params) {
  using namespace device;

  constexpr int64_t kTileDim = kC4TileElements * kWarpThreads;  // 128
  constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  constexpr int64_t kElementSize = kHeadDim * 4;
  static_assert(kHeadDim % kTileDim == 0, "Head dim must be a multiple of tile dim");

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;
  const uint32_t global_pid = global_wid / kNumSplit;
  const uint32_t global_sid = global_wid % kNumSplit;

  /// NOTE: compiler can optimize this if-else at compile time
  const uint32_t num_plans = kWrite ? params.num_write : params.num_compress;
  const auto* plan_ptr = kWrite ? params.write_plan : params.compress_plan;
  if (global_pid >= num_plans) return;

  const PrefillPlan plan = plan_ptr[global_pid];
  const uint32_t ragged_id = plan.ragged_id;
  const uint32_t global_bid = plan.batch_id;
  const uint32_t position = plan.position;
  const uint32_t window_len = plan.window_len;
  const int64_t split_offset = global_sid * kTileDim;

  // kv score
  auto* kv_score_buffer = static_cast<InFloat*>(params.kv_score_buffer);

  // kv input
  const auto* kv_score_input = static_cast<const InFloat*>(params.kv_score_input);
  const auto* kv_src = kv_score_input + ragged_id * kElementSize + split_offset;

  // kv output
  auto* kv_compressed_output = static_cast<OutFloat*>(params.kv_compressed_output);
  auto* kv_out = kv_compressed_output + ragged_id * kHeadDim + split_offset;

  // Invalid sentinel: produced by the planner to pad for cuda-graph.
  if (ragged_id == 0xFFFFFFFFu) [[unlikely]]
    return;

  const auto* score_bias = static_cast<const float*>(params.score_bias) + split_offset;
  const int32_t seq_len = static_cast<int32_t>(position) + 1;
  const int32_t index = params.indices[global_bid];

  PDLWaitPrimary</*kUsePDL*/ true>();

  if constexpr (kMode == C4PageMode::Page4Align) {
    const int32_t write_second_page = index;
    const C4IndexBundle bundle = params.extra[global_bid];

    if constexpr (kWrite) {
      // For write plans, `window_len` is repurposed by the planner to encode
      // `page_back`: number of pages between the token's page and the last page
      // 0 → last page (`indices[bid]`), k>0 → write_pages[k-1].
      const uint32_t page_back = window_len;
      const int32_t page_index = (page_back == 0u) ? write_second_page : bundle.write_pages[page_back - 1];
      auto* kv_buf = kv_score_buffer + page_index * (kElementSize * 4) + split_offset;
      c4_write(kv_buf, kv_src, kHeadDim, /*write_pos=*/position % 4);
    } else {
      int32_t index_overlap;
      int32_t index_normal;
      if (window_len <= 4u) {
        index_overlap = bundle.load_second_page;
        index_normal = bundle.load_second_page;  // unused when window_len <= 4
      } else {
        index_overlap = bundle.load_first_page;
        index_normal = bundle.load_second_page;
      }
      auto* kv_buf = kv_score_buffer + index_normal * (kElementSize * 4) + split_offset;
      auto* kv_overlap = kv_score_buffer + index_overlap * (kElementSize * 4) + split_offset;
      c4_forward</*kPaged=*/true>(kv_buf, kv_src, kv_out, score_bias, kHeadDim, seq_len,
                                  static_cast<int32_t>(window_len), kv_overlap);
    }
  } else {
    static_assert(kMode == C4PageMode::RingBuffer, "Unsupported PageMode");
    auto* kv_buf = kv_score_buffer + index * (kElementSize * 8) + split_offset;
    if constexpr (kWrite) {
      c4_write(kv_buf, kv_src, kHeadDim, /*write_pos=*/position % 8);
    } else {
      c4_forward</*kPaged=*/false>(kv_buf, kv_src, kv_out, score_bias, kHeadDim, seq_len,
                                   static_cast<int32_t>(window_len));
    }
  }

  PDLTriggerSecondary</*kUsePDL*/ true>();
}

// Host launcher helpers
namespace {

constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kWarpsPerBlock = kBlockSize / device::kWarpThreads;  // 4

inline uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePage4>
inline void launch_decode(const Compress4DecodeParams& params, cudaStream_t stream) {
  constexpr uint32_t kNumSplit = kHeadDim / (kC4TileElements * device::kWarpThreads);
  const uint32_t num_blocks = div_ceil(params.batch_size * kNumSplit, kWarpsPerBlock);
  constexpr auto kMode = kUsePage4 ? C4PageMode::Page4Align : C4PageMode::RingBuffer;

  auto kernel = flash_c4_decode_kernel<kHeadDim, InFloat, OutFloat, kMode>;
  LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kBlockSize, 0, stream, params);
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePage4>
inline void launch_prefill(const Compress4PrefillParams& params, cudaStream_t stream) {
  constexpr uint32_t kNumSplit = kHeadDim / (kC4TileElements * device::kWarpThreads);
  constexpr auto kMode = kUsePage4 ? C4PageMode::Page4Align : C4PageMode::RingBuffer;

  if (const uint32_t num_c_blocks = div_ceil(params.num_compress * kNumSplit, kWarpsPerBlock)) {
    constexpr auto kernel = flash_c4_prefill_kernel<kHeadDim, InFloat, OutFloat, kMode, /*kWrite=*/false>;
    LAUNCH_KERNEL_WITH_PDL(kernel, num_c_blocks, kBlockSize, 0, stream, params);
  }
  if (const uint32_t num_w_blocks = div_ceil(params.num_write * kNumSplit, kWarpsPerBlock)) {
    constexpr auto kernel = flash_c4_prefill_kernel<kHeadDim, InFloat, OutFloat, kMode, /*kWrite=*/true>;
    LAUNCH_KERNEL_WITH_PDL(kernel, num_w_blocks, kBlockSize, 0, stream, params);
  }
}

}  // namespace

// Public API
template <typename InFloat, typename OutFloat, int64_t kHeadDim>
inline void flash_c4_decode(void* kv_score_buffer, const void* kv_score_input, void* kv_compressed_output,
                            const void* score_bias, const IndicesT* indices, const IndicesT* seq_lens,
                            const IndicesT* extra, uint32_t batch_size, cudaStream_t stream) {
  Compress4DecodeParams params{};
  params.kv_score_buffer = kv_score_buffer;
  params.kv_score_input = kv_score_input;
  params.kv_compressed_output = kv_compressed_output;
  params.score_bias = score_bias;
  params.indices = indices;
  params.seq_lens = seq_lens;
  params.extra = extra;
  params.batch_size = batch_size;

  if (extra != nullptr) {
    launch_decode<kHeadDim, InFloat, OutFloat, /*kUsePage4*/ true>(params, stream);
  } else {
    launch_decode<kHeadDim, InFloat, OutFloat, /*kUsePage4*/ false>(params, stream);
  }
}

template <typename InFloat, typename OutFloat, int64_t kHeadDim>
inline void flash_c4_prefill(void* kv_score_buffer, const void* kv_score_input, void* kv_compressed_output,
                             const void* score_bias, const IndicesT* indices, const PrefillPlan* compress_plan,
                             const PrefillPlan* write_plan, const void* extra, uint32_t num_compress,
                             uint32_t num_write, cudaStream_t stream) {
  Compress4PrefillParams params{};
  params.kv_score_buffer = kv_score_buffer;
  params.kv_score_input = kv_score_input;
  params.kv_compressed_output = kv_compressed_output;
  params.score_bias = score_bias;
  params.indices = indices;
  params.extra = static_cast<const C4IndexBundle*>(extra);
  params.compress_plan = compress_plan;
  params.write_plan = write_plan;
  params.num_compress = num_compress;
  params.num_write = num_write;

  if (extra != nullptr) {
    launch_prefill<kHeadDim, InFloat, OutFloat, /*kUsePage4*/ true>(params, stream);
  } else {
    launch_prefill<kHeadDim, InFloat, OutFloat, /*kUsePage4*/ false>(params, stream);
  }
}

#define INSTANTIATE_C4(InT, OutT, kHeadDim)                                                                          \
  template void flash_c4_decode<InT, OutT, kHeadDim>(void*, const void*, void*, const void*, const IndicesT*,        \
                                                     const IndicesT*, const IndicesT*, uint32_t, cudaStream_t);      \
  template void flash_c4_prefill<InT, OutT, kHeadDim>(void*, const void*, void*, const void*, const IndicesT*,       \
                                                      const PrefillPlan*, const PrefillPlan*, const void*, uint32_t, \
                                                      uint32_t, cudaStream_t);

INSTANTIATE_C4(float, __nv_bfloat16, 128)
INSTANTIATE_C4(float, __nv_bfloat16, 512)
INSTANTIATE_C4(__nv_bfloat16, __nv_bfloat16, 128)
INSTANTIATE_C4(__nv_bfloat16, __nv_bfloat16, 512)

#undef INSTANTIATE_C4

}  // namespace nvidia
}  // namespace llm_kernels
