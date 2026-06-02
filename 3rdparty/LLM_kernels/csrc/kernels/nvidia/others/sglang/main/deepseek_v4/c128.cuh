/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/c128.cuh
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
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/warp.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

using namespace llm_kernels::utils;

/// \brief Each thread will handle this many elements (split along head_dim)
constexpr int32_t kTileElements = 2;
/// \brief Each warp will handle this many elements (split along 128)
constexpr int32_t kElementsPerWarp = 8;
constexpr uint32_t kNumWarps = 128 / kElementsPerWarp;
constexpr uint32_t kBlockSize = device::kWarpThreads * kNumWarps;

/// \brief Need to reduce register usage to increase occupancy
#define C128_KERNEL __global__ __launch_bounds__(kBlockSize, 2)

struct Compress128DecodeParams {
  /**
   * \brief Shape: `[num_indices, 128, head_dim * 2]` \n
   * last dimension layout:
   * | kv current | score current |
   */
  void* __restrict__ kv_score_buffer;
  /** \brief Shape: `[batch_size, head_dim * 2]` */
  const void* __restrict__ kv_score_input;
  /** \brief Shape: `[batch_size, head_dim]` */
  void* __restrict__ kv_compressed_output;
  /** \brief Shape: `[128, head_dim]` (called `ape`) */
  const void* __restrict__ score_bias;
  /** \brief Shape: `[batch_size, ]`*/
  const IndicesT* __restrict__ indices;
  /** \brief Shape: `[batch_size, ]` */
  const IndicesT* __restrict__ seq_lens;
  /** \NOTE: `batch_size` <= `num_indices` */
  uint32_t batch_size;
};

struct Compress128PrefillParams {
  /**
   * \brief Shape: `[num_indices, 128, head_dim * 2]` \n
   * last dimension layout:
   * | kv current | score current |
   */
  void* __restrict__ kv_score_buffer;
  /** \brief Shape: `[batch_size, head_dim * 2]` */
  const void* __restrict__ kv_score_input;
  /** \brief Shape: `[batch_size, head_dim]` */
  void* __restrict__ kv_compressed_output;
  /** \brief Shape: `[128, head_dim]` (called `ape`) */
  const void* __restrict__ score_bias;
  /** \brief Shape: `[batch_size, ]`*/
  const IndicesT* __restrict__ indices;
  /** \brief Shape: `[batch_size, ]`*/
  const IndicesT* __restrict__ load_indices;
  /** \brief The following part is plan info. */

  const PrefillPlan* __restrict__ compress_plan;
  const PrefillPlan* __restrict__ write_plan;

  uint32_t num_compress;
  uint32_t num_write;
};

struct Compress128SharedBuffer {
  using Storage = device::AlignedVector<float, kTileElements>;
  Storage data[kNumWarps][device::kWarpThreads + 1];  // padding to avoid bank conflict
  SGL_DEVICE Storage& operator()(uint32_t warp_id, uint32_t lane_id) { return data[warp_id][lane_id]; }
  SGL_DEVICE float& operator()(uint32_t warp_id, uint32_t lane_id, uint32_t tile_id) {
    return data[warp_id][lane_id][tile_id];
  }
};

template <typename T>
SGL_DEVICE void c128_write(T* kv_score_buf,  //
                           const T* kv_score_src, const int64_t head_dim, const int32_t write_pos,
                           const uint32_t lane_id) {
  using namespace device;

  using Storage = AlignedVector<T, kTileElements>;
  const auto element_size = head_dim * 2;
  const auto gmem = tile::Memory<Storage>{lane_id, kWarpThreads};
  kv_score_buf += write_pos * element_size;

  /// NOTE: Layout | [0] = kv | [1] = score |
  Storage kv_score[2];
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    kv_score[i] = gmem.load(kv_score_src + head_dim * i);
  }
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    gmem.store(kv_score_buf + head_dim * i, kv_score[i]);
  }
}

template <typename InFloat, typename OutFloat>
SGL_DEVICE void c128_forward(const InFloat* kv_score_buf, const InFloat* kv_score_src, OutFloat* kv_out,
                             const float* score_bias, const int64_t head_dim, const int32_t window_len,
                             const uint32_t warp_id, const uint32_t lane_id) {
  using namespace device;

  const auto element_size = head_dim * 2;
  const auto score_offset = head_dim;

  /// NOTE: part 1: load kv + score
  using StorageIn = AlignedVector<InFloat, kTileElements>;
  using StorageBias = AlignedVector<float, kTileElements>;
  const auto gmem_in = tile::Memory<StorageIn>{lane_id, kWarpThreads};
  const auto gmem_bias = tile::Memory<StorageBias>{lane_id, kWarpThreads};
  StorageIn kv[kElementsPerWarp];
  StorageIn score[kElementsPerWarp];
  StorageBias bias[kElementsPerWarp];
  const int32_t warp_offset = warp_id * kElementsPerWarp;

#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    const int32_t j = i + warp_offset;
    bias[i] = gmem_bias.load(score_bias + j * head_dim);
  }

#pragma unroll
  for (int32_t i = 0; i < kElementsPerWarp; ++i) {
    const int32_t j = i + warp_offset;
    const InFloat* src;
    __builtin_assume(j < 128);
    if (j < window_len) {
      src = kv_score_buf + j * element_size;
    } else {
      /// NOTE: k in [-127, 0]. We'll load from the ragged `kv_score_src`
      const int32_t k = j - 127;
      src = kv_score_src + k * element_size;
    }
    kv[i] = gmem_in.load(src);
    score[i] = gmem_in.load(src + score_offset);
  }

  /// NOTE: part 2: safe online softmax + weighted sum
  using TmpStorage = typename Compress128SharedBuffer::Storage;
  __shared__ Compress128SharedBuffer s_local_val_max;
  __shared__ Compress128SharedBuffer s_local_exp_sum;
  __shared__ Compress128SharedBuffer s_local_product;

  TmpStorage tmp_val_max;
  TmpStorage tmp_exp_sum;
  TmpStorage tmp_product;

#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    float score_fp32[kElementsPerWarp];

#pragma unroll
    for (int32_t j = 0; j < kElementsPerWarp; ++j) {
      score_fp32[j] = cast<float>(score[j][i]) + bias[j][i];
    }

    float max_value = score_fp32[0];
    float sum_exp_value = 0.0f;

#pragma unroll
    for (int32_t j = 1; j < kElementsPerWarp; ++j) {
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

    tmp_val_max[i] = max_value;
    tmp_exp_sum[i] = sum_exp_value;
    tmp_product[i] = sum_product;
  }

  // naturally aligned, so no bank conflict
  s_local_val_max(warp_id, lane_id) = tmp_val_max;
  s_local_exp_sum(warp_id, lane_id) = tmp_exp_sum;
  s_local_product(warp_id, lane_id) = tmp_product;

  __syncthreads();

  /// NOTE: part 3: online softmax
  /// NOTE: We have `kTileElements * kWarpThreads * kNumWarps` values to reduce
  /// each reduce will consume `kNumWarps` threads (use partial warp reduction)
  constexpr uint32_t kReductionCount = kTileElements * kWarpThreads * kNumWarps;
  constexpr uint32_t kIteration = kReductionCount / kBlockSize;

#pragma unroll
  for (uint32_t i = 0; i < kIteration; ++i) {
    /// NOTE: Range `[0, kTileElements * kWarpThreads * kNumWarps)`
    const uint32_t j = i * kBlockSize + warp_id * kWarpThreads + lane_id;
    /// NOTE: Range `[0, kNumWarps)`
    const uint32_t local_warp_id = j % kNumWarps;
    /// NOTE: Range `[0, kTileElements * kWarpThreads)`
    const uint32_t local_elem_id = j / kNumWarps;
    /// NOTE: Range `[0, kTileElements)`
    const uint32_t local_tile_id = local_elem_id % kTileElements;
    /// NOTE: Range `[0, kWarpThreads)`
    const uint32_t local_lane_id = local_elem_id / kTileElements;
    /// NOTE: each warp will access the whole tile (all `kTileElements`)
    /// and for different lanes, the memory access only differ in `local_warp_id`
    /// so there's no bank conflict in shared memory access.
    static_assert(kTileElements * kNumWarps == kWarpThreads, "TODO: support other configs");
    const auto local_val_max = s_local_val_max(local_warp_id, local_lane_id, local_tile_id);
    const auto local_exp_sum = s_local_exp_sum(local_warp_id, local_lane_id, local_tile_id);
    const auto local_product = s_local_product(local_warp_id, local_lane_id, local_tile_id);
    const auto global_val_max = warp::reduce_max<kNumWarps>(local_val_max);
    const auto rescale = expf(local_val_max - global_val_max);
    const auto global_exp_sum = warp::reduce_sum<kNumWarps>(local_exp_sum * rescale);
    const auto final_scale = rescale / global_exp_sum;
    const auto global_product = warp::reduce_sum<kNumWarps>(local_product * final_scale);
    kv_out[local_elem_id] = cast<OutFloat>(global_product);
  }
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat>
C128_KERNEL void flash_c128_decode_kernel(const __grid_constant__ Compress128DecodeParams params) {
  using namespace device;

  constexpr int64_t kTileDim = kTileElements * kWarpThreads;  // 64
  constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  constexpr int64_t kElementSize = kHeadDim * 2;
  static_assert(kHeadDim % kTileDim == 0, "Head dim must be multiple of tile dim");

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;

  const uint32_t global_bid = blockIdx.x / kNumSplit;  // batch id
  const uint32_t global_sid = blockIdx.x % kNumSplit;  // split id
  if (global_bid >= params.batch_size) return;

  const int32_t index = params.indices[global_bid];
  const int32_t seq_len = params.seq_lens[global_bid];
  const int64_t split_offset = global_sid * kTileDim;

  // kv score
  const auto kv_buf = static_cast<InFloat*>(params.kv_score_buffer) + index * (kElementSize * 128) + split_offset;

  // kv input
  const auto kv_src = static_cast<const InFloat*>(params.kv_score_input) + global_bid * kElementSize + split_offset;

  // kv output
  const auto kv_out = static_cast<OutFloat*>(params.kv_compressed_output) + global_bid * kHeadDim + split_offset;

  // score bias (ape) — always FP32
  const auto score_bias = static_cast<const float*>(params.score_bias) + split_offset;

  PDLWaitPrimary</*kUsePDL*/ true>();

  /// NOTE: the write must be visible to the subsequent c128_forward,
  /// so only the last warp can write to HBM
  /// In addition, `position` = `seq_len - 1`. To avoid underflow, we use `seq_len + 127`
  if (warp_id == kNumWarps - 1) {
    c128_write(kv_buf, kv_src, kHeadDim, /*write_pos=*/(seq_len + 127) % 128, lane_id);
  }
  if (seq_len % 128 == 0) {
    c128_forward(kv_buf, kv_src, kv_out, score_bias, kHeadDim, /*window_len=*/128, warp_id, lane_id);
  }

  PDLTriggerSecondary</*kUsePDL*/ true>();
}

// compress kernel (kWrite=false) and write kernel (kWrite=true)
template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kWrite>
C128_KERNEL void flash_c128_prefill_kernel(const __grid_constant__ Compress128PrefillParams params) {
  using namespace device;

  constexpr int64_t kTileDim = kTileElements * kWarpThreads;  // 64
  constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  constexpr int64_t kElementSize = kHeadDim * 2;
  static_assert(kHeadDim % kTileDim == 0, "Head dim must be multiple of tile dim");

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;

  uint32_t global_id;
  if constexpr (kWrite) {
    // for write kernel, we use global warp_id to dispatch work
    global_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpThreads;
  } else {
    // for compress kernel, we use block id to dispatch work
    global_id = blockIdx.x;  // block id
  }
  const uint32_t global_pid = global_id / kNumSplit;  // plan id
  const uint32_t global_sid = global_id % kNumSplit;  // split id

  /// NOTE: compiler can optimize this if-else at compile time
  const auto num_plans = kWrite ? params.num_write : params.num_compress;
  const auto plan_ptr = kWrite ? params.write_plan : params.compress_plan;
  if (global_pid >= num_plans) return;

  const auto& [ragged_id, global_bid, position, window_len] = plan_ptr[global_pid];
  const auto indices_ptr = kWrite ? params.indices : params.load_indices;

  const int64_t split_offset = global_sid * kTileDim;

  // kv input
  const auto kv_src = static_cast<const InFloat*>(params.kv_score_input) + ragged_id * kElementSize + split_offset;

  // kv output
  const auto kv_out = static_cast<OutFloat*>(params.kv_compressed_output) + ragged_id * kHeadDim + split_offset;

  // score bias (ape) — always FP32
  const auto score_bias = static_cast<const float*>(params.score_bias) + split_offset;

  if (ragged_id == 0xFFFFFFFF) [[unlikely]]
    return;

  const int32_t index = indices_ptr[global_bid];
  // kv score
  const auto kv_buf = static_cast<InFloat*>(params.kv_score_buffer) + index * (kElementSize * 128) + split_offset;

  PDLWaitPrimary</*kUsePDL*/ true>();

  // only responsible for the compress part
  if constexpr (kWrite) {
    c128_write(kv_buf, kv_src, kHeadDim, /*write_pos=*/position % 128, lane_id);
  } else {
    c128_forward(kv_buf, kv_src, kv_out, score_bias, kHeadDim, window_len, warp_id, lane_id);
  }

  PDLTriggerSecondary</*kUsePDL*/ true>();
}

// Host launchers
template <typename InFloat, typename OutFloat, int64_t kHeadDim>
inline void flash_c128_decode(void* kv_score_buffer, const void* kv_score_input, void* kv_compressed_output,
                              const void* score_bias, const IndicesT* indices, const IndicesT* seq_lens,
                              uint32_t batch_size, cudaStream_t stream) {
  constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;  // 64
  constexpr int64_t kNumSplit = kHeadDim / kTileDim;

  const Compress128DecodeParams params{kv_score_buffer, kv_score_input, kv_compressed_output, score_bias, indices,
                                       seq_lens,        batch_size};

  const uint32_t num_blocks = batch_size * static_cast<uint32_t>(kNumSplit);
  constexpr auto kernel = flash_c128_decode_kernel<kHeadDim, InFloat, OutFloat>;
  LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kBlockSize, 0, stream, params);
}

template <typename InFloat, typename OutFloat, int64_t kHeadDim>
inline void flash_c128_prefill(void* kv_score_buffer, const void* kv_score_input, void* kv_compressed_output,
                               const void* score_bias, const IndicesT* indices, const IndicesT* load_indices,
                               const PrefillPlan* compress_plan, const PrefillPlan* write_plan, uint32_t num_compress,
                               uint32_t num_write, cudaStream_t stream) {
  constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;  // 64
  constexpr int64_t kNumSplit = kHeadDim / kTileDim;
  constexpr uint32_t kWriteBlockSize = 128u;
  constexpr uint32_t kWarpsPerWriteBlock = kWriteBlockSize / device::kWarpThreads;

  const Compress128PrefillParams params{kv_score_buffer, kv_score_input, kv_compressed_output, score_bias,   indices,
                                        load_indices,    compress_plan,  write_plan,           num_compress, num_write};

  if (num_compress > 0u) {
    const uint32_t num_c_blocks = num_compress * static_cast<uint32_t>(kNumSplit);
    constexpr auto kernel = flash_c128_prefill_kernel<kHeadDim, InFloat, OutFloat, /*kWrite=*/false>;
    LAUNCH_KERNEL_WITH_PDL(kernel, num_c_blocks, kBlockSize, 0, stream, params);
  }
  if (num_write > 0u) {
    const uint32_t total_write_warps = num_write * static_cast<uint32_t>(kNumSplit);
    const uint32_t num_w_blocks = (total_write_warps + kWarpsPerWriteBlock - 1u) / kWarpsPerWriteBlock;
    constexpr auto kernel = flash_c128_prefill_kernel<kHeadDim, InFloat, OutFloat, /*kWrite=*/true>;
    LAUNCH_KERNEL_WITH_PDL(kernel, num_w_blocks, kWriteBlockSize, 0, stream, params);
  }
}

#define INSTANTIATE_C128(InFloat, OutFloat, kHeadDim)                                                                 \
  template void flash_c128_decode<InFloat, OutFloat, kHeadDim>(                                                       \
      void*, const void*, void*, const void*, const IndicesT*, const IndicesT*, uint32_t, cudaStream_t);              \
  template void flash_c128_prefill<InFloat, OutFloat, kHeadDim>(void*, const void*, void*, const void*,               \
                                                                const IndicesT*, const IndicesT*, const PrefillPlan*, \
                                                                const PrefillPlan*, uint32_t, uint32_t, cudaStream_t);

INSTANTIATE_C128(float, __nv_bfloat16, 128)
INSTANTIATE_C128(float, __nv_bfloat16, 512)
INSTANTIATE_C128(__nv_bfloat16, __nv_bfloat16, 128)
INSTANTIATE_C128(__nv_bfloat16, __nv_bfloat16, 512)

#undef INSTANTIATE_C128

}  // namespace nvidia
}  // namespace llm_kernels
