/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/4807f6c5220c8ac53e395d6d2ed493cda241bc8e/python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh
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

// Ksana-local port of topk_v2.cuh.
//
// Modifications vs. the original:
//   * Replaced all sgl_kernel / TVM / dlpack includes with repo-relative compat headers.
//   * Replaced anonymous `namespace {}` with `namespace llm_kernels::nvidia::topk {}`.
//   * setup_kernel_smem_once: removed host::DebugInfo parameter; replaced std::bit_cast
//     with reinterpret_cast; removed host::RuntimeDeviceCheck call.
//   * topk_plan: replaced std::size(kCandidates) with sizeof(kCandidates)/sizeof(kCandidates[0]).
//   * Removed #include <iterator> (no longer needed).
//   * Stripped CombinedTopKKernel struct (TVM-dependent) entirely.
//   * Added launch_topk_v2() plain-CUDA host launcher as replacement.

#pragma once

#include "sgl_kernel/deepseek_v4/topk/cluster.cuh"
#include "sgl_kernel/deepseek_v4/topk/common.cuh"
#include "sgl_kernel/deepseek_v4/topk/register.cuh"
#include "sgl_kernel/deepseek_v4/topk/streaming.cuh"
#include "sgl_kernel/deepseek_v4/topk/utils_compat.cuh"

#include "csrc/utils/nvidia/cuda_utils.h"

#include <cooperative_groups.h>
#include <cfloat>
#include <cstdint>

namespace llm_kernels::nvidia::topk {

// ---------------------------------------------------------------------------
// One-shot helper: set the dynamic-SMEM limit for a kernel (idempotent).
// ---------------------------------------------------------------------------
template <auto* f, size_t kMaxDynamicSMEM>
void setup_kernel_smem_once() {
  [[maybe_unused]] static const auto result = [] {
    const auto fptr = reinterpret_cast<const void*>(f);
    return ::cudaFuncSetAttribute(fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
  }();
}

namespace impl = ::device::top512;

// Template aliases parameterized on K_VAL (supported: 512, 1024)
template <uint32_t K_VAL>
using Large_t = impl::ClusterTopK<K_VAL>;
template <uint32_t K_VAL>
using Medium_t = impl::StreamingTopK<K_VAL>;
template <uint32_t K_VAL>
using Small_t = impl::RegisterTopK<K_VAL>;

// Metadata layout is identical for all K values (does not depend on K).
using Metadata = typename Large_t<512>::Metadata;

constexpr uint32_t kBlockSize = impl::kBlockSize;
constexpr uint32_t kNumClusters = 15;  // based on hardware limits
template <uint32_t K_VAL>
inline constexpr uint32_t kClusterSize_v = Large_t<K_VAL>::kClusterSize;
template <uint32_t K_VAL>
inline constexpr uint32_t kMax2PassLength_v = Small_t<K_VAL>::kMax2PassLength;
template <uint32_t K_VAL>
inline constexpr uint32_t kMaxSupportedLength_v = Large_t<K_VAL>::kMaxLength;

/// Common metadata lives at metadata[0] (first row of the [batch_size+1, 4] tensor).
/// Per-item metadata starts at metadata[1..batch_size]. The plan kernel writes both.
struct alignas(16) GlobalMetadata {
  uint32_t cluster_threshold;  // decided per-batch in plan kernel
  uint32_t num_cluster_items;  // N = number of items routed to the cluster path
  uint32_t reserved[2];
};
static_assert(sizeof(GlobalMetadata) == sizeof(Metadata), "layout: row 0 must occupy one Metadata-sized slot");

// Cluster size is the same (8) for all K values
constexpr uint32_t kClusterSize = 8;

// optimize occupancy for prefill
#define SMALL_TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, 2)
// cluster at y dim — requires SM 90+ (Hopper); stripped for older architectures.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#  define LARGE_CLUSTER __cluster_dims__(1, kClusterSize, 1)
#else
#  define LARGE_CLUSTER /* no cluster support */
#endif
// stage-1 is persistent cluster, and shared memory usage is huge (can not 2)
#define LARGE_TOPK_STAGE_1 __global__ __launch_bounds__(kBlockSize, 1) LARGE_CLUSTER
// stage-2 is non-persistent non-cluster, with less shared memory and higher occupancy
#define LARGE_TOPK_STAGE_2 __global__ __launch_bounds__(kBlockSize, 2)
// fused into 1 stage when batch-size <= kNumPersistentClusters
#define FUSED_COMBINE_KERNEL __global__ __launch_bounds__(kBlockSize, 1) LARGE_CLUSTER
// plan runs once as a single block before the combine kernels
#define PLAN_KERNEL __global__ __launch_bounds__(kBlockSize, 1)

template <uint32_t K_VAL>
struct TopKParams {
  const uint32_t* __restrict__ seq_lens;
  const float* __restrict__ scores;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  uint8_t* __restrict__ workspace;  // [batch, kWorkspaceBytes] -- internally allocated
  /// Pointer to the full metadata tensor: metadata[0] is GlobalMetadata, metadata[1..]
  /// are per-item entries (at most kNumClusters * rounds of them).
  const Metadata* __restrict__ metadata = nullptr;
  int64_t workspace_stride;  // bytes per batch
  uint32_t batch_size;
  uint32_t page_bits;
  const int32_t* __restrict__ batch_indices = nullptr;  // [num_works], maps work_id -> batch_id for page_table
  uint32_t extend_len = 0;  // tokens per batch in decode (0 = use batch_indices)

  SGL_DEVICE const float* get_scores(const uint32_t batch_id) const { return scores + batch_id * score_stride; }
  SGL_DEVICE uint32_t get_batch_id(const uint32_t work_id) const {
    if (batch_indices != nullptr) {
      return static_cast<uint32_t>(batch_indices[work_id]);
    }
    return (extend_len > 0) ? (work_id / extend_len) : work_id;
  }
  SGL_DEVICE impl::TransformParams get_transform(const uint32_t work_id, int32_t* indices) const {
    const uint32_t bid = get_batch_id(work_id);
    return {
        .page_table = page_table + bid * page_table_stride,
        .indices_in = indices,
        .indices_out = page_indices + work_id * K_VAL,
        .page_bits = page_bits,
    };
  }
  SGL_DEVICE const GlobalMetadata& get_global_metadata() const {
    return *reinterpret_cast<const GlobalMetadata*>(metadata);
  }
  SGL_DEVICE const Metadata& get_item_metadata(uint32_t work_id) const {
    return metadata[1 + work_id];  // +1 to skip the GlobalMetadata row
  }
};

template <uint32_t K_VAL>
SGL_DEVICE uint2 partition_work(uint32_t length, uint32_t rank) {
  constexpr uint32_t kTMAAlign = 4;
  constexpr uint32_t kCS = kClusterSize_v<K_VAL>;
  const auto total_units = (length + kTMAAlign - 1) / kTMAAlign;
  const auto base = total_units / kCS;
  const auto extra = total_units % kCS;
  const auto local_units = base + (rank < extra ? 1u : 0u);
  const auto offset_units = rank * base + min(rank, extra);
  const auto offset = offset_units * kTMAAlign;
  const auto finish = min(offset + local_units * kTMAAlign, length);
  return {offset, finish - offset};
}

/// Persistent scheduler. A single block:
///  1. Decides a cluster_threshold from the real seq_lens distribution (or
///     uses the caller-supplied `static_cluster_threshold` when non-zero).
///  2. Writes that threshold + N into metadata[0] (the GlobalMetadata row).
///  3. Compacts items with seq_len > threshold into metadata[1..N+1), laid out
///     to match the persistent consumer's round-robin stride (kNumClusters).
///     Entries for clusters that get no work are zero-filled.
template <uint32_t K_VAL>
PLAN_KERNEL void topk_plan(const uint32_t* __restrict__ seq_lens, Metadata* __restrict__ metadata,
                           const uint32_t batch_size, const uint32_t static_cluster_threshold) {
  // Candidate thresholds, strictly increasing. Picked to give the auto-heuristic
  // reasonable granularity without needing a full sort. Must all be >= kMax2PassLength.

  struct Pair {
    uint32_t threshold;
    uint32_t max_batch_size;
  };
  /// NOTE: only tuned on B200
  constexpr Pair kCandidates[] = {
      {32768, 30}, {40960, 45}, {49152, 45}, {65536, 60}, {98304, 60}, {131072, 75}, {196608, 90}, {262144, 105},
  };
  constexpr uint32_t kNumCandidates = sizeof(kCandidates) / sizeof(kCandidates[0]);
  constexpr uint32_t kMinBatchSize = kCandidates[0].max_batch_size;
  static_assert(kCandidates[0].threshold == kMax2PassLength_v<K_VAL>);
  static_assert(kCandidates[kNumCandidates - 1].threshold == kMaxSupportedLength_v<K_VAL>);

  __shared__ uint32_t s_count;  // final N after compaction
  __shared__ uint32_t s_counts[kNumCandidates];
  __shared__ uint32_t s_threshold;

  const auto tx = threadIdx.x;
  if (tx == 0) s_count = 0;
  if (tx < kNumCandidates) s_counts[tx] = 0;
  __syncthreads();

  // --- Phase 1: decide threshold ------------------------------------------
  if (static_cluster_threshold > 0) {
    if (tx == 0) s_threshold = static_cluster_threshold;
  } else if (batch_size <= kMinBatchSize) {
    if (tx == 0) s_threshold = kMax2PassLength_v<K_VAL>;  // always prefer cluster
  } else {
    // Count items above each candidate threshold. Monotonically non-increasing in T.
    for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
      const uint32_t sl = seq_lens[i];
      assert(sl <= kMaxSupportedLength_v<K_VAL>);
      uint32_t count = 0;
#pragma unroll
      for (uint32_t j = 0; j < kNumCandidates; ++j) {
        count += (sl > kCandidates[j].threshold ? 1 : 0);
      }
      if (count > 0) {
        atomicAdd(&s_counts[count - 1], 1);
      }
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t accum = 0;
      uint32_t chosen = kMaxSupportedLength_v<K_VAL>;
#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const auto j = kNumCandidates - 1 - i;
        accum += s_counts[j];
        /// NOTE: `accum` increasing, while `max_batch_size` decreasing
        if (accum > kCandidates[j].max_batch_size) break;
        chosen = kCandidates[j].threshold;
      }
      s_threshold = chosen;
    }
  }
  __syncthreads();
  // sanity check: below 2 pass threshold, must fits in small path
  const auto cluster_threshold = max(s_threshold, kMax2PassLength_v<K_VAL>);

  // --- Phase 2: compact items with seq_len > threshold into metadata[1..] -
  // Per-item rows live at metadata[1 + pos]; metadata[0] is the GlobalMetadata row.
  for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
    const uint32_t sl = seq_lens[i];
    if (sl > cluster_threshold) {
      const auto pos = atomicAdd(&s_count, 1);
      metadata[1 + pos] = {i, sl, false};
    }
  }
  __syncthreads();
  const auto N = s_count;

  // --- Phase 3: has_next + sentinels + GlobalMetadata ---------------------
  for (uint32_t i = tx; i < N; i += kBlockSize) {
    if (i + kNumClusters < N) metadata[1 + i].has_next = true;
  }
  // Zero-fill the first kNumClusters sentinel slots that got no valid entry.
  if (tx < kNumClusters && tx >= N) metadata[1 + tx] = {0, 0, false};
  // Write global metadata (row 0).
  if (tx == 0) {
    auto* g = reinterpret_cast<GlobalMetadata*>(metadata);
    *g = {
        .cluster_threshold = cluster_threshold,
        .num_cluster_items = N,
        .reserved = {0, 0},
    };
  }
}

template <uint32_t K_VAL>
SMALL_TOPK_KERNEL void  // short context
topk_short_transform(const __grid_constant__ TopKParams<K_VAL> params) {
  using Small = Small_t<K_VAL>;
  alignas(128) extern __shared__ uint8_t smem[];
  __shared__ int32_t s_topk_indices[K_VAL];
  const auto batch_id = blockIdx.x;
  const auto seq_len = params.seq_lens[batch_id];
  const auto transform = params.get_transform(batch_id, s_topk_indices);
  // trivial case
  if (seq_len <= K_VAL) {
    impl::trivial_transform(transform, seq_len, K_VAL);
  } else {
    Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem, /*use_pdl=*/true);
    PDLTriggerSecondary<true>();
    Small::transform(transform);
  }
}

template <uint32_t K_VAL>
LARGE_TOPK_STAGE_1 void  // long context, middle to large batch size
topk_combine_preprocess(const __grid_constant__ TopKParams<K_VAL> params) {
  using Large = Large_t<K_VAL>;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
  alignas(128) extern __shared__ uint8_t smem[];
  __shared__ int32_t s_topk_indices[K_VAL];
  uint32_t work_id = blockIdx.x;
  uint32_t batch_id;
  uint32_t seq_len;
  bool has_next;
  uint32_t length;
  uint32_t offset;
  const auto cluster_rank = blockIdx.y;

  const auto prefetch_metadata = [&] {
    const auto metadata = params.get_item_metadata(work_id);
    batch_id = metadata.batch_id;
    seq_len = metadata.seq_len;
    has_next = metadata.has_next;
    work_id += kNumClusters;  // advance to the next item for this cluster
  };
  const auto launch_prologue = [&] {
    const auto partition = partition_work<K_VAL>(seq_len, cluster_rank);
    offset = partition.x;
    length = partition.y;
    Large::stage1_prologue(params.get_scores(batch_id) + offset, length, smem);
  };

  PDLWaitPrimary<true>();
  PDLTriggerSecondary<true>();

  prefetch_metadata();
  if (seq_len == 0) return;
  Large::stage1_init(smem);
  launch_prologue();
  while (true) {
    const auto this_length = length;
    const auto this_offset = offset;
    const auto need_prefetch = has_next;
    const auto transform = params.get_transform(batch_id, s_topk_indices);
    const auto ws = params.workspace + batch_id * params.workspace_stride;
    if (need_prefetch) prefetch_metadata();
    Large::stage1(s_topk_indices, this_length, smem, /*reuse=*/true);
    if (need_prefetch) launch_prologue();
    Large::stage1_epilogue(transform, this_offset, ws, smem);
    if (!need_prefetch) break;
  }
#else
  // Cluster path not available on SM < 90; should not be reached at runtime.
  __trap();
#endif
}

template <uint32_t K_VAL>
LARGE_TOPK_STAGE_2 void  // long context, middle to large batch size
topk_combine_transform(const __grid_constant__ TopKParams<K_VAL> params) {
  using Large = Large_t<K_VAL>;
  using Medium = Medium_t<K_VAL>;
  using Small = Small_t<K_VAL>;
  alignas(128) extern __shared__ uint8_t smem[];
  __shared__ int32_t s_topk_indices[K_VAL];
  const auto batch_id = blockIdx.x;
  const auto seq_len = params.seq_lens[batch_id];
  const auto cluster_threshold = params.get_global_metadata().cluster_threshold;
  const auto transform = params.get_transform(batch_id, s_topk_indices);
  if (seq_len <= K_VAL) {
    impl::trivial_transform(transform, seq_len, K_VAL);
  } else if (seq_len <= kMax2PassLength_v<K_VAL>) {
    if (seq_len <= Small::kMax1PassLength) {
      Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem);
    } else {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
      __syncwarp();
      Small::run<true>(params.get_scores(batch_id), s_topk_indices, seq_len, smem);
#else
      // 2-pass RegisterTopK requires TMA (SM 90+); not available on this arch.
      __trap();
#endif
    }
    Small::transform(transform);
  } else if (seq_len <= cluster_threshold) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    Medium::run(params.get_scores(batch_id), seq_len, s_topk_indices, smem);
    Medium::transform(transform, smem);
#else
    // StreamingTopK requires TMA (SM 90+); not available on this arch.
    __trap();
#endif
  } else {
    const auto ws = params.workspace + batch_id * params.workspace_stride;
    PDLWaitPrimary<true>();
    Large::transform(transform, ws, smem);
  }
}

template <uint32_t K_VAL>
FUSED_COMBINE_KERNEL void  // long context, small batch size
topk_fused_transform(const __grid_constant__ TopKParams<K_VAL> params) {
  using Large = Large_t<K_VAL>;
  using Small = Small_t<K_VAL>;
  alignas(128) extern __shared__ uint8_t smem[];
  __shared__ int32_t s_topk_indices[K_VAL];
  const auto batch_id = blockIdx.x;
  const auto cluster_rank = blockIdx.y;
  const auto seq_len = params.seq_lens[batch_id];
  const auto transform = params.get_transform(batch_id, s_topk_indices);
  if (seq_len <= K_VAL) {
    if (cluster_rank != 0) return;  // only first rank work
    impl::trivial_transform(transform, seq_len, K_VAL);
  } else if (seq_len <= Small::kMax1PassLength) {
    if (cluster_rank != 0) return;  // only first rank work
    Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem, /*use_pdl=*/true);
    Small::transform(transform);
  } else {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    const auto [offset, length] = partition_work<K_VAL>(seq_len, cluster_rank);
    const auto ws = params.workspace + batch_id * params.workspace_stride;
    Large::stage1_init(smem);
    PDLWaitPrimary<true>();
    Large::stage1_prologue(params.get_scores(batch_id) + offset, length, smem);
    Large::stage1(s_topk_indices, length, smem);
    Large::stage1_epilogue(transform, offset, ws, smem);
    cooperative_groups::this_cluster().sync();
    if (cluster_rank != 0) return;  // only first rank do the stage-2
    Large::transform(transform, ws, smem);
#else
    // Cluster path not available on SM < 90; should not be reached at runtime.
    __trap();
#endif
  }
}

// ---------------------------------------------------------------------------
// Host launcher: replaces the stripped CombinedTopKKernel::plan +
// CombinedTopKKernel::transform pair.
//
// Template parameter K_VAL: top-K value (512, 1024, or 2048).
// ---------------------------------------------------------------------------
template <uint32_t K_VAL>
inline void launch_topk_v2(const uint32_t* seq_lens, const float* scores, const int32_t* page_table,
                           int32_t* page_indices, int64_t score_stride, int64_t page_table_stride, uint8_t* workspace,
                           Metadata* metadata, int64_t workspace_stride_bytes, uint32_t batch_size,
                           uint32_t page_bits, uint32_t max_seq_len, uint32_t static_cluster_threshold,
                           const int32_t* batch_indices, uint32_t extend_len, cudaStream_t stream) {
  using Large = Large_t<K_VAL>;
  using Medium = Medium_t<K_VAL>;
  using Small = Small_t<K_VAL>;

  constexpr size_t kStage1SMEM = sizeof(typename Large::Smem) + 128;
  constexpr size_t kStage2SMEM =
      (sizeof(typename Small::Smem) > sizeof(typename Medium::Smem) ? sizeof(typename Small::Smem) : sizeof(typename Medium::Smem)) + 128;
  constexpr size_t kFusedSMEM = kStage1SMEM > kStage2SMEM ? kStage1SMEM : kStage2SMEM;

  // Set dynamic SMEM limits (each static initializer runs exactly once).
  setup_kernel_smem_once<topk_short_transform<K_VAL>, kStage2SMEM>();
  setup_kernel_smem_once<topk_combine_transform<K_VAL>, kStage2SMEM>();

  // Detect cluster support at runtime (SM 90+).
  static const bool has_cluster = (llm_kernels::utils::GetSMVersion() >= 90);

  if (has_cluster) {
    setup_kernel_smem_once<topk_combine_preprocess<K_VAL>, kStage1SMEM>();
    setup_kernel_smem_once<topk_fused_transform<K_VAL>, kFusedSMEM>();
  }

  const TopKParams<K_VAL> params = {
      .seq_lens = seq_lens,
      .scores = scores,
      .page_table = page_table,
      .page_indices = page_indices,
      .score_stride = score_stride,
      .page_table_stride = page_table_stride,
      .workspace = workspace,
      .metadata = reinterpret_cast<const Metadata*>(metadata),
      .workspace_stride = workspace_stride_bytes,
      .batch_size = batch_size,
      .page_bits = page_bits,
      .batch_indices = batch_indices,
      .extend_len = extend_len,
  };

  constexpr uint32_t kCS = kClusterSize_v<K_VAL>;

  if (max_seq_len <= Small::kMax1PassLength) {
    // All items fit in the short (register) path — no stage-1 needed.
    LAUNCH_KERNEL_WITH_PDL(topk_short_transform<K_VAL>, batch_size, kBlockSize, kStage2SMEM, stream, params);
  } else if (has_cluster && batch_size <= kNumClusters) {
    // Small batch: fuse stage-1 and stage-2 into one cluster kernel (SM 90+ only).
    LAUNCH_KERNEL_WITH_PDL(topk_fused_transform<K_VAL>, dim3(batch_size, kCS), kBlockSize, kFusedSMEM, stream, params);
  } else if (has_cluster) {
    // Large batch: plan -> stage-1 cluster -> stage-2 (SM 90+ only).
    topk_plan<K_VAL><<<1, kBlockSize, 0, stream>>>(seq_lens, metadata, batch_size, static_cluster_threshold);
    const uint32_t num_clusters = batch_size < kNumClusters ? batch_size : kNumClusters;
    LAUNCH_KERNEL_WITH_PDL(topk_combine_preprocess<K_VAL>, dim3(num_clusters, kCS), kBlockSize, kStage1SMEM, stream, params);
    LAUNCH_KERNEL_WITH_PDL(topk_combine_transform<K_VAL>, batch_size, kBlockSize, kStage2SMEM, stream, params);
  } else {
    // SM < 90 fallback: use plan + stage-2 only.
    topk_plan<K_VAL><<<1, kBlockSize, 0, stream>>>(seq_lens, metadata, batch_size, static_cluster_threshold);
    LAUNCH_KERNEL_WITH_PDL(topk_combine_transform<K_VAL>, batch_size, kBlockSize, kStage2SMEM, stream, params);
  }
}

}  // namespace llm_kernels::nvidia::topk
