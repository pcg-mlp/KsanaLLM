#pragma once

#include <humming/utils/all.cuh>

template <
    class SharedStorage,
    class ProblemShape, class BlockShape,
    class LayerConfig, class ComputeConfig, class TuningConfig>
class Scheduler {
private:
  static constexpr bool kUseCpAsync = TuningConfig::kUseCpAsync;
  static constexpr bool kUseWarpSpec = TuningConfig::kUseWarpSpec;
  static constexpr bool kUseStreamK = TuningConfig::kUseStreamK;
  static constexpr bool kIsIndexedGemm = ComputeConfig::kGemmType == GemmType::INDEXED;
  static constexpr bool kIsGroupedGemm = ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS || ComputeConfig::kGemmType == GemmType::GROUPED_MASKED;
  static constexpr uint32_t kNumExperts = LayerConfig::kNumExperts;
  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  static constexpr uint32_t kNumMathThreads = TuningConfig::kNumMathThreads;
  static constexpr uint32_t kNumLoadThreads = TuningConfig::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = kNumThreads - kNumLoadThreads;
  static constexpr uint32_t kMultiCastSizeA = TuningConfig::kMultiCastSizeA;
  static constexpr uint32_t kMultiCastSizeB = TuningConfig::kMultiCastSizeB;
  static constexpr uint32_t kMultiCastSize = kMultiCastSizeA * kMultiCastSizeB;

  static constexpr uint32_t kInputScaleGroupSize = LayerConfig::kInputScaleGroupSize > 0 ? LayerConfig::kInputScaleGroupSize : 1;
  static constexpr uint32_t kWeightScaleGroupSize = LayerConfig::kWeightScaleGroupSize > 0 ? LayerConfig::kWeightScaleGroupSize : 1;
  static constexpr uint32_t kMaxGroupSize = MAX(kInputScaleGroupSize, kWeightScaleGroupSize);

  static constexpr uint32_t N_BLOCKS = ProblemShape::N / BlockShape::N / kMultiCastSizeA;
  static constexpr uint32_t K_BLOCKS = ProblemShape::K / BlockShape::K;

  uint32_t m_blocks;
  uint32_t mn_blocks;
  uint32_t mnk_blocks;

  uint32_t streamk_mnk_total_iters;
  uint32_t streamk_mnk_iters;
  uint32_t streamk_mnk_next_index;
  uint32_t dp_mn_total_iters;
  uint32_t dp_mn_iters;
  uint32_t dp_mn_next_index;

public:
  SharedStorage &smem;
  uint32_t shape_m;

  uint32_t m_block_id;
  uint32_t n_block_id;
  uint32_t k_block_id;
  uint32_t old_m_block_id = 0;

  // for stream-k
  uint32_t slice_iters;
  uint32_t slice_count;
  uint32_t slice_id;
  uint32_t locks_offset;

  // for tma multi-cast
  uint32_t cluster_rank = blockIdx.x % kMultiCastSize;

  // for moe gemm (indexed gemm or grouped gemm)
  uint32_t old_expert_id = (1 << 30);
  uint32_t expert_id = 0;

  // for indexed gemm
  const uint32_t *row_index_blocks;
  const uint32_t *expert_ids;
  uint32_t top_k;

  // for grouped gemm
  uint32_t current_shape_m;
  uint32_t expert_max_num_tokens;
  uint32_t offset_in_expert = 0;
  uint32_t m_offset = 0;
  bool use_int64_expert_layout;

  CUtensorMap *tensor_map_buffer;

  CUDA_INLINE
  Scheduler(
      SharedStorage &smem, const void *output_ptr, CUtensorMap *tensor_map_buffer,
      uint32_t shape_m, uint32_t top_k, const uint32_t *row_index_blocks, const uint32_t *expert_ids,
      const uint32_t *num_tokens_padded_ptr, const uint32_t *expert_layout_ptr, bool use_int64_expert_layout)
      : smem(smem), tensor_map_buffer(tensor_map_buffer), shape_m(shape_m), top_k(top_k),
        row_index_blocks(row_index_blocks), expert_ids(expert_ids), use_int64_expert_layout(use_int64_expert_layout) {

    if constexpr (kIsGroupedGemm && TuningConfig::kUseTmaC) {
      if (threadIdx.x == 0) smem.tensor_map_buffer[0] = reinterpret_cast<const CUtensorMap *>(output_ptr)[0];
      __syncwarp();
    }

    current_shape_m = shape_m;
    expert_max_num_tokens = shape_m / LayerConfig::kNumExperts;
    calc_m_blocks(shape_m, num_tokens_padded_ptr, expert_layout_ptr);
    mn_blocks = m_blocks * N_BLOCKS;
    mnk_blocks = mn_blocks * K_BLOCKS;
    uint32_t kNumCtaGroups = gridDim.x / kMultiCastSize;

    if constexpr (kUseStreamK) {
      uint32_t streamk_mn_blocks = mn_blocks;
      if (mn_blocks > kNumCtaGroups) {
        streamk_mn_blocks = mn_blocks % kNumCtaGroups;
        if (streamk_mn_blocks && streamk_mn_blocks * 10 <= kNumCtaGroups) streamk_mn_blocks += kNumCtaGroups;
      }

      dp_mn_iters = (mn_blocks - streamk_mn_blocks) / kNumCtaGroups;

      uint32_t streamk_mnk_blocks = streamk_mn_blocks * K_BLOCKS;

      streamk_mnk_total_iters = CEIL_DIV(streamk_mnk_blocks, kNumCtaGroups);

      constexpr int32_t blocks_per_group = kMaxGroupSize / BlockShape::K;

      if constexpr (blocks_per_group > 1) {
        streamk_mnk_total_iters = blocks_per_group * CEIL_DIV(streamk_mnk_total_iters, blocks_per_group);
      };

      streamk_mnk_next_index = kNumCtaGroups * dp_mn_iters * K_BLOCKS + streamk_mnk_total_iters * (blockIdx.x / kMultiCastSize);

      if (streamk_mnk_next_index >= mnk_blocks) {
        streamk_mnk_iters = 0;
      } else {
        streamk_mnk_iters = mnk_blocks - streamk_mnk_next_index;
        if (streamk_mnk_iters > streamk_mnk_total_iters) streamk_mnk_iters = streamk_mnk_total_iters;
      };
    } else {
      dp_mn_iters = mn_blocks / kNumCtaGroups;
      if (blockIdx.x / kMultiCastSize < mn_blocks % kNumCtaGroups) { dp_mn_iters += 1; };
    }

    dp_mn_total_iters = dp_mn_iters;
    slice_count = 1;
    slice_id = 0;

    if (dp_mn_iters) { dp_mn_next_index = blockIdx.x / kMultiCastSize; };
  };

  CUDA_INLINE
  void calc_m_blocks(const uint32_t shape_m, const uint32_t *num_tokens_padded_ptr, const uint32_t *expert_layout) {
    if constexpr (ComputeConfig::kGemmType == GemmType::DENSE) {
      m_blocks = CEIL_DIV(shape_m, BlockShape::M * kMultiCastSizeB);
    } else if constexpr (kIsIndexedGemm) {
      uint32_t padded_shape_m = num_tokens_padded_ptr[0];
      m_blocks = CEIL_DIV(padded_shape_m, BlockShape::M * kMultiCastSizeB);
    } else if constexpr (kIsGroupedGemm) {
      if constexpr (ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS) {
        if (use_int64_expert_layout)
          legacy_load_2d<kUseCpAsync, kNumExperts + 1, kNumThreads, 2, 1>(expert_layout, smem.expert_offset);
        else
          legacy_load_2d<kUseCpAsync, kNumExperts + 1, kNumThreads, 1, 1>(expert_layout, smem.expert_offset);
      } else {
        if (use_int64_expert_layout)
          legacy_load_2d<kUseCpAsync, kNumExperts, kNumThreads, 2, 1>(expert_layout, smem.expert_tokens);
        else
          legacy_load_2d<kUseCpAsync, kNumExperts, kNumThreads, 1, 1>(expert_layout, smem.expert_tokens);
      }
      if constexpr (kUseCpAsync) cp_async_commit_group();
      if constexpr (kUseCpAsync) cp_async_wait_group<0>();
      __syncthreads();

      if constexpr (ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS) {
        smem.expert_tokens[kNumExperts - 1] = shape_m - smem.expert_offset[kNumExperts - 1];
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kNumExperts - 1, kNumThreads); i++) {
          uint32_t index = kNumThreads * i + threadIdx.x;
          if (index < kNumExperts - 1) {
            smem.expert_tokens[index] = smem.expert_offset[index + 1] - smem.expert_offset[index];
          }
        }

        __syncthreads();
      }

      if (threadIdx.x / 32 == 0) {
        uint32_t tmp_m_blocks = 0;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kNumExperts, 32); i++) {
          uint32_t index = 32 * i + threadIdx.x;
          if (index < kNumExperts) {
            tmp_m_blocks += CEIL_DIV(smem.expert_tokens[index], BlockShape::M);
          }
        }

        __syncwarp();
        m_blocks = warp_reduce_add(tmp_m_blocks);
        if (threadIdx.x == 0) smem.total_m_blocks[0] = m_blocks;
      }

      __syncthreads();
      m_blocks = smem.total_m_blocks[0];
    }
  }

  CUDA_INLINE
  bool get_next_block() {
    bool has_next_block = false;
    if (dp_mn_iters) {
      slice_iters = K_BLOCKS;

      m_block_id = dp_mn_next_index / N_BLOCKS;
      n_block_id = dp_mn_next_index % N_BLOCKS;

      if constexpr (kMultiCastSizeB > 1) {
        m_block_id = m_block_id * kMultiCastSizeB + cluster_rank;
      } else if constexpr (kMultiCastSizeA > 1) {
        n_block_id = n_block_id * kMultiCastSizeA + cluster_rank;
      }
      k_block_id = 0;
      dp_mn_next_index += gridDim.x / kMultiCastSize;
      dp_mn_iters--;
      has_next_block = true;
    } else if constexpr (kUseStreamK) {
      has_next_block = get_streamk_next_block();
    }

    if constexpr (kIsIndexedGemm) {
      if (has_next_block) fetch_moe_index_block();
    }
    if constexpr (kIsGroupedGemm) {
      if (has_next_block) fetch_moe_group_block();
    }

    return has_next_block;
  };

  CUDA_INLINE
  bool get_streamk_next_block() {
    if (!streamk_mnk_iters) return false;
    uint32_t streamk_mn_index = streamk_mnk_next_index / K_BLOCKS;

    m_block_id = streamk_mn_index / N_BLOCKS;
    n_block_id = streamk_mn_index % N_BLOCKS;
    if constexpr (kMultiCastSizeB > 1) {
      m_block_id = m_block_id * kMultiCastSizeB + cluster_rank;
    } else if constexpr (kMultiCastSizeA > 1) {
      n_block_id = n_block_id * kMultiCastSizeA + cluster_rank;
    }
    k_block_id = streamk_mnk_next_index - streamk_mn_index * K_BLOCKS;

    slice_iters = K_BLOCKS - k_block_id;
    slice_iters = slice_iters > streamk_mnk_iters ? streamk_mnk_iters : slice_iters;

    streamk_mnk_iters -= slice_iters;
    streamk_mnk_next_index += slice_iters;

    if (k_block_id == 0) {
      slice_id = 0;
      slice_count = CEIL_DIV(K_BLOCKS - slice_iters, streamk_mnk_total_iters) + 1;
    } else {
      slice_id = k_block_id / streamk_mnk_total_iters;
      uint32_t slice_first_block_iters = k_block_id - slice_id * streamk_mnk_total_iters;
      slice_count = CEIL_DIV(K_BLOCKS - slice_first_block_iters, streamk_mnk_total_iters);
      if (slice_first_block_iters) {
        slice_id++;
        slice_count++;
      }
    }

    slice_id = slice_count - 1 - slice_id;

    locks_offset = streamk_mn_index - dp_mn_total_iters * gridDim.x / kMultiCastSize;
    locks_offset = locks_offset * kMultiCastSize + cluster_rank;

    return true;
  };

  CUDA_INLINE
  void fetch_moe_group_block() {
    uint32_t delta_m_block_id = m_block_id - old_m_block_id;
    offset_in_expert += delta_m_block_id * BlockShape::M;

    while (offset_in_expert >= smem.expert_tokens[expert_id]) {
      offset_in_expert -= CEIL_DIV(smem.expert_tokens[expert_id], BlockShape::M) * BlockShape::M;
      expert_id++;
    }

    old_m_block_id = m_block_id;
    if constexpr (ComputeConfig::kGemmType == GemmType::GROUPED_MASKED) {
      m_offset = expert_id * expert_max_num_tokens + offset_in_expert;
      current_shape_m = expert_id * expert_max_num_tokens + smem.expert_tokens[expert_id];
    } else if constexpr (ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS) {
      m_offset = smem.expert_offset[expert_id] + offset_in_expert;
      current_shape_m = smem.expert_offset[expert_id] + smem.expert_tokens[expert_id];
    }

    if (old_expert_id != expert_id) update_tensor_map_c();
    old_expert_id = expert_id;
  };

  CUDA_INLINE
  void fetch_moe_index_block() {
    if (kUseWarpSpec && threadIdx.x < kNumMathThreads) return;

    expert_id = expert_ids[m_block_id];

    const uint32_t *gmem_ptr = row_index_blocks + m_block_id * BlockShape::M;
    const int4 *gmem_ptr_load = reinterpret_cast<const int4 *>(gmem_ptr);
    int4 *smem_ptr_load = reinterpret_cast<int4 *>(smem.wr_row_index);

    legacy_load_1d<kUseCpAsync, BlockShape::M / 4, kNumLoadThreads>(gmem_ptr_load, smem_ptr_load);
    if constexpr (kUseCpAsync) cp_async_commit_group();
    if constexpr (kUseCpAsync) cp_async_wait_group<0>();

    sync_part_threads<kNumLoadThreads, kNumThreads, 2>();

    uint32_t thread_id = threadIdx.x;
    if constexpr (kUseWarpSpec) thread_id = thread_id - kNumMathThreads;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
      uint32_t index = kNumLoadThreads * i + thread_id;
      if (index < BlockShape::M) {
        uint32_t idx = smem.wr_row_index[index];
        smem.rd_row_index[index] = idx / top_k;
      };
    }

    sync_part_threads<kNumLoadThreads, kNumThreads, 2>();
  };

  CUDA_INLINE
  void update_tensor_map_c() {
    if constexpr (kIsGroupedGemm && TuningConfig::kUseTmaC) {
      if (threadIdx.x < 32) {
        __syncwarp();
        if (threadIdx.x == 0) {
          tensor_map_replace_global_dim<1>(smem.tensor_map_buffer, current_shape_m);
          tensor_map_buffer[blockIdx.x] = smem.tensor_map_buffer[0];
          tensor_map_release_cta();
          tensor_map_acquire_cta(tensor_map_buffer + blockIdx.x);
        }
        __syncwarp();
      }
    }
  }
};
