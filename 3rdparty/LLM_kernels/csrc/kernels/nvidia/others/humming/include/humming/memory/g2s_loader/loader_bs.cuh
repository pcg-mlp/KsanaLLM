#pragma once

#include <humming/utils/all.cuh>


template <
    class ProblemShape, class BlockShape,
    class ElementBS,
    class LayerConfig, class TuningConfig>
class G2SMemoryLoaderBS {
private:
  static constexpr bool kUseWarpSpec = TuningConfig::kUseWarpSpec;
  static constexpr bool kUseTma = TuningConfig::kUseTmaBS;
  static constexpr bool kUseCpAsync = TuningConfig::kUseCpAsync;
  static constexpr uint32_t kNumLoadThreads = TuningConfig::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = TuningConfig::kNumThreads - kNumLoadThreads;

  static constexpr bool kIsChannel = LayerConfig::kIsChannelWeightScale;
  static constexpr bool kIsGroup = LayerConfig::kIsGroupWeightScale;
  static constexpr bool kIsBlock = LayerConfig::kIsBlockWeightScale;
  static constexpr bool kIsTensor = LayerConfig::kIsTensorWeightScale;
  static constexpr bool kIsGroupOrBlock = kIsGroup || kIsBlock;
  static constexpr uint32_t kGroupSize = !kIsGroupOrBlock ? ProblemShape::K : LayerConfig::kWeightScaleGroupSize;
  static constexpr uint32_t kGroupSizeN = kIsBlock ? LayerConfig::kWeightScaleGroupSizeN : 1;

  static constexpr uint32_t kSmemStride = CEIL_DIV(BlockShape::N, kGroupSizeN) * ElementBS::kBits / 32 / 4;
  static constexpr uint32_t kGmemStride = ProblemShape::N * ElementBS::kBits / 32 / 4;
  static constexpr uint32_t kProblemNumGroups = CEIL_DIV(ProblemShape::K, kGroupSize);
  static constexpr uint32_t kGmemExpertStride = kGmemStride * kProblemNumGroups;
  static constexpr uint32_t kNumGroups = CEIL_DIV(BlockShape::K, kGroupSize);
  static constexpr uint32_t kNumInt4s = kSmemStride * kNumGroups;
  static constexpr uint32_t kLoadsPerGroup = kIsChannel ? 1 : CEIL_DIV(kGroupSize, BlockShape::K);

public:
  const CUtensorMap *tensor_map_ptr;
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  uint32_t row_offset = 0;
  uint32_t col_offset;
  uint32_t counter = 0;

  CUDA_INLINE
  G2SMemoryLoaderBS(const void *ptr) {
    if constexpr (kUseTma) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<const int4 *>(ptr);
    }
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(int4 *smem_ptr, void *mbar_ptr) {
    counter = kLoadsPerGroup != 1 ? (counter + 1) % kLoadsPerGroup : 0;
    if constexpr (kUseTma) load_tma(smem_ptr, mbar_ptr);
    else load_legacy(smem_ptr);
    if constexpr (kShouldAdvance) advance();
  };

  CUDA_INLINE
  void load_tma(int4 *smem_ptr, void *mbar_ptr) {
    if (threadIdx.x == kLoadThreadOffset) tma_load_3d(tensor_map_ptr, smem_ptr, mbar_ptr, 0, col_offset, row_offset);
  }

  CUDA_INLINE
  void load_legacy(int4 *smem_ptr) {
    if constexpr (kIsBlock) {
      constexpr uint32_t kLoadStride = ProblemShape::N / kGroupSizeN;
      if (threadIdx.x < CEIL_DIV(BlockShape::K, kGroupSize) * CEIL_DIV(BlockShape::N, kGroupSizeN)) {
        const uint32_t *gmem_ptr_load = reinterpret_cast<const uint32_t *>(gmem_ptr_raw);
        uint32_t *smem_ptr_load = reinterpret_cast<uint32_t *>(smem_ptr);

        const uint32_t gmem_row = row_offset + threadIdx.x / CEIL_DIV(BlockShape::N, kGroupSizeN);
        const uint32_t gmem_col = col_offset + threadIdx.x % CEIL_DIV(BlockShape::N, kGroupSizeN);
        uint32_t gmem_index = gmem_row * kLoadStride + gmem_col;
        legacy_load<TuningConfig::kUseCpAsync>(&gmem_ptr_load[gmem_index], &smem_ptr_load[threadIdx.x]);
      }
    } else {
      legacy_load_2d<
          kUseCpAsync, kNumInt4s, kNumLoadThreads,
          kGmemStride, kSmemStride, kLoadThreadOffset>(gmem_ptr, smem_ptr);
    }
  }

  CUDA_INLINE
  void advance() {
    if (kIsGroupOrBlock && (kLoadsPerGroup == 1 || counter == 0)) {
      row_offset += kNumGroups;
      gmem_ptr += kGmemStride * kNumGroups;
    }
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t n_block_id, uint32_t k_block_id) {
    row_offset = kProblemNumGroups * expert_id;

    if constexpr (kIsGroupOrBlock) {
      if constexpr (BlockShape::K >= kGroupSize) {
        row_offset += k_block_id * kNumGroups;
      } else {
        row_offset += (k_block_id * BlockShape::K) / kGroupSize;
      }
    }

    if constexpr (kIsBlock) {
      col_offset = (n_block_id * BlockShape::N) / kGroupSizeN;
    } else {
      col_offset = n_block_id * (BlockShape::N / 16);
    }

    uint32_t gmem_offset = row_offset * kGmemStride + n_block_id * kSmemStride;
    gmem_ptr = gmem_ptr_raw + gmem_offset;
  }
};
