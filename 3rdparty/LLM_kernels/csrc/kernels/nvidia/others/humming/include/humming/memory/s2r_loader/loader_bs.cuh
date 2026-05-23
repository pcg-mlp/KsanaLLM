#pragma once

#include <humming/utils/all.cuh>


template <
    class MmaOpClass,
    class BlockShape, class WarpShape,
    class ElementA, class ElementBS,
    class LayerConfig, class TuningConfig>
class S2RMemoryLoaderBS {
private:
  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  static constexpr bool kUseWgmma = MmaOpClass::kMmaType == MmaType::WGMMA;

  static constexpr bool kIsChannel = LayerConfig::kIsChannelWeightScale;
  static constexpr bool kIsGroup = LayerConfig::kIsGroupWeightScale;
  static constexpr bool kIsBlock = LayerConfig::kIsBlockWeightScale;
  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;
  static constexpr uint32_t kGroupSize = kIsChannel ? BlockShape::K : LayerConfig::kWeightScaleGroupSize;
  static constexpr uint32_t kGroupSizeN = LayerConfig::kWeightScaleGroupSizeN;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

  static constexpr uint32_t kNumSubBlocks = WarpShape::N / 16;
  static constexpr uint32_t kNumScalesPerSubBlock = !kUseFusedE8m0Scale && (kIsChannel || (ElementA::kBits != 16 && !kUseWgmma)) ? 4 : 2;
  static constexpr uint32_t kNumScales = kNumSubBlocks * kNumScalesPerSubBlock;
  static constexpr uint32_t kNumBytesPerThread = kNumScales * ElementBS::kBits / 8;

  static constexpr uint32_t kNumRowsPerMiniBlock = 128 / kNumScalesPerSubBlock;
  static constexpr uint32_t kNumWarpsPerMiniBlock = CEIL_DIV(kNumRowsPerMiniBlock, WarpShape::N);
  static constexpr uint32_t kMaxBytesPerLoad = ElementBS::kBits / kNumWarpsPerMiniBlock;
  static constexpr uint32_t kNumBytesPerLoad = MIN(kNumBytesPerThread, kMaxBytesPerLoad);
  using LoadType = typename LoadTypeChooser<kMaxBytesPerLoad>::Type;

  static constexpr uint32_t kLoadItersPerGroup = CEIL_DIV(kNumBytesPerThread, sizeof(LoadType));
  static constexpr uint32_t kSmemStride = BlockShape::N * ElementBS::kBits / 32 / 4;
  static constexpr uint32_t kSmemStrideLoadType = kSmemStride * 16 / sizeof(LoadType);

public:
  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    if constexpr (kIsBlock) {
      load_block(smem_ptr, regs_ptr, iter_id);
    } else {
      load_group_or_channel(smem_ptr, regs_ptr, iter_id);
    }
  }

  CUDA_INLINE
  void load_block(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    static_assert(kGroupSizeN >= 64);

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t n_warp_id = warp_id % N_WARPS;

    uint32_t index = (n_warp_id * WarpShape::N) / kGroupSizeN;
    if constexpr (BlockShape::K >= kGroupSize) {
      uint32_t k_index = (warp_id / (M_WARPS * N_WARPS)) * WarpShape::K + iter_id * kPartMmaShapeK;
      uint32_t group_index = k_index / kGroupSize;
      index += group_index * CEIL_DIV(BlockShape::N, kGroupSizeN);
    }
    regs_ptr[0] = reinterpret_cast<const uint32_t *>(smem_ptr)[index];
  };

  CUDA_INLINE
  void load_group_or_channel(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    uint32_t warp_id = threadIdx.x / 32;

    uint32_t n_warp_id = warp_id % N_WARPS / kNumWarpsPerMiniBlock;
    constexpr uint32_t warp_load_delta = (16 / kNumScalesPerSubBlock);
    uint32_t s_sh_rd = (kLoadItersPerGroup * warp_load_delta * kNumWarpsPerMiniBlock) * n_warp_id;

    if constexpr (kUseFusedE8m0Scale) {
      s_sh_rd += (threadIdx.x % 32) / 4 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    } else if constexpr (kUseWgmma && kIsChannel) {
      s_sh_rd += (threadIdx.x % 32) / 8 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    } else if constexpr (kUseWgmma && ElementA::kBits != 16) {
      s_sh_rd += (threadIdx.x % 32) / 4 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    } else if constexpr (!kUseWgmma && (kIsChannel || ElementA::kBits != 16)) {
      s_sh_rd += threadIdx.x % 4 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    } else if constexpr (kIsGroup && ElementA::kBits == 16) {
      s_sh_rd += (threadIdx.x % 32) / 4 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    }

    if constexpr (kGroupSize < BlockShape::K) {
      uint32_t k_index = (warp_id / (M_WARPS * N_WARPS)) * WarpShape::K + iter_id * kPartMmaShapeK;
      uint32_t group_index = k_index / kGroupSize;
      s_sh_rd += group_index * kSmemStrideLoadType;
    };

    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);
    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);

    PRAGMA_UNROLL
    for (uint32_t j = 0; j < kLoadItersPerGroup; j++) {
      uint32_t smem_idx = warp_load_delta * j + s_sh_rd;
      reg_ptr_load[j] = smem_ptr_load[smem_idx];
    }
  };
};
