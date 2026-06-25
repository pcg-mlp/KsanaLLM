#pragma once

#include <humming/utils/all.cuh>


template <
    class BlockShape, class WarpShape,
    class ElementA, class ElementB,
    class LayerConfig, class TuningConfig>
class S2RMemoryLoaderBZP {
private:
  static constexpr bool kIsFpZeroPoint = LayerConfig::kIsFpZeroPoint;
  static constexpr bool kIsChannelScale = LayerConfig::kIsChannelWeightScale;
  static constexpr bool kIsGroupScale = LayerConfig::kIsGroupWeightScale;
  static constexpr uint32_t kGroupSize = kIsChannelScale ? BlockShape::K : LayerConfig::kWeightScaleGroupSize;
  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

  static constexpr uint32_t kNumZPBits = kIsFpZeroPoint ? 16 : MAX(4, static_next_power_of_2(ElementB::kBits));
  static constexpr uint32_t kLoadBytes = WarpShape::N / 8 * kNumZPBits / 8;
  using LoadType = typename LoadTypeChooser<kLoadBytes>::Type;
  static constexpr uint32_t kSmemStride = BlockShape::N * kNumZPBits / 32 / 4;
  static constexpr uint32_t kSmemStrideLoadType = kSmemStride * 16 / sizeof(LoadType);

public:
  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    constexpr uint32_t kNumLoadBlockEvery64Rows = (64 * kNumZPBits) / kLoadBytes;
    constexpr uint32_t kNumWarpsEvery64Rows = 64 / WarpShape::N;
    uint32_t warp_id = threadIdx.x / 32;

    if constexpr (!kIsFpZeroPoint) {
      uint32_t zp_sh_rd = warp_id % N_WARPS / kNumWarpsEvery64Rows * (8 * kNumWarpsEvery64Rows);
      zp_sh_rd += (threadIdx.x % 32) / 4 * kNumWarpsEvery64Rows + warp_id % kNumWarpsEvery64Rows;

      if constexpr (kGroupSize < BlockShape::K) {
        uint32_t k_index = (warp_id / (M_WARPS * N_WARPS)) * WarpShape::K + iter_id * kPartMmaShapeK;
        uint32_t group_index = k_index / kGroupSize;
        zp_sh_rd += group_index * kSmemStrideLoadType;
      };

      LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);
      const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);

      reg_ptr_load[0] = smem_ptr_load[zp_sh_rd];
    } else {
      static_assert(ElementA::kBits == 16);
      uint32_t zp_sh_rd = (threadIdx.x % 32) / 4 + (warp_id % N_WARPS / (64 / WarpShape::N)) * 8;
      if constexpr (kGroupSize < BlockShape::K) {
        uint32_t k_index = (warp_id / (M_WARPS * N_WARPS)) * WarpShape::K + iter_id * kPartMmaShapeK;
        uint32_t group_index = k_index / kGroupSize;
        zp_sh_rd += group_index * kSmemStrideLoadType;
      };
      LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);
      const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);
      if (WarpShape::N == 32) zp_sh_rd = zp_sh_rd * 2 + warp_id % 2;
      reg_ptr_load[0] = smem_ptr_load[zp_sh_rd];
    }
  }
};
