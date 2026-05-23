#pragma once

#include <humming/utils/all.cuh>


template <class BlockShape, class WarpShape, class ElementA, class ElementB, class TuningConfig>
class S2RMemoryLoaderB {
private:
  static constexpr uint32_t kNumMathThreads = TuningConfig::kNumMathThreads;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kWarpItersK = WarpShape::K / kPartMmaShapeK;

  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

  static constexpr bool kIsWarpHalfGroup = WarpShape::N == ElementA::kBits * 2;
  static constexpr bool kLoadHalfGroup = ElementB::kBits % 2 == 0 && kIsWarpHalfGroup;
  static constexpr uint32_t TRUE_N_WARPS = kIsWarpHalfGroup ? N_WARPS / 2 : N_WARPS;
  static constexpr uint32_t kSmemStride = BlockShape::N * kPartMmaShapeK * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kNumIntsPerThread = ElementB::kBits / (kLoadHalfGroup ? 2 : 1);
  using LoadType = typename LoadTypeChooser<kNumIntsPerThread * 4>::Type;
  static constexpr uint32_t kLoadIters = kNumIntsPerThread / (sizeof(LoadType) / 4);

public:
  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    uint32_t warp_id = (threadIdx.x / 32);
    uint32_t n_warp_id = warp_id % N_WARPS;
    if (kIsWarpHalfGroup) n_warp_id = n_warp_id / 2;
    uint32_t lane_id = threadIdx.x % 32;
    constexpr uint32_t warp_weight_blocks = MAX(WarpShape::N / (ElementA::kBits * 4), 1);
    uint32_t idx = warp_weight_blocks * 32 * n_warp_id + lane_id;

    if constexpr (K_WARPS > 1) {
      uint32_t k_warp_id = (threadIdx.x / (kNumMathThreads / K_WARPS));
      idx = TRUE_N_WARPS * 32 * warp_weight_blocks * kWarpItersK * k_warp_id + idx;
    }

    uint32_t smem_start_idx = idx * kLoadIters;
    smem_ptr = smem_ptr + kSmemStride * iter_id;
    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);
    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < warp_weight_blocks; i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kLoadIters; j++) {
        if constexpr (kLoadHalfGroup) {
          reg_ptr_load[i * kLoadIters + j] = smem_ptr_load[(smem_start_idx + 32 * kLoadIters * i) * 2 + warp_id % 2 * kLoadIters + j];
        } else {
          reg_ptr_load[i * kLoadIters + j] = smem_ptr_load[smem_start_idx + 32 * kLoadIters * i + j];
        }
      }
    }
  };
};
