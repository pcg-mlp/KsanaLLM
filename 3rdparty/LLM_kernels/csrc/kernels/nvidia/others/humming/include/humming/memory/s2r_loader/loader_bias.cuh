#pragma once

#include <humming/utils/all.cuh>


template <class MmaOpClass, class BlockShape, class WarpShape, class TuningConfig>
class S2RMemoryLoaderBias {
private:
  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  static constexpr bool kUseWgmma = MmaOpClass::kMmaType == MmaType::WGMMA;

  static constexpr uint32_t kSmemStride = BlockShape::N * 16 / 32 / 4;
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

  static constexpr uint32_t kLoadBytes = WarpShape::N / 4 * 16 / 8;
  static constexpr uint32_t kLoadIters = kLoadBytes / 16;

public:
  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t pred) {
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t bias_sh_rd;
    if constexpr (kUseWgmma) {
      bias_sh_rd = (threadIdx.x % 32) / 8;
    } else {
      bias_sh_rd = threadIdx.x % 4;
    }

    if constexpr (WarpShape::N == 16) {
      bias_sh_rd = (warp_id % N_WARPS / 2) * 8 + bias_sh_rd * 2 + warp_id % 2;

      const int2 *smem_ptr_load = reinterpret_cast<const int2 *>(smem_ptr) + bias_sh_rd;
      int2 *reg_ptr_load = reinterpret_cast<int2 *>(regs_ptr);

      reg_ptr_load[0] = pred ? smem_ptr_load[0] : int2();

    } else {
      bias_sh_rd += warp_id % N_WARPS * (WarpShape::N / 16 * 2);

      const int4 *smem_ptr_load = smem_ptr + bias_sh_rd;
      int4 *reg_ptr_load = reinterpret_cast<int4 *>(regs_ptr);

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kLoadIters; i++) {
        reg_ptr_load[i] = pred ? smem_ptr_load[i * 4] : int4();
      };
    }
  }
};
