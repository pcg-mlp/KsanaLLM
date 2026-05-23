#pragma once

#include <humming/utils/all.cuh>


template <class MmaOpClass, class BlockShape, class WarpShape, class ElementA, class TuningConfig>
class S2RMemoryLoaderA {
private:
  using MmaShape = typename MmaOpClass::MmaShape;
  static constexpr uint32_t kNumMathThreads = TuningConfig::kNumMathThreads;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;
  static constexpr uint32_t kWarpItersK = WarpShape::K / kPartMmaShapeK;

public:
  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t m_iter_id = M_WARPS > 1 ? warp_id / N_WARPS % M_WARPS : 0;
    const uint32_t k_warp_id = warp_id / (M_WARPS * N_WARPS);
    constexpr uint32_t row_stride_m_iter = BlockShape::M / M_WARPS;
    uint32_t smem = cast_smem_ptr_to_uint(smem_ptr) / 128;

    PRAGMA_UNROLL
    for (uint32_t load_iter_id = 0; load_iter_id < CEIL_DIV(WarpShape::M, 16); load_iter_id++) {

      uint32_t row = m_iter_id * (BlockShape::M / M_WARPS) + load_iter_id * 16;
      uint32_t col = iter_id * 2 + k_warp_id * (kWarpItersK * 2);

      if constexpr (MmaShape::M == 8) {
        row += (lane_id / 16) * 8 + lane_id % 8;
        col += (lane_id / 8) % 2;
      } else {
        row += lane_id % 16;
        col += lane_id / 16;
      }

      if constexpr (BlockShape::K * ElementA::kBits > 1024) {
        row = BlockShape::M * (col / 8) + row;
        col = (col % 8) ^ ((row + smem) % 8);
      } else if constexpr (BlockShape::K * ElementA::kBits == 1024) {
        col = col ^ ((row + smem) % 8);
      } else if constexpr (BlockShape::K * ElementA::kBits == 512) {
        col = row % 2 * 4 + col;
        row = row / 2;
        col = col ^ ((row + smem) % 4);
      }

      uint32_t a_sh_rd = row * 8 + col;

      if ((load_iter_id == CEIL_DIV(WarpShape::M, 16) - 1) && WarpShape::M % 16 == 8) {
        ld_shared<2>(smem_ptr + a_sh_rd, reinterpret_cast<int4 *>(regs_ptr) + load_iter_id);
      } else {
        ld_shared<4>(smem_ptr + a_sh_rd, reinterpret_cast<int4 *>(regs_ptr) + load_iter_id);
      }
    };
  };
};
