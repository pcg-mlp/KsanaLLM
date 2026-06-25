#pragma once

#include <humming/utils/all.cuh>

template <class MmaOpClass, class BlockShape, class WarpShape, class ElementC, class LayerConfig, class TuningConfig>
class EpilogueSmemReducer {
private:
  static constexpr bool kHasGroupScale = LayerConfig::kWeightScaleGroupSize > 0 || LayerConfig::kInputScaleGroupSize > 0;
  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;
  static constexpr bool kForceFloatAccum = std::is_same<typename MmaOpClass::ValTypeC, int32_t>::value && kHasGroupScale && !LayerConfig::kUseIntWeightScale && !kUseFusedE8m0Scale;
  using MmaTypeC = std::conditional_t<kForceFloatAccum, float, typename MmaOpClass::ValTypeC>;
  using MmaShape = typename MmaOpClass::MmaShape;
  using OutputType32 = std::conditional_t<std::is_same<ElementC, Float16>::value, half2, nv_bfloat162>;
  using ValTypeC32 = std::conditional_t<sizeof(MmaTypeC) == 2, OutputType32, MmaTypeC>;

  static constexpr bool kUseWgmma = MmaOpClass::kMmaType == MmaType::WGMMA;
  using CRegistersType = typename MmaOpClass::CRegisters;
  using MMA_CRegistersArrayType = CRegistersType[MAX(WarpShape::M / MmaShape::M, 1)][MAX(WarpShape::N / MmaShape::N, 1)];
  using WGMMA_CRegistersArrayType = CRegistersType[WarpShape::N * 4 / MmaShape::N][WarpShape::M / MmaShape::M];
  using CRegistersArrayType = std::conditional_t<kUseWgmma, WGMMA_CRegistersArrayType, MMA_CRegistersArrayType>;

public:
  int4 *smem_ptr;

  CUDA_INLINE
  EpilogueSmemReducer(int4 *smem_ptr)
      : smem_ptr(smem_ptr) {
  }

  CUDA_INLINE
  void reduce(uint32_t *regs_ptr) {
    constexpr uint32_t num_int4s = sizeof(CRegistersArrayType) / 16;
    constexpr uint32_t num_int4s_per_time = num_int4s / MAX(WarpShape::M / 16, 1);
    constexpr uint32_t group_num_warps = BlockShape::K / WarpShape::K;
    constexpr uint32_t num_groups = TuningConfig::kNumMathThreads / 32 / group_num_warps;
    uint32_t group_id = threadIdx.x / 32 % num_groups;
    uint32_t group_warp_id = threadIdx.x / (32 * num_groups);
    uint32_t laneid = threadIdx.x % 32;

    using ReductionSmemType = int4[group_num_warps / 2][num_groups][num_int4s_per_time][32];
    auto &smem_arr = *reinterpret_cast<ReductionSmemType *>(smem_ptr);

    auto write_to_smem = [&](uint32_t buffer_id, uint32_t m) {
      int4 *regs_int4_ptr = reinterpret_cast<int4 *>(regs_ptr) + m * num_int4s_per_time;

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < num_int4s_per_time; i++) {
        smem_arr[buffer_id][group_id][i][laneid] = regs_int4_ptr[i];
      };
    };

    auto read_from_smem_and_reduce = [&](uint32_t buffer_id, uint32_t m) {
      int4 *regs_int4_ptr = reinterpret_cast<int4 *>(regs_ptr) + m * num_int4s_per_time;

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < num_int4s_per_time; i++) {
        int4 val = smem_arr[buffer_id][group_id][i][laneid];

        ValTypeC32 *sval_scalar_ptr = reinterpret_cast<ValTypeC32 *>(&val);
        ValTypeC32 *regs_scalar_ptr = reinterpret_cast<ValTypeC32 *>(regs_int4_ptr + i);

        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 4; j++) {
          if constexpr (sizeof(MmaTypeC) == 2) {
            regs_scalar_ptr[j] = __hadd2(regs_scalar_ptr[j], sval_scalar_ptr[j]);
          } else {
            regs_scalar_ptr[j] += sval_scalar_ptr[j];
          }
        }
      };
    };

    PRAGMA_UNROLL
    for (uint32_t m = 0; m < MAX(WarpShape::M / 16, 1); m++) {
      PRAGMA_UNROLL
      for (uint32_t i = 1; i < group_num_warps; i *= 2) {
        uint32_t buffer_id = group_warp_id % (group_num_warps / (2 * i));
        if (group_warp_id >= group_num_warps / i) {
          sync_part_threads<TuningConfig::kNumMathThreads, TuningConfig::kNumThreads>();
        } else if (group_warp_id >= group_num_warps / (2 * i)) {
          write_to_smem(buffer_id, m);
          sync_part_threads<TuningConfig::kNumMathThreads, TuningConfig::kNumThreads>();
        } else {
          sync_part_threads<TuningConfig::kNumMathThreads, TuningConfig::kNumThreads>();
          read_from_smem_and_reduce(buffer_id, m);
        };

        sync_part_threads<TuningConfig::kNumMathThreads, TuningConfig::kNumThreads>();
      };
    };
  };
};
