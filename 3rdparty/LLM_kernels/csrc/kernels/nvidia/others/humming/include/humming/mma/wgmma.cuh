#pragma once

#include <humming/utils/all.cuh>


template <uint32_t swizzle_bytes = 128>
CUDA_INLINE uint64_t make_wgmma_smem_desc(void *smem_ptr, uint32_t iter_id) {
  static_assert(swizzle_bytes == 128 || swizzle_bytes == 64);

  constexpr uint64_t swizzle_type = swizzle_bytes == 128 ? 1 : 2;
  constexpr uint64_t stride = (swizzle_bytes * 8) >> 4;
  constexpr uint64_t desc_base = (swizzle_type << 62) | (stride << 32);

  uint32_t addr = cast_smem_ptr_to_uint(smem_ptr);
  uint64_t desc = desc_base;

  reinterpret_cast<uint32_t *>(&desc)[0] = (addr >> 4);

  return desc;
};


template <
    class MmaOpClass_, class SharedStorage, class ArithClass,
    class BlockShape, class WarpShape,
    class ElementA, class ElementB,
    class LayerConfig>
struct WGMMA {
public:
  using MmaOpClass = MmaOpClass_;
  using MmaShape = typename MmaOpClass::MmaShape;

  static constexpr bool kHasZeroPoint = LayerConfig::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = LayerConfig::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;
  static constexpr uint32_t kWarpItersK = WarpShape::K / (256 / ElementA::kBits);
  static constexpr uint32_t kSwizzleBytes = ElementA::kBits * BlockShape::K >= 1024 ? 128 : 64;
  static constexpr uint32_t kNumWarpShapeNSplits = WarpShape::N == ElementA::kBits * 2 ? 2 : 1;

  SharedStorage &smem;
  ArithClass &arith;
  uint32_t regs_qb[2][ElementB::kBits * (16 / ElementA::kBits)];
  typename MmaOpClass::BRegisters regs_b[2][WarpShape::N * 4 / MmaShape::N][kPartMmaShapeK / MmaShape::K];
  typename MmaOpClass::CRegisters regs_c[2][WarpShape::N * 4 / MmaShape::N][WarpShape::M / MmaShape::M];
  uint32_t smem_offset = 0;

  CUDA_INLINE
  WGMMA(SharedStorage &smem, ArithClass &arith)
      : smem(smem), arith(arith) {
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t m_warp_id = warp_id / N_WARPS % M_WARPS;
    uint32_t k_warp_id = warp_id / (N_WARPS * M_WARPS);

    constexpr uint32_t kSwizzleSizeK = kSwizzleBytes * 8 / ElementA::kBits;
    static_assert(kSwizzleSizeK >= WarpShape::K);

    const uint32_t row_offset = M_WARPS > 1 ? WarpShape::M * m_warp_id : 0;
    const uint32_t col_offset = K_WARPS > 1 ? WarpShape::K * k_warp_id : 0;

    smem_offset = row_offset * (kSwizzleBytes / 16);
    smem_offset += (col_offset % kSwizzleSizeK) * ElementA::kBits / 128;
    smem_offset += (col_offset / kSwizzleSizeK) * (BlockShape::M * kSwizzleBytes / 16);
  }

  CUDA_INLINE
  void zero_accum() {
    uint32_t *regs_c_ptr = regs_c_as_ptr();
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs_c) / 4; i++) {
      regs_c_ptr[i] = 0;
    };
  };

  CUDA_INLINE
  void transform_b(uint32_t buffer_id) {
    if constexpr (std::is_same<ElementA, ElementB>::value) return;

    if constexpr (kUseFusedE8m0Scale) {
      uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id]);
      fused_dequant_for_mxfp4<ElementA, WarpShape::N / 16, true>(regs_qb[buffer_id], regs_b_ptr, arith.bs[buffer_id]);
    } else {
      if constexpr (ElementB::kBits == 1 && kNumWarpShapeNSplits == 2) {
        regs_qb[buffer_id][0] = regs_qb[buffer_id][0] >> (threadIdx.x / 32 % 2 * 8);
      }

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < WarpShape::N / (MmaShape::N / 4); i++) {
        uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][i * 64 / MmaShape::N]);
        uint4 zp_vals = arith.prepare_zp_for_dequant(buffer_id, i);
        uint32_t *zp_vals_ptr = reinterpret_cast<uint32_t *>(&zp_vals);
        dequant<ElementB, ElementA, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(regs_qb[buffer_id], regs_b_ptr, i, zp_vals_ptr);
        arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
      };
    }
  };

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    static_assert(WarpShape::M == MmaShape::M);
    uint32_t buffer_id = iter_id % 2;

    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kPartMmaShapeK / MmaShape::K; k++) {
      int4 *smem_ptr = smem.a[stage_id] + smem_offset + iter_id * 2 + k;
      uint64_t desc = make_wgmma_smem_desc<kSwizzleBytes>(smem_ptr, iter_id);

      constexpr uint32_t kNumIters = WarpShape::N / (MmaShape::N / 4);

      bool scale_d = true;
      constexpr bool kApplyScaleOnC = !kUseFusedE8m0Scale && ElementA::kBits != 16 &&
          (LayerConfig::kInputScaleGroupSize > 0 || LayerConfig::kWeightScaleGroupSize > 0);
      if constexpr (!kUseFusedE8m0Scale && ElementA::kBits != 16 && LayerConfig::kInputScaleGroupSize > 0) {
        scale_d = (iter_id * kPartMmaShapeK) % LayerConfig::kInputScaleGroupSize > 0;
      }
      if constexpr (!kUseFusedE8m0Scale && ElementA::kBits != 16 && LayerConfig::kWeightScaleGroupSize > 0) {
        scale_d = scale_d && (iter_id * kPartMmaShapeK) % LayerConfig::kWeightScaleGroupSize > 0;
      }

      wgmma_fence();
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++) {
        if constexpr (kApplyScaleOnC) fence_regs(regs_c[0][j][0]);
        MmaOpClass::fma(desc, regs_b[buffer_id][j][k], regs_c[0][j][0], scale_d);
        wgmma_commit();
        wgmma_wait<0>();
        if constexpr (kApplyScaleOnC) fence_regs(regs_c[0][j][0]);
        arith.may_apply_as_and_bs_on_wgmma_c(regs_c_as_ptr(), j, k, iter_id);
      }
    }
  };

  template <class T>
  CUDA_INLINE void fence_regs(T &regs) {
    PRAGMA_UNROLL
    for (uint32_t r = 0; r < sizeof(T) / 4; r++) {
      warpgroup_fence_operand(reinterpret_cast<uint32_t *>(regs)[r]);
    }
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_qb_as_ptr(uint32_t buffer_id) {
    if constexpr (std::is_same<ElementA, ElementB>::value) {
      return reinterpret_cast<T *>(regs_b[buffer_id]);
    } else {
      return reinterpret_cast<T *>(regs_qb[buffer_id]);
    };
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr(uint32_t buffer_id = 0) {
    return reinterpret_cast<T *>(regs_c[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    uint32_t index = 0;
    constexpr bool kIsGroupInputScale = LayerConfig::kInputScaleGroupSize > 0;
    constexpr bool kIsGroupWeightScale = LayerConfig::kIsGroupWeightScale;
    constexpr bool kIsBlockWeightScale = LayerConfig::kIsBlockWeightScale;

    if constexpr (ElementA::kBits < 16 && !kUseFusedE8m0Scale && kIsGroupInputScale) {
      index = 1;
    }

    if constexpr (ElementA::kBits < 16 && !kUseFusedE8m0Scale && (kIsGroupWeightScale || kIsBlockWeightScale)) {
      index = 1;
    }

    return regs_c_as_ptr<T>(index);
  };
};
