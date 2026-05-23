#pragma once

#include <humming/utils/all.cuh>


template <
    class ProblemShape, class BlockShape,
    class ElementA, class ElementB,
    class ComputeConfig, class TuningConfig>
class G2SMemoryLoaderB {
private:
  static constexpr bool kUseWarpSpec = TuningConfig::kUseWarpSpec;
  static constexpr bool kUseTma = TuningConfig::kUseTmaB;
  static constexpr bool kUseCpAsync = TuningConfig::kUseCpAsync;
  static constexpr uint32_t kNumLoadThreads = TuningConfig::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = TuningConfig::kNumThreads - kNumLoadThreads;
  static constexpr uint32_t kMultiCastSizeB = TuningConfig::kMultiCastSizeB;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kSmemStride = BlockShape::N * kPartMmaShapeK * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kGmemStride = ProblemShape::N * kPartMmaShapeK * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kGmemExpertStride = ProblemShape::N * ProblemShape::K * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kNumInt4s = kSmemStride * BlockShape::K / kPartMmaShapeK;

public:
  const CUtensorMap *tensor_map_ptr;
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  uint32_t row_offset;
  uint32_t col_offset;
  uint32_t cluster_rank = blockIdx.x % kMultiCastSizeB;

  CUDA_INLINE
  G2SMemoryLoaderB(const void *ptr) {
    if constexpr (kUseTma) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<const int4 *>(ptr);
    }
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(int4 *smem_ptr, void *mbar_ptr) {
    if constexpr (kUseTma) load_tma(smem_ptr, mbar_ptr);
    else load_legacy(smem_ptr);
    if constexpr (kShouldAdvance) advance();
  }

  CUDA_INLINE
  void load_tma(int4 *smem_ptr, void *mbar_ptr) {
    if (threadIdx.x == kLoadThreadOffset) {
      if constexpr (kMultiCastSizeB == 1) {
        tma_load_3d(tensor_map_ptr, smem_ptr, mbar_ptr, 0, col_offset, row_offset);
      } else if (cluster_rank == 0) {
        tma_load_3d<kMultiCastSizeB>(tensor_map_ptr, smem_ptr, mbar_ptr, 0, col_offset, row_offset);
      }
    }
  }

  CUDA_INLINE
  void load_legacy(int4 *smem_ptr) {
    legacy_load_2d<
        kUseCpAsync, kNumInt4s, kNumLoadThreads,
        kGmemStride, kSmemStride, kLoadThreadOffset>(gmem_ptr, smem_ptr);
  }

  CUDA_INLINE
  void advance() {
    row_offset += BlockShape::K / kPartMmaShapeK;
    gmem_ptr += kGmemStride * BlockShape::K / kPartMmaShapeK;
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t n_block_id, uint32_t k_block_id) {
    row_offset = expert_id * (ProblemShape::K / kPartMmaShapeK) + k_block_id * (BlockShape::K / kPartMmaShapeK);
    col_offset = n_block_id * (BlockShape::N / 32);

    uint64_t gmem_offset = expert_id * kGmemExpertStride;
    gmem_offset += n_block_id * kSmemStride + k_block_id * (kGmemStride * BlockShape::K / kPartMmaShapeK);
    gmem_ptr = gmem_ptr_raw + gmem_offset;
  }
};
