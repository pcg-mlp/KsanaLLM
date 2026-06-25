#pragma once

#include <humming/utils/all.cuh>


template <
    class SharedStorage, class ProblemShape, class BlockShape, class PadShape,
    class ElementA, class ComputeConfig, class TuningConfig>
class G2SMemoryLoaderA {
private:
  static constexpr bool kUseWarpSpec = TuningConfig::kUseWarpSpec;
  static constexpr bool kUseTma = TuningConfig::kUseTmaA;
  static constexpr bool kUseCpAsync = TuningConfig::kUseCpAsync;
  static constexpr bool kIsDenseGemm = ComputeConfig::kGemmType == GemmType::DENSE;
  static constexpr bool kIsIndexedGemm = ComputeConfig::kGemmType == GemmType::INDEXED;
  static constexpr bool kIsGroupedGemm = ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS || ComputeConfig::kGemmType == GemmType::GROUPED_MASKED;

  static constexpr uint32_t kNumLoadThreads = TuningConfig::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = TuningConfig::kNumThreads - kNumLoadThreads;
  static constexpr uint32_t kMultiCastSizeA = TuningConfig::kMultiCastSizeA;

  static constexpr uint32_t kSmemStride = BlockShape::K * ElementA::kBits / 32 / 4;
  static constexpr uint32_t kGmemStride = (ProblemShape::K - PadShape::K) * ElementA::kBits / 32 / 4;
  static constexpr uint32_t kNumInt4s = kSmemStride * BlockShape::M;

  static_assert(BlockShape::K * ElementA::kBits >= 512);
  static constexpr uint32_t kSwizzleBytes = BlockShape::K * ElementA::kBits == 512 ? 64 : 128;
  static constexpr uint32_t kNumTmaLoadsPerLine = CEIL_DIV(BlockShape::K * ElementA::kBits, kSwizzleBytes * 8);
  static constexpr uint32_t kLoadIters = CEIL_DIV(kNumInt4s, kNumLoadThreads);

public:
  SharedStorage &smem;
  const uint32_t thread_id = threadIdx.x - kLoadThreadOffset;
  const CUtensorMap *tensor_map_ptr;
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  uint32_t shape_m;
  uint32_t block_shape_m;
  uint32_t load_row_index[kUseTma ? CEIL_DIV(BlockShape::M, kNumLoadThreads) : kLoadIters];
  uint32_t row_offset;
  uint32_t col_offset;
  uint32_t cluster_rank = blockIdx.x % kMultiCastSizeA;

  CUDA_INLINE
  G2SMemoryLoaderA(const void *ptr, SharedStorage &smem, uint32_t shape_m)
      : smem(smem), shape_m(shape_m) {
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
    if (thread_id < kNumTmaLoadsPerLine) {
      const uint32_t block_idx = thread_id;
      const uint32_t smem_offset = BlockShape::M * 8 * block_idx;
      const uint32_t col_offset2 = col_offset + (1024 / ElementA::kBits) * block_idx;
      if constexpr (kMultiCastSizeA == 1) {
        tma_load_2d(tensor_map_ptr, smem_ptr + smem_offset, mbar_ptr, col_offset2, row_offset);
      } else if (cluster_rank == 0) {
        tma_load_2d<kMultiCastSizeA>(tensor_map_ptr, smem_ptr + smem_offset, mbar_ptr, col_offset2, row_offset);
      }
    }
  }

  CUDA_INLINE
  void load_legacy(int4 *smem_ptr) {
    if constexpr (kSwizzleBytes == 128) {
      load_legacy_swizzled_128B(smem_ptr);
    } else if constexpr (kSwizzleBytes == 64) {
      load_legacy_swizzled_64B(smem_ptr);
    }
  }

  CUDA_INLINE
  void load_legacy_swizzled_128B(int4 *smem_ptr) {
    static_assert(BlockShape::K * ElementA::kBits >= 1024);
    uint32_t smem_base = cast_smem_ptr_to_uint(smem_ptr);
    uint32_t smem_swizzled_col = (thread_id % 8) ^ (((thread_id % 64) / 8 + smem_base / 128)) % 8;

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kLoadIters; i++) {
      uint32_t smem_offset = i * kNumLoadThreads + thread_id;
      uint32_t smem_row = smem_offset / 8;
      uint32_t smem_col = smem_offset % 8;
      uint32_t smem_swizzled_offset = smem_row * 8 + smem_swizzled_col;

      uint32_t gmem_col = smem_row / BlockShape::M * 8 + smem_col;
      uint32_t gmem_row = kIsIndexedGemm ? load_row_index[i] : (smem_row % BlockShape::M);
      uint32_t gmem_offset = gmem_row * kGmemStride + gmem_col;

      bool pred0 = (gmem_col * (128 / ElementA::kBits) + col_offset) < (ProblemShape::K - PadShape::K);
      bool pred1 = kNumInt4s % kNumLoadThreads == 0 || i != kLoadIters - 1 || smem_offset < kNumInt4s;
      bool pred2 = gmem_row < (kIsIndexedGemm ? shape_m : block_shape_m);

      if constexpr (PadShape::K == 0) {
        legacy_load_pred<kUseCpAsync>(gmem_ptr + gmem_offset, smem_ptr + smem_swizzled_offset, pred1 && pred2);
      } else {
        legacy_load_zfill_pred<kUseCpAsync>(gmem_ptr + gmem_offset, smem_ptr + smem_swizzled_offset, pred0, pred1 && pred2);
      }
    }
  }

  CUDA_INLINE
  void load_legacy_swizzled_64B(int4 *smem_ptr) {
    static_assert(BlockShape::K * ElementA::kBits == 512);
    uint32_t smem_uint = cast_smem_ptr_to_uint(smem_ptr);
    uint32_t smem_swizzled_col = (thread_id % 8) ^ (((thread_id % 32) / 8 + smem_uint / 128) % 4);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kLoadIters; i++) {
      uint32_t smem_offset = i * kNumLoadThreads + thread_id;
      uint32_t smem_row = smem_offset / 8;
      uint32_t smem_col = smem_offset % 8;
      uint32_t smem_swizzled_offset = smem_row * 8 + smem_swizzled_col;

      uint32_t gmem_row = smem_row % (BlockShape::M / 2) * 2 + smem_col / 4;
      gmem_row = kIsIndexedGemm ? load_row_index[i] : gmem_row;
      uint32_t gmem_col = smem_col % 4;
      uint32_t gmem_offset = gmem_row * kGmemStride + gmem_col;

      bool pred0 = (gmem_col * (128 / ElementA::kBits) + col_offset) < (ProblemShape::K - PadShape::K);
      bool pred1 = kNumInt4s % kNumLoadThreads == 0 || i != kLoadIters - 1 || smem_offset < kNumInt4s;
      bool pred2 = gmem_row < (kIsIndexedGemm ? shape_m : block_shape_m);
      if constexpr (PadShape::K == 0) {
        legacy_load_pred<kUseCpAsync>(gmem_ptr + gmem_offset, smem_ptr + smem_swizzled_offset, pred1 && pred2);
      } else {
        legacy_load_zfill_pred<kUseCpAsync>(gmem_ptr + gmem_offset, smem_ptr + smem_swizzled_offset, pred0, pred1 && pred2);
      }
    }
  }

  CUDA_INLINE
  void advance() {
    col_offset += BlockShape::K;
    gmem_ptr += kSmemStride;
  }

  CUDA_INLINE
  void seek(uint32_t m_block_id, uint32_t k_block_id, uint32_t current_shape_m, uint32_t m_offset) {
    if constexpr (kIsGroupedGemm) {
      shape_m = current_shape_m;
      row_offset = m_offset;
    } else {
      row_offset = m_block_id * BlockShape::M;
    }
    col_offset = k_block_id * BlockShape::K;
    block_shape_m = MIN(shape_m - row_offset, BlockShape::M);

    uint32_t gmem_offset = k_block_id * kSmemStride;
    gmem_offset += kIsIndexedGemm ? 0 : (row_offset * kGmemStride);
    gmem_ptr = gmem_ptr_raw + gmem_offset;

    if constexpr (kIsIndexedGemm) {
      static_assert(!kUseTma);

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kLoadIters; i++) {
        uint32_t smem_offset = i * kNumLoadThreads + thread_id;
        uint32_t smem_row = smem_offset / 8;
        uint32_t smem_col = smem_offset % 8;
        uint32_t gmem_row;

        if constexpr (BlockShape::K * ElementA::kBits >= 1024) {
          gmem_row = smem_row % BlockShape::M;
        } else {
          gmem_row = smem_row % (BlockShape::M / 2) * 2 + smem_col / 4;
        }
        load_row_index[i] = smem.rd_row_index[gmem_row];
      }
    }
  }
};
