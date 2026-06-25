#pragma once

#include <humming/utils/all.cuh>


template <typename scalar_t2, typename T>
CUDA_INLINE T reduce_add_f162(T &a, T &b) {
  scalar_t2 *a_half2_ptr = reinterpret_cast<scalar_t2 *>(&a);
  scalar_t2 *b_half2_ptr = reinterpret_cast<scalar_t2 *>(&b);

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < sizeof(T) / 4; i++) {
    a_half2_ptr[i] = __hadd2(a_half2_ptr[i], b_half2_ptr[i]);
  };
  return a;
};


template <typename scalar_t2, typename T>
CUDA_INLINE T atomic_reduce_add_f162(T &a, T &b) {
  scalar_t2 *a_half2_ptr = reinterpret_cast<scalar_t2 *>(&a);
  scalar_t2 *b_half2_ptr = reinterpret_cast<scalar_t2 *>(&b);

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < sizeof(T) / 4; i++) {
    atomicAdd(&b_half2_ptr[i], a_half2_ptr[i]);
  };
  return a;
};

template <
    class SharedStorage, class ArithClass,
    class ProblemShape, class BlockShape, class PadShape, class ElementC,
    class ComputeConfig, class TuningConfig>
class EpilogueGmemWriter : F16Conversion<ElementC> {
private:
  static constexpr bool kUseStreamK = TuningConfig::kUseStreamK;
  static constexpr bool kUseTmaC = TuningConfig::kUseTmaC;

  static constexpr bool kIsDenseGemm = ComputeConfig::kGemmType == GemmType::DENSE;
  static constexpr bool kIsIndexedGemm = ComputeConfig::kGemmType == GemmType::INDEXED;
  static constexpr bool kIsGroupedGemm = ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS || ComputeConfig::kGemmType == GemmType::GROUPED_MASKED;

  static constexpr uint32_t kNumMathThreads = TuningConfig::kNumMathThreads;
  static constexpr uint32_t kNumWriteSplits = TuningConfig::kNumWriteSplits;

  using scalar_t = typename F16Conversion<ElementC>::scalar_t;
  using scalar_t2 = typename F16Conversion<ElementC>::scalar_t2;
  using OutputPtrType = std::conditional_t<kUseTmaC, const void *, void *>;

public:
  ArithClass &arith;
  SharedStorage &smem;
  int4 *smem_ptr;
  int4 *gmem_ptr_raw;
  int4 *gmem_ptr;
  const CUtensorMap *tensor_map_ptr;

  uint32_t row_offset;
  uint32_t col_offset;
  uint32_t output_shape_m;
  uint32_t block_output_shape_m;

  CUDA_INLINE
  EpilogueGmemWriter(
      ArithClass &arith,
      int4 *smem_ptr, OutputPtrType output_ptr,
      SharedStorage &smem, uint32_t shape_m, uint32_t top_k)
      : arith(arith), smem_ptr(smem_ptr), smem(smem) {

    if constexpr (kUseTmaC) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(output_ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<int4 *>(output_ptr);
    }

    output_shape_m = shape_m * (kIsIndexedGemm ? top_k : 1);
  }

  CUDA_INLINE
  void write(uint32_t slice_id, uint32_t slice_count, uint32_t split_idx) {
    if constexpr (kUseTmaC) {
      write_tma(slice_id, slice_count);
    } else {
      write_legacy(slice_id, slice_count, split_idx);
    }
  };

  CUDA_INLINE
  void write_legacy(uint32_t slice_id, uint32_t slice_count, uint32_t split_idx) {
    constexpr uint32_t total_write_int4s = BlockShape::M * BlockShape::N * 2 / 16 / kNumWriteSplits;
    constexpr bool is_full_div = total_write_int4s % kNumMathThreads == 0;
    constexpr uint32_t iters = CEIL_DIV(total_write_int4s, kNumMathThreads);
    uint32_t smem_base = cast_smem_ptr_to_uint(smem_ptr) / 128;

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < iters; i++) {
      uint32_t smem_offset = threadIdx.x + kNumMathThreads * i;
      if (is_full_div || i != iters - 1 || smem_offset < total_write_int4s) {
        uint32_t smem_row = smem_offset / 8;
        uint32_t smem_col = smem_offset % 8;

        uint32_t smem_col_swizzled = smem_col ^ ((smem_row + smem_base) % 8);
        uint32_t smem_offset_swizzled = smem_row * 8 + smem_col_swizzled;
        uint32_t gmem_row = smem_row % (BlockShape::M / kNumWriteSplits);
        if constexpr (kNumWriteSplits == 2) gmem_row += BlockShape::M / 2 * split_idx;
        if constexpr (ComputeConfig::kGemmType == GemmType::INDEXED) gmem_row = smem.wr_row_index[gmem_row];
        uint32_t gmem_col = smem_row / (BlockShape::M / kNumWriteSplits) * 8 + smem_col;
        bool pred1 = gmem_row < (kIsIndexedGemm ? output_shape_m : block_output_shape_m);
        bool pred2 = PadShape::N == 0 || (col_offset + gmem_col * 8 < ProblemShape::N - PadShape::N);

        if (!pred1 || !pred2) continue;

        int4 val = smem_ptr[smem_offset_swizzled];
        uint32_t gmem_offset = gmem_row * ((ProblemShape::N - PadShape::N) / 8) + gmem_col;
        if (!kUseStreamK || slice_count == 1 || slice_id == 0) {
          gmem_ptr[gmem_offset] = val;
        } else if (slice_count > 3) {
          atomic_reduce_add_f162<scalar_t2>(val, gmem_ptr[gmem_offset]);
        } else {
          gmem_ptr[gmem_offset] = reduce_add_f162<scalar_t2>(val, gmem_ptr[gmem_offset]);
        }
      };
    };
  };

  CUDA_INLINE
  void write_tma(uint32_t slice_id, uint32_t slice_count) {
    static_assert(!kIsIndexedGemm);
    constexpr uint32_t count = BlockShape::N / 64;
    const uint32_t block_idx = threadIdx.x;
    const uint32_t smem_offset = BlockShape::M * 64 / 8 * block_idx;
    const uint32_t col_offset2 = col_offset + 64 * block_idx;
    if (block_idx < count) {
      if constexpr (!kUseStreamK) {
        tma_store_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, row_offset);
      } else if (slice_count == 1 || slice_id == 0) {
        tma_store_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, row_offset);
        if (slice_count > 1) tma_wait_store_group<0>();
      } else {
        tma_reduce_add_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, row_offset);
        if (slice_id != slice_count - 1) tma_wait_store_group<0>();
      }
    }
  }

  CUDA_INLINE
  void seek(uint32_t m_block_id, uint32_t n_block_id, uint32_t current_shape_m, uint32_t m_offset) {
    if constexpr (kIsGroupedGemm) {
      output_shape_m = current_shape_m;
      row_offset = m_offset;
    } else {
      row_offset = m_block_id * BlockShape::M;
    }
    block_output_shape_m = MIN(output_shape_m - row_offset, BlockShape::M);
    col_offset = n_block_id * BlockShape::N;

    uint32_t offset;
    offset = n_block_id * (BlockShape::N * 2 / 16);
    if constexpr (!kIsIndexedGemm) {
      constexpr uint32_t kShapeN = ProblemShape::N - PadShape::N;
      offset += row_offset * (kShapeN / 8);
    }
    gmem_ptr = gmem_ptr_raw + offset;
  };

  CUDA_INLINE
  void update_tensor_map_ptr(const CUtensorMap *tensor_map_ptr_) {
    tensor_map_ptr = tensor_map_ptr_;
  };
};
