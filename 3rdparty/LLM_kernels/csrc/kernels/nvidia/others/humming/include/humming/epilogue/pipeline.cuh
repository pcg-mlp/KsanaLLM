#pragma once

#include <humming/utils/all.cuh>

#include <humming/epilogue/gmem_writer.cuh>
#include <humming/epilogue/smem_reducer.cuh>
#include <humming/epilogue/smem_writer.cuh>


template <
    class MmaOpClass, class SharedStorage, class ArithClass,
    class ProblemShape, class BlockShape, class WarpShape, class PadShape,
    class ElementA, class ElementC,
    class LayerConfig, class ComputeConfig, class TuningConfig>
class EpiloguePipeline {
private:
  using SmemReducer = EpilogueSmemReducer<MmaOpClass, BlockShape, WarpShape, ElementC, LayerConfig, TuningConfig>;
  using SmemWriter = EpilogueSmemWriter<MmaOpClass, ArithClass, BlockShape, WarpShape, ElementA, ElementC, LayerConfig, TuningConfig>;
  using GmemWriter = EpilogueGmemWriter<SharedStorage, ArithClass, ProblemShape, BlockShape, PadShape, ElementC, ComputeConfig, TuningConfig>;
  using OutputPtrType = std::conditional_t<TuningConfig::kUseTmaC, const void *, void *>;

  static constexpr bool kIsGroupedGemm = ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS || ComputeConfig::kGemmType == GemmType::GROUPED_MASKED;
  static constexpr uint32_t kNumWriteSplits = TuningConfig::kNumWriteSplits;

public:
  SmemReducer smem_reducer;
  SmemWriter smem_writer;
  GmemWriter gmem_writer;
  ArithClass &arith;
  const uint32_t *GS;
  int32_t *locks;

  uint32_t slice_count;
  uint32_t slice_id;
  uint32_t locks_offset;

  CUDA_INLINE
  EpiloguePipeline(
      SharedStorage &smem, OutputPtrType output_ptr, CUtensorMap *tensor_map_buffer, ArithClass &arith,
      const uint32_t *GS, int32_t *locks, uint32_t output_shape_m, uint32_t top_k)
      : GS(GS), locks(locks), arith(arith),
        smem_reducer(smem.reduce), smem_writer(smem.reduce, arith),
        gmem_writer(arith, smem.reduce, output_ptr, smem, output_shape_m, top_k) {
    if constexpr (TuningConfig::kUseTmaC) {
      if constexpr (kIsGroupedGemm) gmem_writer.update_tensor_map_ptr(tensor_map_buffer + blockIdx.x);
      else if (threadIdx.x == 0) prefetch_tensor_map(output_ptr);
    }
    sync_math_threads();
  }

  CUDA_INLINE
  void call(uint32_t *regs_c_ptr) {
    sync_math_threads();
    if constexpr (BlockShape::K > WarpShape::K) smem_reducer.reduce(regs_c_ptr);
    static_assert(kNumWriteSplits == 1 || kNumWriteSplits == 2);
    if constexpr (kNumWriteSplits > 1) {
      static_assert(BlockShape::M == WarpShape::M);
      static_assert(BlockShape::M % 32 == 0);
      static_assert(!TuningConfig::kUseTmaC);
    }

    if (slice_count > 1) acquire_gmem_barrier();
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kNumWriteSplits; i++) {
      smem_writer.write(regs_c_ptr, slice_count, i);
      sync_math_threads();
      gmem_writer.write(slice_id, slice_count, i);
      sync_math_threads();
    }
    if (slice_count > 1) release_gmem_barrier();
  }

  CUDA_INLINE
  void sync_math_threads() {
    sync_part_threads<TuningConfig::kNumMathThreads, TuningConfig::kNumThreads>();
  }

  CUDA_INLINE
  void acquire_gmem_barrier() {
    if (TuningConfig::kUseTmaC || slice_count > 3) {
      int32_t val = slice_id == 0 ? 0 : -1;
      barrier_acquire2<TuningConfig::kNumMathThreads, TuningConfig::kNumThreads>(&locks[locks_offset], val);
    } else {
      barrier_acquire<TuningConfig::kNumMathThreads, TuningConfig::kNumThreads>(&locks[locks_offset], slice_id);
    }
  }

  CUDA_INLINE
  void release_gmem_barrier() {
    if (TuningConfig::kUseTmaC || slice_count > 3) {
      int32_t val = slice_id == 0 ? 1 - static_cast<int32_t>(slice_count) : 0;
      barrier_release2<TuningConfig::kNumMathThreads, TuningConfig::kNumThreads>(&locks[locks_offset], val);
    } else {
      barrier_release<TuningConfig::kNumMathThreads, TuningConfig::kNumThreads>(&locks[locks_offset], slice_id == slice_count - 1);
    }
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t m_block_id, uint32_t n_block_id, uint32_t current_shape_m, uint32_t m_offset) {
    gmem_writer.seek(m_block_id, n_block_id, current_shape_m, m_offset);
    if constexpr (LayerConfig::kIsTensorWeightScale) {
      arith.gs = GS[ComputeConfig::kGemmType == GemmType::DENSE ? 0 : expert_id];
    }
  };

  CUDA_INLINE
  void set_streamk_state(uint32_t slice_count_, uint32_t slice_id_, uint32_t locks_offset_) {
    slice_count = slice_count_;
    slice_id = slice_id_;
    locks_offset = locks_offset_;
  };
};
