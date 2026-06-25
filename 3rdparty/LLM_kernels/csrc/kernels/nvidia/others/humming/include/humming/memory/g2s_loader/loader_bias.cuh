#pragma once

#include <humming/utils/all.cuh>


template <class ProblemShape, class BlockShape, class TuningConfig>
class G2SMemoryLoaderBias {
private:
  static constexpr bool kUseWarpSpec = TuningConfig::kUseWarpSpec;
  static constexpr bool kUseTma = TuningConfig::kUseTmaBias;
  static constexpr bool kUseCpAsync = TuningConfig::kUseCpAsync;
  static constexpr uint32_t kNumLoadThreads = TuningConfig::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = TuningConfig::kNumThreads - kNumLoadThreads;

  static constexpr uint32_t kSmemStride = BlockShape::N * 16 / 32 / 4;
  static constexpr uint32_t kGmemStride = ProblemShape::N * 16 / 32 / 4;
  static constexpr uint32_t kNumInt4s = kSmemStride;

public:
  const CUtensorMap *tensor_map_ptr;
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  uint32_t offset;

  CUDA_INLINE
  G2SMemoryLoaderBias(const void *ptr) {
    if constexpr (kUseTma) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<const int4 *>(ptr);
    }
  }

  CUDA_INLINE
  void load(int4 *smem_ptr, void *mbar_ptr) {
    if constexpr (kUseTma) load_tma(smem_ptr, mbar_ptr);
    else load_legacy(smem_ptr);
  }

  CUDA_INLINE
  void load_tma(int4 *smem_ptr, void *mbar_ptr) {
    if (threadIdx.x == kLoadThreadOffset) tma_load_2d(tensor_map_ptr, smem_ptr, mbar_ptr, 0, offset);
  }

  CUDA_INLINE
  void load_legacy(int4 *smem_ptr) {
    legacy_load_1d<kUseCpAsync, kNumInt4s, kNumLoadThreads, kLoadThreadOffset>(gmem_ptr, smem_ptr);
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t n_block_id) {
    offset = expert_id * (ProblemShape::N / 64) + n_block_id * (BlockShape::N / 64);
    uint32_t gmem_offset = expert_id * kGmemStride + n_block_id * kSmemStride;
    gmem_ptr = gmem_ptr_raw + gmem_offset;
  }
};
