#pragma once

#include <cooperative_groups.h>
#include <cuda_awbarrier_primitives.h>
#include <humming/memory/g2s_loader/loader_a.cuh>
#include <humming/memory/g2s_loader/loader_as.cuh>
#include <humming/memory/g2s_loader/loader_b.cuh>
#include <humming/memory/g2s_loader/loader_bias.cuh>
#include <humming/memory/g2s_loader/loader_bs.cuh>
#include <humming/memory/g2s_loader/loader_bzp.cuh>


template <
    class SharedStorage,
    class ProblemShape, class BlockShape, class PadShape,
    class ElementA, class ElementB, class ElementBS,
    class LayerConfig, class ComputeConfig, class TuningConfig>
class ProducerPipeline {
private:
  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  static constexpr uint32_t kNumLoadThreads = TuningConfig::kNumLoadThreads;
  static constexpr uint32_t kNumMathThreads = TuningConfig::kNumMathThreads;
  static constexpr uint32_t kLoadThreadOffset = kNumThreads - kNumLoadThreads;

  static constexpr bool kUseMBarrier = TuningConfig::kUseMBarrier;
  static constexpr bool kUseCpAsync = TuningConfig::kUseCpAsync;
  static constexpr bool kUseTma = TuningConfig::kUseTma;
  static constexpr bool kUseTmaA = TuningConfig::kUseTmaA;
  static constexpr bool kUseTmaB = TuningConfig::kUseTmaB;
  static constexpr bool kUseTmaBS = TuningConfig::kUseTmaBS;
  static constexpr bool kUseTmaBZP = TuningConfig::kUseTmaBZP;
  static constexpr bool kUseTmaBias = TuningConfig::kUseTmaBias;

  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelInputScale = kHasInputScale && LayerConfig::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupInputScale = kHasInputScale && LayerConfig::kInputScaleGroupSize > 0;
  static constexpr bool kIsChannelWeightScale = LayerConfig::kIsChannelWeightScale;
  static constexpr bool kIsGroupWeightScale = LayerConfig::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = LayerConfig::kIsBlockWeightScale;
  static constexpr bool kHasZeroPoint = LayerConfig::kHasZeroPoint;
  static constexpr bool kHasBias = LayerConfig::kHasBias;
  static constexpr bool kHasChannelData = kIsChannelInputScale || kIsChannelWeightScale || kHasBias;

  static constexpr uint32_t kNumStages = TuningConfig::kNumStages;

  template <bool kIsFirst = false>
  static constexpr uint2 get_stage_load_bytes() {
    uint32_t tma_load_bytes = 0;
    uint32_t legacy_load_bytes = 0;

    if constexpr (kUseTmaA) tma_load_bytes += SharedStorage::kStageBytesA;
    else legacy_load_bytes += SharedStorage::kStageBytesA;

    if constexpr (kUseTmaB) tma_load_bytes += SharedStorage::kStageBytesB;
    else legacy_load_bytes += SharedStorage::kStageBytesB;

    if constexpr (kIsGroupInputScale) {
      legacy_load_bytes += SharedStorage::kStageBytesAS;
    }

    if constexpr (kIsGroupWeightScale || kIsBlockWeightScale) {
      if constexpr (kUseTmaBS) tma_load_bytes += SharedStorage::kStageBytesBS;
      else legacy_load_bytes += SharedStorage::kStageBytesBS;
    }

    if constexpr (kHasZeroPoint && (kIsGroupWeightScale || kIsFirst)) {
      if constexpr (kUseTmaBZP) tma_load_bytes += SharedStorage::kStageBytesBZP;
      else legacy_load_bytes += SharedStorage::kStageBytesBZP;
    }

    return {tma_load_bytes, legacy_load_bytes};
  }

  static constexpr uint2 get_channel_load_bytes() {
    uint32_t tma_load_bytes = 0;
    uint32_t legacy_load_bytes = 0;

    if constexpr (kIsChannelInputScale) {
      legacy_load_bytes += SharedStorage::kChannelBytesAS;
    }

    if constexpr (kIsChannelWeightScale) {
      if constexpr (kUseTmaBS) tma_load_bytes += SharedStorage::kChannelBytesBS;
      else legacy_load_bytes += SharedStorage::kChannelBytesBS;
    }

    if constexpr (kHasBias) {
      if constexpr (kUseTmaBias) tma_load_bytes += SharedStorage::kBiasBytes;
      else legacy_load_bytes += SharedStorage::kBiasBytes;
    }

    return {tma_load_bytes, legacy_load_bytes};
  }

public:
  static constexpr bool kHasFirstStageTmaMBarrier = get_stage_load_bytes<true>().x > 0;
  static constexpr bool kHasFirstStageCpAsyncMBarrier = get_stage_load_bytes<true>().y > 0;
  static constexpr bool kHasStageTmaMBarrier = get_stage_load_bytes().x > 0;
  static constexpr bool kHasStageCpAsyncMBarrier = get_stage_load_bytes().y > 0;
  static constexpr bool kHasChannelTmaMBarrier = get_channel_load_bytes().x > 0;
  static constexpr bool kHasChannelCpAsyncMBarrier = get_channel_load_bytes().y > 0;
  static constexpr uint32_t kMultiCastSizeA = TuningConfig::kMultiCastSizeA;
  static constexpr uint32_t kMultiCastSizeB = TuningConfig::kMultiCastSizeB;
  static constexpr uint32_t kMultiCastSize = kMultiCastSizeA * kMultiCastSizeB;

  using LoaderA = G2SMemoryLoaderA<SharedStorage, ProblemShape, BlockShape, PadShape, ElementA, ComputeConfig, TuningConfig>;
  using LoaderB = G2SMemoryLoaderB<ProblemShape, BlockShape, ElementA, ElementB, ComputeConfig, TuningConfig>;
  using LoaderAS = G2SMemoryLoaderAS<SharedStorage, ProblemShape, BlockShape, PadShape, ElementA, LayerConfig, ComputeConfig, TuningConfig>;
  using LoaderBS = G2SMemoryLoaderBS<ProblemShape, BlockShape, ElementBS, LayerConfig, TuningConfig>;
  using LoaderBZP = G2SMemoryLoaderBZP<ProblemShape, BlockShape, ElementB, LayerConfig, TuningConfig>;
  using LoaderBias = G2SMemoryLoaderBias<ProblemShape, BlockShape, TuningConfig>;

  SharedStorage &smem;
  LoaderA loader_a;
  LoaderB loader_b;
  LoaderAS loader_as;
  LoaderBS loader_bs;
  LoaderBZP loader_bzp;
  LoaderBias loader_bias;
  uint32_t phases[TuningConfig::kNumStages + 1] = {0};
  const uint32_t thread_id = threadIdx.x - kLoadThreadOffset;
  uint32_t cluster_rank = blockIdx.x % kMultiCastSize;

  CUDA_INLINE
  ProducerPipeline(
      SharedStorage &smem,
      const void *void_ptr_a,
      const void *void_ptr_b,
      const void *void_ptr_as,
      const void *void_ptr_bs,
      const void *void_ptr_bzp,
      const void *void_ptr_bias,
      uint32_t shape_m)
      : smem(smem),
        loader_a(void_ptr_a, smem, shape_m),
        loader_b(void_ptr_b),
        loader_as(void_ptr_as, smem, shape_m),
        loader_bs(void_ptr_bs),
        loader_bzp(void_ptr_bzp),
        loader_bias(void_ptr_bias) {

    if (thread_id == 0) {
      if constexpr (kUseTmaA) prefetch_tensor_map(void_ptr_a);
      if constexpr (kUseTmaB) prefetch_tensor_map(void_ptr_b);
      if constexpr (kUseTmaBS) prefetch_tensor_map(void_ptr_bs);
      if constexpr (kUseTmaBZP) prefetch_tensor_map(void_ptr_bzp);
      if constexpr (kUseTmaBias) prefetch_tensor_map(void_ptr_bias);
    }
    __syncwarp();
  }

  CUDA_INLINE void init_mbarrier() {
    if constexpr (kUseMBarrier) {
      uint32_t count;
      if (thread_id < kNumStages) {
        constexpr uint32_t cp_async_thread_count = kHasStageCpAsyncMBarrier ? kNumLoadThreads : 0;
        constexpr uint32_t tma_thread_count = kHasStageTmaMBarrier ? 1 : 0;
        count = cp_async_thread_count + tma_thread_count;
      } else if (thread_id == kNumStages) {
        constexpr uint32_t cp_async_thread_count = kHasFirstStageCpAsyncMBarrier ? kNumLoadThreads : 0;
        constexpr uint32_t tma_thread_count = kHasFirstStageTmaMBarrier ? 1 : 0;
        count = cp_async_thread_count + tma_thread_count;
      } else if (thread_id == kNumStages + 1) {
        constexpr uint32_t cp_async_thread_count = kHasChannelCpAsyncMBarrier ? kNumLoadThreads : 0;
        constexpr uint32_t tma_thread_count = kHasChannelTmaMBarrier ? 1 : 0;
        count = cp_async_thread_count + tma_thread_count;
      }

      if (thread_id < kNumStages + 2) __mbarrier_init(&smem.load_mbar[thread_id], count);
      uint32_t factor = (kMultiCastSize > 1 && cluster_rank == 0 && thread_id < kNumStages) ? kMultiCastSize : 1;
      if constexpr (TuningConfig::kUseWarpSpec) {
        if (thread_id < kNumStages + 1) __mbarrier_init(&smem.math_mbar[thread_id], TuningConfig::kNumMathThreads * factor / 32);
      }
    }
  }

  template <bool kShouldAdvance = true, bool kIsFirst = false>
  CUDA_INLINE void load_stage(uint32_t stage_id, bool pred = true) {
    stage_id = stage_id % kNumStages;

    uint32_t mbar_index = kIsFirst ? kNumStages : stage_id;

    uint2 load_bytes;
    if (pred) {
      uint64_t *mbar_ptr = nullptr;
      if constexpr (kUseMBarrier) mbar_ptr = &smem.load_mbar[mbar_index];
      loader_a.template load<kShouldAdvance>(smem.a[stage_id], mbar_ptr);
      loader_b.template load<kShouldAdvance>(smem.b[stage_id], mbar_ptr);
      if constexpr (kIsGroupInputScale) {
        loader_as.template load<kShouldAdvance>(smem.as[stage_id], mbar_ptr);
      };
      if constexpr (kIsGroupWeightScale || kIsBlockWeightScale) {
        loader_bs.template load<kShouldAdvance>(smem.bs[stage_id], mbar_ptr);
      };
      if constexpr (kHasZeroPoint && (kIsGroupWeightScale || kIsFirst)) {
        loader_bzp.template load<kShouldAdvance>(smem.bzp[stage_id], mbar_ptr);
      }
      load_bytes = get_stage_load_bytes<kIsFirst>();
    }

    if constexpr (kIsFirst) {
      commit_load<kHasFirstStageCpAsyncMBarrier, kHasFirstStageTmaMBarrier>(mbar_index, load_bytes, pred);
    } else {
      commit_load<kHasStageCpAsyncMBarrier, kHasStageTmaMBarrier>(mbar_index, load_bytes, pred);
    }
  }

  CUDA_INLINE void load_channel() {
    uint64_t *channel_mbar_ptr = nullptr;
    if constexpr (kUseMBarrier) channel_mbar_ptr = &smem.load_mbar[kNumStages + 1];
    if constexpr (kIsChannelInputScale) loader_as.load(smem.as_c, channel_mbar_ptr);
    if constexpr (kIsChannelWeightScale) loader_bs.load(smem.bs_c, channel_mbar_ptr);
    if constexpr (kHasBias) loader_bias.load(smem.bias, channel_mbar_ptr);

    constexpr uint2 load_bytes = get_channel_load_bytes();
    if constexpr (load_bytes.x > 0 || load_bytes.y > 0) {
      commit_load<kHasChannelCpAsyncMBarrier, kHasChannelTmaMBarrier>(kNumStages + 1, load_bytes);
    }
  }

  template <bool kHasCpAsyncMBarrier, bool kHasTmaMBarrier>
  CUDA_INLINE void commit_load(uint32_t stage_id, uint2 load_bytes, bool pred = true) {
    if constexpr (kUseMBarrier) {
      if (!pred) return;
      if constexpr (kHasCpAsyncMBarrier) {
        cp_async_commit_mbarrier(&smem.load_mbar[stage_id]);
      }
      if constexpr (kHasTmaMBarrier) {
        if (thread_id == 0) tma_commit_mbarrier(&smem.load_mbar[stage_id], load_bytes.x);
        __syncwarp();
      }
    } else if constexpr (kUseCpAsync) {
      cp_async_commit_group();
    }
  }

  CUDA_INLINE void wait_stage(uint32_t stage_id) {
    mbarrier_wait(&smem.math_mbar[stage_id], phases[stage_id]);
    phases[stage_id] ^= 1;
  }

  CUDA_INLINE void wait_channel() {
    if constexpr (kHasChannelData && kUseMBarrier) {
      mbarrier_wait(&smem.math_mbar[kNumStages], phases[kNumStages]);
      phases[kNumStages] ^= 1;
    }
  }

  CUDA_INLINE void wait_math_epilogue() {
    mbarrier_wait(&smem.math_mbar[kNumStages], phases[kNumStages]);
    phases[kNumStages] ^= 1;
  }

  CUDA_INLINE void seek(
      uint32_t expert_id, uint32_t m_block_id, uint32_t n_block_id, uint32_t k_block_id,
      uint32_t current_shape_m, uint32_t m_offset) {
    loader_a.seek(m_block_id, k_block_id, current_shape_m, m_offset);
    loader_b.seek(expert_id, n_block_id, k_block_id);
    loader_as.seek(m_block_id, k_block_id, current_shape_m, m_offset);
    loader_bs.seek(expert_id, n_block_id, k_block_id);
    loader_bzp.seek(expert_id, n_block_id, k_block_id);
    loader_bias.seek(expert_id, n_block_id);
  }
};


template <
    class SharedStorage, class ElementA,
    class LayerConfig, class TuningConfig>
class ConsumerPipeline {
private:
  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  static constexpr uint32_t kNumMathThreads = TuningConfig::kNumMathThreads;

  static constexpr bool kUseMBarrier = TuningConfig::kUseMBarrier;
  static constexpr bool kUseCpAsync = TuningConfig::kUseCpAsync;

  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelInputScale = kHasInputScale && LayerConfig::kInputScaleGroupSize == 0;
  static constexpr bool kIsChannelWeightScale = LayerConfig::kIsChannelWeightScale;
  static constexpr bool kHasBias = LayerConfig::kHasBias;
  static constexpr bool kHasChannelData = kIsChannelInputScale || kIsChannelWeightScale || kHasBias;

  static constexpr uint32_t kNumStages = TuningConfig::kNumStages;
  static constexpr uint32_t kMultiCastSizeA = TuningConfig::kMultiCastSizeA;
  static constexpr uint32_t kMultiCastSizeB = TuningConfig::kMultiCastSizeB;
  static constexpr uint32_t kMultiCastSize = kMultiCastSizeA * kMultiCastSizeB;

public:
  SharedStorage &smem;
  uint32_t phases[TuningConfig::kNumStages + 2] = {0};
  uint32_t cluster_rank = blockIdx.x % kMultiCastSize;
  const uint32_t lane_id = threadIdx.x % 32;

  CUDA_INLINE
  ConsumerPipeline(SharedStorage &smem)
      : smem(smem) {
  }

  CUDA_INLINE void init_mbarrier() {
  }

  template <bool kIsFirst = false>
  CUDA_INLINE void wait_stage(uint32_t stage_id) {
    stage_id = kIsFirst ? kNumStages : (stage_id % kNumStages);
    if constexpr (kUseMBarrier) {
      mbarrier_wait(&smem.load_mbar[stage_id], phases[stage_id]);
      phases[stage_id] ^= 1;
    } else if constexpr (kUseCpAsync) {
      cp_async_wait_group<kNumStages - 2>();
      __syncthreads();
    } else {
      __syncthreads();
    }
  }

  CUDA_INLINE void wait_channel() {
    if constexpr (kHasChannelData) {
      if constexpr (kUseMBarrier) {
        mbarrier_wait(&smem.load_mbar[kNumStages + 1], phases[kNumStages + 1]);
        phases[kNumStages + 1] ^= 1;
      } else if constexpr (kUseCpAsync) {
        cp_async_wait_group<0>();
        __syncthreads();
      } else {
        __syncthreads();
      }
    }
  }

  CUDA_INLINE void arrive(uint32_t stage_id) {
    if (lane_id == 0) {
      mbarrier_arrive(&smem.math_mbar[stage_id]);
      if constexpr (kMultiCastSize > 1) {
        if (cluster_rank >= 1 && stage_id < kNumStages) {
          void *aa = __cluster_map_shared_rank(&smem.math_mbar[stage_id], 0);
          mbarrier_arrive<true>(aa);
        }
      }
    }
    __syncwarp();
  }
};
