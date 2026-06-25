#pragma once

#include <humming/scheduler.cuh>
#include <humming/utils/all.cuh>

#include <humming/arith/epilogue_arith.cuh>
#include <humming/arith/mainloop_arith.cuh>

#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#include <humming/mma/wgmma.cuh>
#include <humming/mma/wmma.cuh>

#include <humming/datatype/dequant.cuh>


template <bool kUseTma>
class KernelTensorParamType {
public:
  using Type = std::conditional_t<kUseTma, CUtensorMap const, void *const>;
};


template <
    class MmaOpClass,
    class ProblemShape, class BlockShape, class WarpShape, class PadShape,
    class ElementA, class ElementB, class ElementC, class ElementBS,
    class LayerConfig, class ComputeConfig, class TuningConfig>
__global__ __launch_bounds__(TuningConfig::kNumThreads, TuningConfig::kNumCtasPerSm) void humming(
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaA>::Type A,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaB>::Type B,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaC>::Type C,
    const uint32_t *AS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBS>::Type BS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBZP>::Type BZP,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBias>::Type Bias,
    const uint32_t *GS,
    const uint32_t *sorted_ids_ptr,
    const uint32_t *expert_ids_ptr,
    const uint32_t *num_tokens_padded_ptr,
    const uint32_t *expert_layout_ptr,
    CUtensorMap *tensor_map_buffer,
    int32_t *locks,
    uint32_t shape_m,
    uint32_t top_k,
    bool use_int64_expert_layout) {

  constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  constexpr uint32_t kNumStages = TuningConfig::kNumStages;

  using SharedStorage = SharedStorage<
      MmaOpClass, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using Scheduler = Scheduler<
      SharedStorage, ProblemShape, BlockShape,
      LayerConfig, ComputeConfig, TuningConfig>;
  using ProducerPipeline = ProducerPipeline<
      SharedStorage, ProblemShape, BlockShape, PadShape, ElementA, ElementB, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using ConsumerPipeline = ConsumerPipeline<SharedStorage, ElementA, LayerConfig, TuningConfig>;
  using MainloopArithmetic = MainloopArithmetic<
      MmaOpClass, BlockShape, WarpShape,
      ElementA, ElementB, ElementC, ElementBS, LayerConfig>;
  using EpilogueArithmetic = EpilogueArithmetic<
      MmaOpClass, BlockShape, WarpShape,
      ElementA, ElementB, ElementC, ElementBS,
      LayerConfig, TuningConfig>;
  using WMMA = WMMA<MmaOpClass, SharedStorage, MainloopArithmetic, WarpShape, ElementA, ElementB, LayerConfig>;
  using WGMMA = WGMMA<MmaOpClass, SharedStorage, MainloopArithmetic, BlockShape, WarpShape, ElementA, ElementB, LayerConfig>;
  using MMA = std::conditional_t<MmaOpClass::kMmaType == MmaType::WGMMA, WGMMA, WMMA>;
  using Epilogue = EpiloguePipeline<
      MmaOpClass, SharedStorage, EpilogueArithmetic, ProblemShape, BlockShape, WarpShape, PadShape,
      ElementA, ElementC, LayerConfig, ComputeConfig, TuningConfig>;
  using S2RMemoryPipeline = S2RMemoryPipeline<
      SharedStorage, MMA, Epilogue, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      LayerConfig, TuningConfig>;

  extern __shared__ int4 shared_memory[];
  auto &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  auto pa = [&]() {if constexpr (TuningConfig::kUseTmaA) return &A; else return A; };
  auto pb = [&]() {if constexpr (TuningConfig::kUseTmaB) return &B; else return B; };
  auto pc = [&]() {if constexpr (TuningConfig::kUseTmaC) return &C; else return C; };
  auto pas = [&]() { return AS; };
  auto pbs = [&]() {if constexpr (TuningConfig::kUseTmaBS) return &BS; else return BS; };
  auto pbzp = [&]() {if constexpr (TuningConfig::kUseTmaBZP) return &BZP; else return BZP; };
  auto pbias = [&]() {if constexpr (TuningConfig::kUseTmaBias) return &Bias; else return Bias; };
  auto scheduler = Scheduler(smem, pc(), tensor_map_buffer, shape_m, top_k, sorted_ids_ptr, expert_ids_ptr, num_tokens_padded_ptr, expert_layout_ptr, use_int64_expert_layout);
  auto mainloop_arith = MainloopArithmetic();
  auto epilogue_arith = EpilogueArithmetic();
  auto mma = MMA(smem, mainloop_arith);
  auto epilogue = Epilogue(smem, pc(), tensor_map_buffer, epilogue_arith, GS, locks, shape_m, top_k);
  auto producer = ProducerPipeline(smem, pa(), pb(), pas(), pbs(), pbzp(), pbias(), shape_m);
  auto consumer = ConsumerPipeline(smem);
  auto s2r_pipe = S2RMemoryPipeline(smem, mma, epilogue);

  producer.init_mbarrier();
  __syncthreads();

  while (scheduler.get_next_block()) {
    mma.zero_accum();
    __syncthreads();

    uint32_t &slice_iters = scheduler.slice_iters;
    producer.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.k_block_id, scheduler.current_shape_m, scheduler.m_offset);
    epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.current_shape_m, scheduler.m_offset);
    epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);

    if constexpr (TuningConfig::kUseTmaC) tma_wait_store_group<0, true>();
    producer.template load_stage<true, true>(0);
    PRAGMA_UNROLL
    for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
      producer.load_stage(stage_id, stage_id < slice_iters);
    };

    consumer.template wait_stage<true>(kNumStages);
    s2r_pipe.template load_stage_iter<true>(0, 0);
    mma.transform_b(0);

    while (slice_iters) {
      PRAGMA_UNROLL
      for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
        if (slice_iters == 1) producer.load_channel();
        constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
        constexpr uint32_t warp_k_iters = WarpShape::K / kPartMmaShapeK;

        PRAGMA_UNROLL
        for (uint32_t warp_k_iter_id = 0; warp_k_iter_id < warp_k_iters; warp_k_iter_id++) {
          s2r_pipe.load_stage_iter(stage_id, warp_k_iter_id + 1);
          mma.run(stage_id, warp_k_iter_id);
          if (warp_k_iter_id == warp_k_iters - 2) {
            if constexpr (kNumStages == 2) {
              __syncthreads();
              if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
              producer.load_stage(stage_id, slice_iters > kNumStages);
            } else {
              producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
              if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            }
          }

          mma.transform_b((warp_k_iter_id + 1) % 2);
        }

        slice_iters--;
        if (!slice_iters) break;
      };
    };

    consumer.wait_channel();
    s2r_pipe.load_channel(scheduler.slice_id);
    __syncthreads();
    epilogue.call(mma.final_regs_c_as_ptr());
  }
};
