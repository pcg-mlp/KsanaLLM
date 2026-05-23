#pragma once

#include <humming/memory/s2r_loader/loader_a.cuh>
#include <humming/memory/s2r_loader/loader_as.cuh>
#include <humming/memory/s2r_loader/loader_b.cuh>
#include <humming/memory/s2r_loader/loader_bias.cuh>
#include <humming/memory/s2r_loader/loader_bs.cuh>
#include <humming/memory/s2r_loader/loader_bzp.cuh>

template <
    class SharedStorage, class MMA, class Epilogue,
    class BlockShape, class WarpShape,
    class ElementA, class ElementB, class ElementBS,
    class LayerConfig, class TuningConfig>
class S2RMemoryPipeline {
private:
  static constexpr bool kUseWgmma = MMA::MmaOpClass::kMmaType == MmaType::WGMMA;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kWarpItersK = WarpShape::K / kPartMmaShapeK;
  static constexpr uint32_t kNumStages = TuningConfig::kNumStages;

  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelInputScale = kHasInputScale && LayerConfig::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupInputScale = kHasInputScale && LayerConfig::kInputScaleGroupSize > 0;
  static constexpr bool kIsChannelWeightScale = LayerConfig::kIsChannelWeightScale;
  static constexpr bool kIsGroupWeightScale = LayerConfig::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = LayerConfig::kIsBlockWeightScale;
  static constexpr bool kIsGroupOrBlockWeightScale = kIsGroupWeightScale || kIsBlockWeightScale;

  static constexpr bool kHasZeroPoint = LayerConfig::kHasZeroPoint;
  static constexpr bool kHasBias = LayerConfig::kHasBias;

  using MmaOpClass = typename MMA::MmaOpClass;
  using LoaderA = S2RMemoryLoaderA<MmaOpClass, BlockShape, WarpShape, ElementA, TuningConfig>;
  using LoaderB = S2RMemoryLoaderB<BlockShape, WarpShape, ElementA, ElementB, TuningConfig>;
  using LoaderAS = S2RMemoryLoaderAS<MmaOpClass, BlockShape, WarpShape, ElementA, LayerConfig, TuningConfig>;
  using LoaderBS = S2RMemoryLoaderBS<MmaOpClass, BlockShape, WarpShape, ElementA, ElementBS, LayerConfig, TuningConfig>;
  using LoaderBZP = S2RMemoryLoaderBZP<BlockShape, WarpShape, ElementA, ElementB, LayerConfig, TuningConfig>;
  using LoaderBias = S2RMemoryLoaderBias<MmaOpClass, BlockShape, WarpShape, TuningConfig>;

public:
  SharedStorage &smem;
  MMA &mma;
  Epilogue &epilogue;
  LoaderA loader_a;
  LoaderB loader_b;
  LoaderAS loader_as;
  LoaderBS loader_bs;
  LoaderBZP loader_bzp;
  LoaderBias loader_bias;

  CUDA_INLINE
  S2RMemoryPipeline(SharedStorage &smem, MMA &mma, Epilogue &epilogue)
      : smem(smem), mma(mma), epilogue(epilogue) {
  }

  template <bool kIsFirst = false>
  CUDA_INLINE void load_stage_iter(uint32_t stage_id, uint32_t iter_id) {
    stage_id = (stage_id + iter_id / kWarpItersK) % kNumStages;
    iter_id = iter_id % kWarpItersK;
    uint32_t buffer_id = iter_id % 2;

    loader_b.load(smem.b[stage_id], mma.regs_qb_as_ptr(buffer_id), iter_id);
    if constexpr (!kUseWgmma)
      loader_a.load(smem.a[stage_id], mma.regs_a_as_ptr(buffer_id), iter_id);
    if constexpr (kIsGroupInputScale)
      loader_as.load(smem.as[stage_id], mma.arith.regs_as_as_ptr(buffer_id), iter_id);
    if constexpr (kIsGroupOrBlockWeightScale)
      loader_bs.load(smem.bs[stage_id], mma.arith.regs_bs_as_ptr(buffer_id), iter_id);
    if constexpr (kHasZeroPoint && (kIsGroupOrBlockWeightScale || kIsFirst))
      loader_bzp.load(smem.bzp[stage_id], mma.arith.regs_zp_as_ptr(buffer_id), iter_id);
  }

  CUDA_INLINE void load_channel(uint32_t slice_id) {
    if constexpr (kIsChannelInputScale) loader_as.load(smem.as_c, epilogue.arith.regs_as_as_ptr(), -1);
    if constexpr (kIsChannelWeightScale) loader_bs.load(smem.bs_c, epilogue.arith.regs_bs_as_ptr(), -1);
    if constexpr (kHasBias) loader_bias.load(smem.bias, epilogue.arith.regs_bias_as_ptr(), slice_id == 0);
  }
};
