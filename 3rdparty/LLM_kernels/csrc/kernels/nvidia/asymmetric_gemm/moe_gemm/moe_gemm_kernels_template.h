/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "csrc/kernels/nvidia/cutlass_extensions/compute_occupancy.h"
#include "csrc/kernels/nvidia/cutlass_extensions/epilogue_helpers.h"
#include "csrc/kernels/nvidia/cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "csrc/kernels/nvidia/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "csrc/kernels/nvidia/cutlass_extensions/gemm/threadblock/default_mma.h"

#pragma GCC diagnostic pop

// NOTE(karlluo): prevent conflict with cutlass we import cutlass and its extension first
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>
#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_heuristic.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/moe_gemm/moe_gemm_kernels.h"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {

namespace nvidia {

// ============================= Variable batched Gemm things ===========================
template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
          typename WarpShape, int32_t Stages>
void GenericMoeGemmKernelLauncher(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                                  int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k,
                                  int32_t num_experts, cutlass_extensions::CutlassGemmConfig gemm_config,
                                  const int32_t multi_processor_count, cudaStream_t stream,
                                  int32_t* kernel_occupancy = nullptr) {
#ifdef ENABLE_BF16
  static_assert(cutlass::platform::is_same<T, __nv_bfloat16>::value || cutlass::platform::is_same<T, half>::value ||
                    cutlass::platform::is_same<T, float>::value,
                "Specialized for bfloat16, half, float");
#else
  static_assert(cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, float>::value,
                "Specialized for half, float");
#endif

  static_assert(cutlass::platform::is_same<T, WeightType>::value ||
                    cutlass::platform::is_same<WeightType, uint8_t>::value ||
                    cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
                "");

  // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
  using ElementType_ =
      typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
  using ElementType =
      typename cutlass::platform::conditional<cutlass::platform::is_same<ElementType_, __nv_bfloat16>::value,
                                              cutlass::bfloat16_t, ElementType_>::type;
#else
  using ElementType = ElementType_;
#endif

  using CutlassWeightType_ =
      typename cutlass::platform::conditional<cutlass::platform::is_same<WeightType, half>::value, cutlass::half_t,
                                              WeightType>::type;
#ifdef ENABLE_BF16
  using CutlassWeightType =
      typename cutlass::platform::conditional<cutlass::platform::is_same<CutlassWeightType_, __nv_bfloat16>::value,
                                              cutlass::bfloat16_t, CutlassWeightType_>::type;
#else
  using CutlassWeightType = CutlassWeightType_;
#endif

  // We need separate config for each architecture since we will target different tensorcore instructions. For float,
  // we do not target TCs.
  using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
  using ElementAccumulator = typename MixedGemmArchTraits::AccType;

  using EpilogueOp =
      typename llm_kernels::nvidia::cutlass_extensions::Epilogue<ElementType, MixedGemmArchTraits::ElementsPerAccessC,
                                                                 ElementAccumulator, EpilogueTag>::Op;

  // Finally, set up the kernel.
  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      ElementType, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, MixedGemmArchTraits::ElementsPerAccessA,
      CutlassWeightType, typename MixedGemmArchTraits::LayoutB, cutlass::ComplexTransform::kNone,
      MixedGemmArchTraits::ElementsPerAccessB, ElementType, cutlass::layout::RowMajor, ElementAccumulator,
      typename MixedGemmArchTraits::OperatorClass, arch, ThreadblockShape, WarpShape,
      typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, typename MixedGemmArchTraits::Operator>::GemmKernel;

  using GemmKernel = cutlass::gemm::kernel::MoeFCGemm<typename GemmKernel_::Mma, typename GemmKernel_::Epilogue,
                                                      typename GemmKernel_::ThreadblockSwizzle,
                                                      arch,  // Ensure top level arch is used for dispatch
                                                      GemmKernel_::kGroupScheduleMode>;

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  if (kernel_occupancy != nullptr) {
    *kernel_occupancy = llm_kernels::nvidia::cutlass_extensions::ComputeOccupancyForKernel<GemmKernel>();
    return;
  }
  int32_t occupancy = std::min(2, GemmGrouped::maximum_active_blocks());
  if (occupancy <= 0) {
    throw std::invalid_argument("GPU lacks the shared memory resources to run GroupedGEMM kernel");
  }
  const int32_t threadblock_count = multi_processor_count * occupancy;

  typename EpilogueOp::Params epilogue_op(ElementAccumulator(1.f),
                                          biases ? ElementAccumulator(1.f) : ElementAccumulator(0.f));

  const int32_t group_size = gemm_k;
  typename GemmGrouped::Arguments args(
      num_experts, threadblock_count, group_size, epilogue_op, reinterpret_cast<const ElementType*>(A),
      reinterpret_cast<const CutlassWeightType*>(B), reinterpret_cast<const ElementType*>(weight_scales),
      reinterpret_cast<const ElementType*>(biases), reinterpret_cast<ElementType*>(C), total_rows_before_expert, gemm_n,
      gemm_k);

  GemmGrouped gemm;

  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    throw std::runtime_error("MoE FC kernel will fail for params. Error: " +
                             std::string(cutlass::cutlassGetStatusString(can_implement)));
  }

  auto init_status = gemm.initialize(args);
  if (can_implement != cutlass::Status::kSuccess) {
    throw std::runtime_error("Failed to initialize cutlass variable batched gemm. Error: " +
                             std::string(cutlass::cutlassGetStatusString(init_status)));
  }

  auto run_status = gemm.run(stream);
  if (can_implement != cutlass::Status::kSuccess) {
    throw std::runtime_error("Failed to run cutlass variable batched gemm. Error: " +
                             std::string(cutlass::cutlassGetStatusString(run_status)));
  }
}

template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
          typename WarpShape, int32_t Stages, typename Enable = void>
struct DispatchStages {
  static void Dispatch(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                       int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k,
                       int32_t num_experts, cutlass_extensions::CutlassGemmConfig gemm_config,
                       int32_t multi_processor_count, cudaStream_t stream, int32_t* occupancy = nullptr) {
    throw std::runtime_error("Cutlass fpA_intB gemm. Not instantiated for arch " +
                             std::to_string(arch::kMinComputeCapability) + " with stages set to " +
                             std::to_string(Stages));
  }
};

template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
          typename WarpShape>
struct DispatchStages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2> {
  static void Dispatch(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                       int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k,
                       int32_t num_experts, cutlass_extensions::CutlassGemmConfig gemm_config,
                       int32_t multi_processor_count, cudaStream_t stream, int32_t* occupancy = nullptr) {
    GenericMoeGemmKernelLauncher<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>(
        A, B, weight_scales, biases, C, total_rows_before_expert, num_rows, gemm_n, gemm_k, num_experts, gemm_config,
        multi_processor_count, stream, occupancy);
  }
};

template <typename T, typename WeightType, typename EpilogueTag, typename ThreadblockShape, typename WarpShape,
          int32_t Stages>
struct DispatchStages<T, WeightType, cutlass::arch::Sm80, EpilogueTag, ThreadblockShape, WarpShape, Stages,
                      typename std::enable_if<(Stages > 2)>::type> {
  static void Dispatch(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                       int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k,
                       int32_t num_experts, cutlass_extensions::CutlassGemmConfig gemm_config,
                       int32_t multi_processor_count, cudaStream_t stream, int32_t* occupancy = nullptr) {
    GenericMoeGemmKernelLauncher<T, WeightType, cutlass::arch::Sm80, EpilogueTag, ThreadblockShape, WarpShape, Stages>(
        A, B, weight_scales, biases, C, total_rows_before_expert, num_rows, gemm_n, gemm_k, num_experts, gemm_config,
        multi_processor_count, stream, occupancy);
  }
};

template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
          typename WarpShape>
void DispatchGemmConfig(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                        int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k,
                        int32_t num_experts, cutlass_extensions::CutlassGemmConfig gemm_config,
                        int32_t multi_processor_count, cudaStream_t stream, int32_t* occupancy = nullptr) {
  switch (gemm_config.stages) {
    case 2:
      using DispatcherStages2 = DispatchStages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>;
      DispatcherStages2::Dispatch(A, B, weight_scales, biases, C, total_rows_before_expert, num_rows, gemm_n, gemm_k,
                                  num_experts, gemm_config, multi_processor_count, stream, occupancy);
      break;
    case 3:
      using DispatcherStages3 = DispatchStages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 3>;
      DispatcherStages3::Dispatch(A, B, weight_scales, biases, C, total_rows_before_expert, num_rows, gemm_n, gemm_k,
                                  num_experts, gemm_config, multi_processor_count, stream, occupancy);
      break;
    case 4:
      using DispatcherStages4 = DispatchStages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 4>;
      DispatcherStages4::Dispatch(A, B, weight_scales, biases, C, total_rows_before_expert, num_rows, gemm_n, gemm_k,
                                  num_experts, gemm_config, multi_processor_count, stream, occupancy);
      break;
    default:
      throw std::invalid_argument("DispatchGemmConfig does not support stages " + std::to_string(gemm_config.stages));
      break;
  }
}

// This overload will handle tensorop gemms. It is disabled via SFINAE for fp32.
// This overload is only enabled when T == WeightType.
template <
    typename T, typename WeightType, typename arch, typename EpilogueTag,
    typename std::enable_if<!std::is_same<T, float>::value && std::is_same<T, WeightType>::value>::type* = nullptr>
void DispatchMoeGemmToCutlass(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                              int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                              int32_t num_experts, cutlass_extensions::CutlassGemmConfig gemm_config,
                              int32_t sm_version, int32_t multi_processor_count, cudaStream_t stream,
                              int32_t* occupancy = nullptr) {
  switch (gemm_config.tile_config) {
    case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      DispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
                         cutlass::gemm::GemmShape<32, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                               total_rows, gemm_n, gemm_k, num_experts, gemm_config,
                                                               multi_processor_count, stream, occupancy);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
      DispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
                         cutlass::gemm::GemmShape<32, 64, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                               total_rows, gemm_n, gemm_k, num_experts, gemm_config,
                                                               multi_processor_count, stream, occupancy);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
      DispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 64>,
                         cutlass::gemm::GemmShape<64, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                               total_rows, gemm_n, gemm_k, num_experts, gemm_config,
                                                               multi_processor_count, stream, occupancy);
      break;
    case cutlass_extensions::CutlassTileConfig::Undefined:
      throw std::invalid_argument("GEMM config undefined.");
      break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
      throw std::invalid_argument("GEMM config should have already been set by heuristic.");
      break;
    default:
      throw std::invalid_argument("Config is invalid for same type tensorop GEMM.");
      break;
  }
}

// Tensorop GEMM overload
// Overload for quantize MoE GEMMs. We disable some warp configs here since they will not be used and we can improve
// compile time
template <
    typename T, typename WeightType, typename arch, typename EpilogueTag,
    typename std::enable_if<!std::is_same<T, float>::value && !std::is_same<T, WeightType>::value>::type* = nullptr>
void DispatchMoeGemmToCutlass(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                              int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                              int32_t num_experts, cutlass_extensions::CutlassGemmConfig gemm_config,
                              int32_t sm_version, int32_t multi_processor_count, cudaStream_t stream,
                              int32_t* occupancy = nullptr) {
  switch (gemm_config.tile_config) {
    case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      DispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
                         cutlass::gemm::GemmShape<32, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                               total_rows, gemm_n, gemm_k, num_experts, gemm_config,
                                                               multi_processor_count, stream, occupancy);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
      DispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
                         cutlass::gemm::GemmShape<64, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                               total_rows, gemm_n, gemm_k, num_experts, gemm_config,
                                                               multi_processor_count, stream, occupancy);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
      DispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 64>,
                         cutlass::gemm::GemmShape<128, 32, 64>>(
          A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts,
          gemm_config, multi_processor_count, stream, occupancy);
      break;
    case cutlass_extensions::CutlassTileConfig::Undefined:
      throw std::invalid_argument("GEMM config undefined.");
      break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
      throw std::invalid_argument("GEMM config should have already been set by heuristic.");
      break;
    default:
      throw std::invalid_argument("Config is invalid for same type tensorop GEMM.");
      break;
  }
}

// This overload will handle simt gemms. It is disabled via SFINAE for tensorop.
template <typename T, typename WeightType, typename arch, typename EpilogueTag,
          typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
void DispatchMoeGemmToCutlass(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                              int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                              int32_t num_experts, cutlass_extensions::CutlassGemmConfig gemm_config,
                              int32_t sm_version, int32_t multi_processor_count, cudaStream_t stream,
                              int32_t* occupancy = nullptr) {
  switch (gemm_config.tile_config) {
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
      DispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 8>,
                         cutlass::gemm::GemmShape<64, 64, 8>>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                              total_rows, gemm_n, gemm_k, num_experts, gemm_config,
                                                              multi_processor_count, stream, occupancy);
      break;
    case cutlass_extensions::CutlassTileConfig::Undefined:
      throw std::invalid_argument("GEMM config undefined.");
      break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
      throw std::invalid_argument("GEMM config should have already been set by heuristic.");
      break;
    default:
      throw std::invalid_argument("Unsupported config for float MoE gemm.");
      break;
  }
}

template <typename T, typename WeightType>
std::vector<cutlass_extensions::CutlassGemmConfig> MoeGemmRunner<T, WeightType>::GetConfigs() {
  static constexpr bool is_weight_only = !std::is_same<T, WeightType>::value;
  static constexpr bool only_simt_configs = std::is_same<T, float>::value;
  std::vector<cutlass_extensions::CutlassGemmConfig> candidate_configs =
      llm_kernels::nvidia::GetCandidateConfigs(sm_, is_weight_only, only_simt_configs);
  return candidate_configs;
}

template <typename T, typename WeightType>
MoeGemmRunner<T, WeightType>::MoeGemmRunner() {
  int32_t device{-1};
  CHECK_NVIDIA_CUDA_ERROR(cudaGetDevice(&device));
  sm_ = GetSMVersion();
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::DispatchToArch<EpilogueTag>(const T* A, const WeightType* B, const T* weight_scales,
                                                               const T* biases, T* C, int64_t* total_rows_before_expert,
                                                               int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                                                               int32_t num_experts,
                                                               cutlass_extensions::CutlassGemmConfig gemm_config,
                                                               cudaStream_t stream, int32_t* occupancy) {
  if (sm_ >= NVIDIA_VOLTA_GPU_COMPUTE_CAPABILITY && sm_ < NVIDIA_TURING_GPU_COMPUTE_CAPABILITY) {
    DispatchMoeGemmToCutlass<T, WeightType, cutlass::arch::Sm70, EpilogueTag>(
        A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config,
        sm_, multi_processor_count_, stream, occupancy);
  } else if (sm_ >= NVIDIA_TURING_GPU_COMPUTE_CAPABILITY && sm_ < NVIDIA_AMPERE_GPU_COMPUTE_CAPABILITY) {
    DispatchMoeGemmToCutlass<T, WeightType, cutlass::arch::Sm75, EpilogueTag>(
        A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config,
        sm_, multi_processor_count_, stream, occupancy);
  } else if (sm_ >= NVIDIA_AMPERE_GPU_COMPUTE_CAPABILITY && sm_ < NVIDIA_HOPPER_GPU_COMPUTE_CAPABILITY) {
    DispatchMoeGemmToCutlass<T, WeightType, cutlass::arch::Sm80, EpilogueTag>(
        A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config,
        sm_, multi_processor_count_, stream, occupancy);
  } else if (sm_ >= NVIDIA_HOPPER_GPU_COMPUTE_CAPABILITY) {
    // TODO Update the arch to Sm90 once CUTLASS hopper specialisations are available
    DispatchMoeGemmToCutlass<T, WeightType, cutlass::arch::Sm80, EpilogueTag>(
        A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config,
        sm_, multi_processor_count_, stream, occupancy);
  } else {
    throw std::invalid_argument("Arch unsupported for MoE GEMM");
  }
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::RunGemm<EpilogueTag>(const T* A, const WeightType* B, const T* weight_scales,
                                                        const T* biases, T* C, int64_t* total_rows_before_expert,
                                                        int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                                                        int32_t num_experts, cudaStream_t stream) {
  auto chosen_conf = this->best_config_;
  if (!chosen_conf) {
    auto candidate_configs = GetConfigs();
    std::vector<int32_t> occupancies(candidate_configs.size());

    for (size_t ii = 0; ii < candidate_configs.size(); ++ii) {
      DispatchToArch<EpilogueTag>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k,
                                  num_experts, candidate_configs[ii], stream, &occupancies[ii]);
    }

    static constexpr int32_t workspace_bytes = 0;  // No workspace for MoE GEMMs.
    static constexpr int32_t split_k_limit = 1;    // MoE GEMM does not support split-k.

    static constexpr bool is_weight_only = !std::is_same<T, WeightType>::value;
    chosen_conf = llm_kernels::nvidia::EstimateBestConfigFromOccupancies(
        candidate_configs, occupancies, total_rows, gemm_n, gemm_k, num_experts, split_k_limit, workspace_bytes,
        multi_processor_count_, is_weight_only);
  }
  assert(chosen_conf);
  DispatchToArch<EpilogueTag>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k,
                              num_experts, *chosen_conf, stream);
}

template <typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::InvokeMoeGemmWithBiasAct(const T* A, const WeightType* B, const T* weight_scales,
                                                            const T* biases, T* C, int64_t* total_rows_before_expert,
                                                            int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                                                            int32_t num_experts, ActivationType activation_type,
                                                            cudaStream_t stream) {
  switch (activation_type) {
    case ActivationType::Relu:
      RunGemm<cutlass_extensions::EpilogueOpDefaultReLU>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                         total_rows, gemm_n, gemm_k, num_experts, stream);
      break;
    case ActivationType::Gelu:
      RunGemm<cutlass_extensions::EpilogueOpDefaultFtGelu>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                           total_rows, gemm_n, gemm_k, num_experts, stream);
      break;
    case ActivationType::Silu:
      RunGemm<cutlass_extensions::EpilogueOpDefaultSilu>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                         total_rows, gemm_n, gemm_k, num_experts, stream);
      break;
    case ActivationType::Identity:
      RunGemm<cutlass_extensions::EpilogueOpDefault>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                     total_rows, gemm_n, gemm_k, num_experts, stream);
      break;
    case ActivationType::InvalidType:
      throw std::invalid_argument("Activation type for fpA_intB must be valid.");
      break;
    default:
      throw std::invalid_argument("Invalid activation type.");
      break;
  }
}

template <typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::InvokeMoeGemm(const T* A, const WeightType* B, const T* weight_scales, T* C,
                                                 int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n,
                                                 int64_t gemm_k, int32_t num_experts, cudaStream_t stream) {
  RunGemm<cutlass_extensions::EpilogueOpDefault>(A, B, weight_scales, nullptr, C, total_rows_before_expert, total_rows,
                                                 gemm_n, gemm_k, num_experts, stream);
}

}  // namespace nvidia

}  // namespace llm_kernels
