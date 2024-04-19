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

#pragma once

#ifndef _WIN32
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // #ifndef _WIN32

// clang-format off
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm.h>

#include <csrc/kernels/nvidia/cutlass_extensions/gemm/device/gemm_universal_base_compat.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
// clang-format on

#include "csrc/kernels/nvidia/cutlass_extensions/compute_occupancy.h"
#include "csrc/kernels/nvidia/cutlass_extensions/epilogue/threadblock/epilogue_per_row_per_col_scale.h"
#include "csrc/kernels/nvidia/cutlass_extensions/epilogue/threadblock/epilogue_tensor_op_int32.h"
#include "csrc/kernels/nvidia/cutlass_extensions/epilogue_helpers.h"
#include "csrc/kernels/nvidia/cutlass_extensions/gemm_configs.h"

#include "csrc/kernels/nvidia/cutlass_extensions/gemm/kernel/default_int8_traits.h"
#include "csrc/kernels/nvidia/cutlass_extensions/gemm/kernel/gemm_with_epilogue_visitor.h"

#ifndef _WIN32
#  pragma GCC diagnostic pop
#endif  // #ifndef _WIN32

#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_heuristic.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/int8_gemm/int8_gemm.h"
#include "csrc/utils/nvidia/cuda_utils.h"

// NOTE(karlluo): prevent conflict with cutlass we import cutlass and its extension first
#include <chrono>
#include <sstream>

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T, typename arch, typename ThreadblockShape, typename WarpShape, int32_t Stages>
void GenericInt8GemmKernelLauncher(const int8_t* A, const int8_t* B, llm_kernels::utils::QuantMode quant_option,
                                   const float* alpha_col, const float* alpha_row, T* C, int32_t m, int32_t n,
                                   int32_t k, llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemm_config,
                                   char* workspace, size_t workspace_bytes, cudaStream_t stream,
                                   int32_t* occupancy = nullptr) {
  using ElementInput = int8_t;

  using ElementOutput_ =
      typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
  using ElementOutput =
      typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput_, __nv_bfloat16>::value,
                                              cutlass::bfloat16_t, ElementOutput_>::type;
#else
  using ElementOutput = ElementOutput_;
#endif

  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using OperatorClass = typename cutlass::gemm::kernel::Int8GemmArchTraits<arch>::OperatorClass;
  using InstructionShape = typename cutlass::gemm::kernel::Int8GemmArchTraits<arch>::InstructionShape;

  using DefaultGemmConf =
      typename cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass, arch, ElementInput, ElementInput,
                                                               ElementOutput, ElementCompute>;
  using GemmOp = typename DefaultGemmConf::Operator;
  using EpilogueOp = typename DefaultGemmConf::EpilogueOutputOp;

  // only TN is supported (s8 * s8 + s32)
  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<
      ElementInput, cutlass::layout::RowMajor, DefaultGemmConf::kAlignmentA, ElementInput, cutlass::layout::ColumnMajor,
      DefaultGemmConf::kAlignmentB, ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, OperatorClass, arch,
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOp, ThreadblockSwizzle, Stages, true, GemmOp>::GemmKernel;

  using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
      cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
          GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
          GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess, cutlass::sizeof_bits<ElementOutput>::value>,
      ElementCompute>;

  // Epilogue visitor
  using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
      ThreadblockShape, GemmKernel_::kThreadCount, AlphaColTileIterator,
      typename GemmKernel_::Epilogue::OutputTileIterator, ElementAccumulator, ElementCompute, EpilogueOp>;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
      EpilogueVisitor, typename GemmKernel_::Epilogue>::Epilogue;

  // GEMM
  using GemmKernel =
      cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

  if (occupancy != nullptr) {
    *occupancy = llm_kernels::nvidia::cutlass_extensions::ComputeOccupancyForKernel<GemmKernel>();
    return;
  }

  using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

  typename EpilogueOp::Params linear_scaling_params;  // TODO: right now it's unused (scaling is done in
                                                      // visitor, no activation needed)
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kBatched,
                                {m, n, k},
                                1,
                                {reinterpret_cast<ElementInput*>(const_cast<ElementInput*>(A)), k},
                                {reinterpret_cast<ElementInput*>(const_cast<ElementInput*>(B)), k},
                                quant_option,
                                {reinterpret_cast<ElementCompute*>(const_cast<float*>(alpha_col)), 0},
                                {reinterpret_cast<ElementCompute*>(const_cast<float*>(alpha_row)), 0},
                                {nullptr, 0},
                                {reinterpret_cast<ElementOutput*>(C), n},
                                0,
                                0,
                                typename EpilogueVisitor::Arguments(linear_scaling_params, 0, 0, 0)};

  Gemm gemm;
  // TODO: handle that
  if (gemm.get_workspace_size(args) > workspace_bytes) {
    // If requested split-k factor will require more workspace bytes, revert to standard gemm.
    args.batch_count = 1;
  }

  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg = "int8gemm cutlass kernel will fail for params. Error: " +
                          std::string(cutlass::cutlassGetStatusString(can_implement));
    throw std::runtime_error("[int8gemm Runner] " + err_msg);
  }

  auto init_status = gemm.initialize(args, workspace, stream);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to initialize cutlass int8 gemm. Error: " + std::string(cutlass::cutlassGetStatusString(init_status));
    throw std::runtime_error("[int8gemm Runner] " + err_msg);
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to run cutlass int8 gemm. Error: " + std::string(cutlass::cutlassGetStatusString(run_status));
    throw std::runtime_error("[int8gemm Runner] " + err_msg);
  }
}

template <typename T, typename arch, typename ThreadblockShape, typename WarpShape, int32_t Stages,
          typename Enable = void>
struct dispatchStages {
  static void Dispatch(const int8_t* A, const int8_t* B, llm_kernels::utils::QuantMode quant_option,
                       const float* alpha_col, const float* alpha_row, T* C, int32_t m, int32_t n, int32_t k,
                       llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemm_config, char* workspace,
                       size_t workspace_bytes, cudaStream_t stream, int32_t* occupancy = nullptr) {
    std::string err_msg = "Cutlass int8 gemm. Not instantiates for arch " +
                          std::to_string(arch::kMinComputeCapability) + " with stages set to " + std::to_string(Stages);
    throw std::runtime_error("[dispatchStages::dispatch] " + err_msg);
  }
};

template <typename T, typename arch, typename ThreadblockShape, typename WarpShape>
struct dispatchStages<T, arch, ThreadblockShape, WarpShape, 2> {
  static void Dispatch(const int8_t* A, const int8_t* B, llm_kernels::utils::QuantMode quant_option,
                       const float* alpha_col, const float* alpha_row, T* C, int32_t m, int32_t n, int32_t k,
                       llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemm_config, char* workspace,
                       size_t workspace_bytes, cudaStream_t stream, int32_t* occupancy = nullptr) {
    GenericInt8GemmKernelLauncher<T, arch, ThreadblockShape, WarpShape, 2>(A, B, quant_option, alpha_col, alpha_row, C,
                                                                           m, n, k, gemm_config, workspace,
                                                                           workspace_bytes, stream, occupancy);
  }
};

template <typename T, typename ThreadblockShape, typename WarpShape, int32_t Stages>
struct dispatchStages<T, cutlass::arch::Sm80, ThreadblockShape, WarpShape, Stages,
                      typename std::enable_if<(Stages > 2)>::type> {
  static void Dispatch(const int8_t* A, const int8_t* B, llm_kernels::utils::QuantMode quant_option,
                       const float* alpha_col, const float* alpha_row, T* C, int32_t m, int32_t n, int32_t k,
                       llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemm_config, char* workspace,
                       size_t workspace_bytes, cudaStream_t stream, int32_t* occupancy = nullptr) {
    GenericInt8GemmKernelLauncher<T, cutlass::arch::Sm80, ThreadblockShape, WarpShape, Stages>(
        A, B, quant_option, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace, workspace_bytes, stream,
        occupancy);
  }
};

template <typename T, typename arch, typename ThreadblockShape, typename WarpShape>
void DispatchGemmConfig(const int8_t* A, const int8_t* B, llm_kernels::utils::QuantMode quant_option,
                        const float* alpha_col, const float* alpha_row, T* C, int32_t m, int32_t n, int32_t k,
                        llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemm_config, char* workspace,
                        size_t workspace_bytes, cudaStream_t stream, int32_t* occupancy = nullptr) {
  switch (gemm_config.stages) {
    case 2:
      using DispatcherStages2 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 2>;
      DispatcherStages2::Dispatch(A, B, quant_option, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace,
                                  workspace_bytes, stream, occupancy);
      break;
    case 3:
      using DispatcherStages3 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 3>;
      DispatcherStages3::Dispatch(A, B, quant_option, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace,
                                  workspace_bytes, stream, occupancy);
      break;
    case 4:
      using DispatcherStages4 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 4>;
      DispatcherStages4::Dispatch(A, B, quant_option, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace,
                                  workspace_bytes, stream, occupancy);
      break;
    case 5:
      using DispatcherStages5 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 5>;
      DispatcherStages5::Dispatch(A, B, quant_option, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace,
                                  workspace_bytes, stream, occupancy);
      break;
    case 6:
      using DispatcherStages6 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 6>;
      DispatcherStages6::Dispatch(A, B, quant_option, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace,
                                  workspace_bytes, stream, occupancy);
      break;
    default:
      std::string err_msg = "DispatchGemmConfig does not support stages " + std::to_string(gemm_config.stages);
      throw std::runtime_error("[dispatch_gemm_config] " + err_msg);
      break;
  }
}

template <typename T, typename arch>
void DispatchGemmToCutlass(const int8_t* A, const int8_t* B, llm_kernels::utils::QuantMode quant_option,
                           const float* alpha_col, const float* alpha_row, T* C, int32_t m, int32_t n, int32_t k,
                           char* workspace, size_t workspace_bytes,
                           llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemm_config, cudaStream_t stream,
                           int32_t* occupancy = nullptr) {
  switch (gemm_config.tile_config) {
    case llm_kernels::nvidia::cutlass_extensions::CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64:
      DispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 32, 64>>(
          A, B, quant_option, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace, workspace_bytes, stream,
          occupancy);
      break;
    case llm_kernels::nvidia::cutlass_extensions::CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64:
      DispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<256, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>>(
          A, B, quant_option, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace, workspace_bytes, stream,
          occupancy);
      break;
    case llm_kernels::nvidia::cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      DispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<32, 128, 64>, cutlass::gemm::GemmShape<32, 32, 64>>(
          A, B, quant_option, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace, workspace_bytes, stream,
          occupancy);
      break;
    case llm_kernels::nvidia::cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
      DispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<64, 128, 64>, cutlass::gemm::GemmShape<64, 32, 64>>(
          A, B, quant_option, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace, workspace_bytes, stream,
          occupancy);
      break;
    case llm_kernels::nvidia::cutlass_extensions::CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64:
      DispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<32, 64, 64>>(
          A, B, quant_option, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace, workspace_bytes, stream,
          occupancy);
      break;
    case llm_kernels::nvidia::cutlass_extensions::CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
      DispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>>(
          A, B, quant_option, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace, workspace_bytes, stream,
          occupancy);
      break;
    case llm_kernels::nvidia::cutlass_extensions::CutlassTileConfig::Undefined:
      throw std::runtime_error("[int8][dispatch_gemm_to_cutlass] gemm config undefined.");
      break;
    case llm_kernels::nvidia::cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
      throw std::runtime_error(
          "[int8][dispatch_gemm_to_cutlass] gemm config should have already been set by "
          "heuristic.");
      break;
    default:
      throw std::runtime_error("[int8][dispatch_gemm_to_cutlass] Config is invalid for int8 GEMM.");
      break;
  }
}

template <typename T>
CutlassInt8GemmRunner<T>::CutlassInt8GemmRunner() {
  int32_t device{-1};
  CHECK_NVIDIA_CUDA_ERROR(cudaGetDevice(&device));
  sm_ = llm_kernels::utils::GetSMVersion();
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&multiprocessor_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename T>
CutlassInt8GemmRunner<T>::~CutlassInt8GemmRunner() {}

template <typename T>
void CutlassInt8GemmRunner<T>::DispatchToArch(const int8_t* A, const int8_t* B,
                                              llm_kernels::utils::QuantMode quant_option, const float* alpha_col,
                                              const float* alpha_row, T* C, int32_t m, int32_t n, int32_t k,
                                              llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemm_config,
                                              char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream,
                                              int32_t* occupancy) {
  if (sm_ >= NVIDIA_VOLTA_GPU_COMPUTE_CAPABILITY && sm_ < NVIDIA_AGX_XAVIER_GPU_COMPUTE_CAPABILITY) {
    DispatchGemmToCutlass<T, cutlass::arch::Sm70>(A, B, quant_option, alpha_col, alpha_row, C, m, n, k, workspace_ptr,
                                                  workspace_bytes, gemm_config, stream, occupancy);
  } else if (sm_ >= NVIDIA_AGX_XAVIER_GPU_COMPUTE_CAPABILITY && sm_ < NVIDIA_TURING_GPU_COMPUTE_CAPABILITY) {
    DispatchGemmToCutlass<T, cutlass::arch::Sm72>(A, B, quant_option, alpha_col, alpha_row, C, m, n, k, workspace_ptr,
                                                  workspace_bytes, gemm_config, stream, occupancy);
  } else if (sm_ >= NVIDIA_TURING_GPU_COMPUTE_CAPABILITY && sm_ < NVIDIA_AMPERE_GPU_COMPUTE_CAPABILITY) {
    DispatchGemmToCutlass<T, cutlass::arch::Sm75>(A, B, quant_option, alpha_col, alpha_row, C, m, n, k, workspace_ptr,
                                                  workspace_bytes, gemm_config, stream, occupancy);
  } else if (sm_ >= NVIDIA_AMPERE_GPU_COMPUTE_CAPABILITY && sm_ <= NVIDIA_HOPPER_GPU_COMPUTE_CAPABILITY) {
    DispatchGemmToCutlass<T, cutlass::arch::Sm80>(A, B, quant_option, alpha_col, alpha_row, C, m, n, k, workspace_ptr,
                                                  workspace_bytes, gemm_config, stream, occupancy);
  } else {
    throw std::runtime_error("[CutlassInt8GemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS int8 GEMM");
  }
}

template <typename T>
void CutlassInt8GemmRunner<T>::Gemm(const int8_t* A, const int8_t* B, llm_kernels::utils::QuantMode quant_option,
                                    const float* alpha_col, const float* alpha_row, void* C, int32_t m, int32_t n,
                                    int32_t k, llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemm_config,
                                    char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream) {
  DispatchToArch(A, B, quant_option, alpha_col, alpha_row, reinterpret_cast<T*>(C), m, n, k, gemm_config, workspace_ptr,
                 workspace_bytes, stream);
}

template <typename T>
std::vector<llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig> CutlassInt8GemmRunner<T>::GetConfigs() const {
  static constexpr bool is_weight_only = false;
  std::vector<llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig> candidate_configs =
      GetCandidateConfigs(sm_, is_weight_only, sm_ <= NVIDIA_VOLTA_GPU_COMPUTE_CAPABILITY, /* SIMT configs */
                          true, SPLIT_K_LIMIT);                                            /* INT8 configs */
  return candidate_configs;
}

template <typename T>
size_t CutlassInt8GemmRunner<T>::GetWorkspaceSize(const int32_t m, const int32_t n, const int32_t k) {
  // These are the min tile sizes for each config, which would launch the maximum number of blocks
  const int32_t max_grid_m = cutlass::ceil_div(m, MIN_M_TILE);
  const int32_t max_grid_n = cutlass::ceil_div(m, MIN_N_TILE);
  // We need 4 bytes per block in the worst case. We launch SPLIT_K_LIMIT in z dim.
  constexpr int32_t FPA_INTB_BLOCK_BYTE_NUM = 4;
  return static_cast<size_t>(max_grid_m * max_grid_n * SPLIT_K_LIMIT * FPA_INTB_BLOCK_BYTE_NUM);
}

}  // namespace nvidia
}  // namespace llm_kernels
