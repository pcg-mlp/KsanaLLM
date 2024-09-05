/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "csrc/kernels/nvidia/asymmetric_gemm/asymmetric_gemm_wrapper.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/fpA_intB_gemm/fpA_intB_gemm_template.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/weightOnlyBatchedGemv/kernelLauncher.h"

namespace llm_kernels {
namespace nvidia {

template <WeightType WT>
struct WeightTypeSelector;

template <>
struct WeightTypeSelector<INT4> {
  using type = cutlass::uint4b_t;
};

template <>
struct WeightTypeSelector<INT8> {
  using type = uint8_t;
};

// The config consists of “tile_config, SplitKStyle, split_k_factor, and stages”.
// For details, refer to cutlass_heuristic.cc
// tile_config: determined by sm and quantization configuration, and k_tile is always 64 among all tile sizes
// SplitKStyle: only supports SPLIT_K_SERIAL
// split_k_factor: minimum is 2, maximum is 7
// stages: in Group and AMPERE GPU, minimum value is 2, maximum value is 4
bool IsConfigValid(tkc::CutlassGemmConfig& config, size_t k) {
  if (config.stages >= 5) {
    return false;
  }
  if (config.split_k_style != tkc::SplitKStyle::NO_SPLIT_K) {
    int k_size = (k + config.split_k_factor - 1) / config.split_k_factor;
    if (k_size % 64) {
      return false;
    }
  }
  return true;
}

template <typename T, WeightType WT>
void FpAIntBGroupCutlassGemmWrapper<T, WT>::GetWorkspaceSize(size_t m, size_t n, size_t k, size_t& ws_bytes) {
  if constexpr (!std::is_same_v<T, float>) {
    using weight_type = typename WeightTypeSelector<WT>::type;
    auto gemm = std::make_shared<llm_kernels::nvidia::CutlassFpAIntBGemmRunner<
        T, weight_type, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
    ws_bytes = gemm->GetWorkspaceSize(m, n, k);
  }
}

template <typename T, WeightType WT>
void FpAIntBGroupCutlassGemmWrapper<T, WT>::Gemm(void* output, const void* input, const void* weight,
                                                 const void* scales, const void* zeros, void* ws, size_t m, size_t n,
                                                 size_t k, size_t groupsize, size_t config_index, cudaStream_t stream) {
  if constexpr (!std::is_same_v<T, float>) {
    using weight_type = typename WeightTypeSelector<WT>::type;
    if (zeros == nullptr) {
      auto gemm = std::make_shared<llm_kernels::nvidia::CutlassFpAIntBGemmRunner<
          T, weight_type, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
      gemm->Gemm(reinterpret_cast<const T*>(input), reinterpret_cast<const weight_type*>(weight),
                reinterpret_cast<const T*>(scales),
                nullptr,  // no zeros
                nullptr,  // no bias
                reinterpret_cast<T*>(output), m, n, k, groupsize, gemm->GetConfigs()[config_index],
                reinterpret_cast<char*>(ws), gemm->GetWorkspaceSize(m, n, k), stream);
    } else {
      auto gemm = std::make_shared<llm_kernels::nvidia::CutlassFpAIntBGemmRunner<
          T, weight_type, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
      gemm->Gemm(reinterpret_cast<const T*>(input), reinterpret_cast<const weight_type*>(weight),
                reinterpret_cast<const T*>(scales), reinterpret_cast<const T*>(zeros),
                nullptr,  // no bias
                reinterpret_cast<T*>(output), m, n, k, groupsize, gemm->GetConfigs()[config_index],
                reinterpret_cast<char*>(ws), gemm->GetWorkspaceSize(m, n, k), stream);
    }
  }
}

template <typename T, WeightType WT>
size_t FpAIntBGroupCutlassGemmWrapper<T, WT>::GetBestConfigIndex(size_t warmup, size_t iter, void* output,
                                                                 const void* input, const void* weight,
                                                                 const void* scales, const void* zeros, void* ws,
                                                                 size_t m, size_t n, size_t k, size_t groupsize,
                                                                 cudaStream_t stream) {
  if constexpr (!std::is_same_v<T, float>) {
    using weight_type = typename WeightTypeSelector<WT>::type;

    std::vector<cutlass_extensions::CutlassGemmConfig> configs;
    if (zeros == nullptr) {
      auto gemm = std::make_shared<llm_kernels::nvidia::CutlassFpAIntBGemmRunner<
          T, weight_type, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
      configs = gemm->GetConfigs();
    } else {
      auto gemm = std::make_shared<llm_kernels::nvidia::CutlassFpAIntBGemmRunner<
          T, weight_type, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
      configs = gemm->GetConfigs();
    }

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    float fast_time = std::numeric_limits<float>::max();
    int best_config_index = 0;

    for (size_t config_index = 0; config_index < configs.size(); config_index++) {
      auto& config = configs[config_index];

      // fiter out invalid config
      if (!IsConfigValid(config, k)) {
        continue;
      }

      // warm up
      for (size_t i = 0; i < warmup; ++i) {
        Gemm(output, input, weight, scales, zeros, ws, m, n, k, groupsize, config_index, stream);
      }

      // record time
      cudaEventRecord(begin, stream);
      for (size_t i = 0; i < iter; ++i) {
        Gemm(output, input, weight, scales, zeros, ws, m, n, k, groupsize, config_index, stream);
      }
      cudaEventRecord(end, stream);
      cudaEventSynchronize(end);
      float time;
      cudaEventElapsedTime(&time, begin, end);

      if (time < fast_time) {
        fast_time = time;
        best_config_index = config_index;
      }
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);

    return best_config_index;
  } else {
    return 0;
  }
}

template class FpAIntBGroupCutlassGemmWrapper<float, INT4>;
template class FpAIntBGroupCutlassGemmWrapper<float, INT8>;

template class FpAIntBGroupCutlassGemmWrapper<half, INT4>;
template class FpAIntBGroupCutlassGemmWrapper<half, INT8>;

#ifdef ENABLE_BF16
template class FpAIntBGroupCutlassGemmWrapper<__nv_bfloat16, INT4>;
template class FpAIntBGroupCutlassGemmWrapper<__nv_bfloat16, INT8>;
#endif

template <typename T, WeightType WT>
FpAIntBGroupCudaGemmWrapper<T, WT>::FpAIntBGroupCudaGemmWrapper() {
  arch = llm_kernels::utils::GetSMVersion();
}

template <typename T, WeightType WT>
bool FpAIntBGroupCudaGemmWrapper<T, WT>::IsSupport() {
  if constexpr (WT == WeightType::INT4 && std::is_same_v<T, half>) {
    return llm_kernels::nvidia::weight_only::is_supported(
        arch, llm_kernels::nvidia::weight_only::KernelType::FP16Int4Groupwise);
  } else {
    return false;
  }
}

template <typename T, WeightType WT>
void FpAIntBGroupCudaGemmWrapper<T, WT>::Gemm(void* output, const void* input, const void* weight, const void* scales,
                                              const void* zeros, size_t m, size_t n, size_t k, size_t groupsize,
                                              cudaStream_t stream) {
  if constexpr (WT == WeightType::INT4 && std::is_same_v<T, half>) {
    llm_kernels::nvidia::weight_only::Params params{reinterpret_cast<const void*>(input),
                                                    nullptr,
                                                    reinterpret_cast<const void*>(weight),
                                                    reinterpret_cast<const void*>(scales),
                                                    reinterpret_cast<const void*>(zeros),
                                                    nullptr,  // no bias
                                                    reinterpret_cast<void*>(output),
                                                    1.0f,
                                                    static_cast<int>(m),
                                                    static_cast<int>(n),
                                                    static_cast<int>(k),
                                                    static_cast<int>(groupsize),
                                                    weight_only::KernelType::FP16Int4Groupwise};
    llm_kernels::nvidia::weight_only::kernel_launcher(arch, params, stream);
  }
}

template class FpAIntBGroupCudaGemmWrapper<float, INT4>;
template class FpAIntBGroupCudaGemmWrapper<float, INT8>;

template class FpAIntBGroupCudaGemmWrapper<half, INT4>;
template class FpAIntBGroupCudaGemmWrapper<half, INT8>;

#ifdef ENABLE_BF16
template class FpAIntBGroupCudaGemmWrapper<__nv_bfloat16, INT4>;
template class FpAIntBGroupCudaGemmWrapper<__nv_bfloat16, INT8>;
#endif

}  // namespace nvidia
}  // namespace llm_kernels