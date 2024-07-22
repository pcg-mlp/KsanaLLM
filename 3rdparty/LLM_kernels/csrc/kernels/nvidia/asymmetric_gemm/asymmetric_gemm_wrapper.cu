/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "csrc/kernels/nvidia/asymmetric_gemm/asymmetric_gemm_wrapper.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/fpA_intB_gemm/fpA_intB_gemm_template.h"

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
void FpAIntBGroupGemmWrapper<T, WT>::GetWorkspaceSize(size_t m, size_t n, size_t k, size_t& ws_bytes) {
  if constexpr (!std::is_same_v<T, float>) {
    using weight_type = typename WeightTypeSelector<WT>::type;
    auto gemm = std::make_shared<llm_kernels::nvidia::CutlassFpAIntBGemmRunner<
        T, weight_type, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
    ws_bytes = gemm->GetWorkspaceSize(m, n, k);
  }
}

template <typename T, WeightType WT>
void FpAIntBGroupGemmWrapper<T, WT>::Gemm(void* output, const void* input, const void* weight, const void* scales,
                                         void* ws, size_t m, size_t n, size_t k, size_t groupsize, size_t config_index,
                                         cudaStream_t stream) {
  if constexpr (!std::is_same_v<T, float>) {
    using weight_type = typename WeightTypeSelector<WT>::type;
    auto gemm = std::make_shared<llm_kernels::nvidia::CutlassFpAIntBGemmRunner<
        T, weight_type, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
    const size_t ws_bytes = gemm->GetWorkspaceSize(m, n, k);
    gemm->Gemm(reinterpret_cast<const T*>(input), reinterpret_cast<const weight_type*>(weight),
               reinterpret_cast<const T*>(scales),
               nullptr,  // no zeros
               nullptr,  // no bias
               reinterpret_cast<T*>(output), m, n, k, groupsize, gemm->GetConfigs()[config_index],
               reinterpret_cast<char*>(ws), ws_bytes, stream);
  }
}

template <typename T, WeightType WT>
size_t FpAIntBGroupGemmWrapper<T, WT>::GetBestConfigIndex(size_t warmup, size_t iter, void* output, const void* input,
                                                         const void* weight, const void* scales, void* ws, size_t m,
                                                         size_t n, size_t k, size_t groupsize, cudaStream_t stream) {
  if constexpr (!std::is_same_v<T, float>) {
    using weight_type = typename WeightTypeSelector<WT>::type;
    auto gemm = std::make_shared<llm_kernels::nvidia::CutlassFpAIntBGemmRunner<
        T, weight_type, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    float fast_time = std::numeric_limits<float>::max();
    auto configs = gemm->GetConfigs();
    int best_config_index = 0;

    for (size_t config_index = 0; config_index < configs.size(); config_index++) {
      auto& config = configs[config_index];

      // fiter out invalid config
      if (!IsConfigValid(config, k)) {
        continue;
      }

      // warm up
      for (size_t i = 0; i < warmup; ++i) {
        Gemm(output, input, weight, scales, ws, m, n, k, groupsize, config_index, stream);
      }

      // record time
      cudaEventRecord(begin, stream);
      for (size_t i = 0; i < iter; ++i) {
        Gemm(output, input, weight, scales, ws, m, n, k, groupsize, config_index, stream);
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

    return best_config_index;
  } else {
    return 0;
  }
}

template class FpAIntBGroupGemmWrapper<float, INT4>;
template class FpAIntBGroupGemmWrapper<float, INT8>;

template class FpAIntBGroupGemmWrapper<half, INT4>;
template class FpAIntBGroupGemmWrapper<half, INT8>;

#ifdef ENABLE_BF16
template class FpAIntBGroupGemmWrapper<__nv_bfloat16, INT4>;
template class FpAIntBGroupGemmWrapper<__nv_bfloat16, INT8>;
#endif

}  // namespace nvidia
}  // namespace llm_kernels