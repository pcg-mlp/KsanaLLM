/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "csrc/kernels/nvidia/asymmetric_gemm/asymmetric_gemm_wrapper.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace llm_kernels {
namespace nvidia {

template <typename T>
std::vector<tkc::CutlassGemmConfig> GetConfigs(T& runner, int k) {
  auto configs = runner.GetConfigs();
  std::vector<tkc::CutlassGemmConfig> rets;
  for (auto config : configs) {
    if (config.stages >= 5) {
      continue;
    }
    if (config.split_k_style != tkc::SplitKStyle::NO_SPLIT_K) {
      int k_size = (k + config.split_k_factor - 1) / config.split_k_factor;
      if (k_size % 64) {
        continue;
      }
    }
    rets.push_back(config);
  }
  return rets;
}

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

template <typename T, WeightType WT>
void FpAIntBGPTQGemmWrapper<T, WT>::GetWorkspaceSize(size_t m, size_t n, size_t k, size_t& ws_bytes) {
  if constexpr (std::is_same_v<T, float>) {
    throw std::runtime_error("Not supported activation data type == float.");
  } else {
    using weight_type = typename WeightTypeSelector<WT>::type;

    auto runner = std::make_shared<llm_kernels::nvidia::CutlassFpAIntBGemmRunner<
        T, weight_type, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
    auto& gemm = *runner;

    ws_bytes = gemm.GetWorkspaceSize(m, n, k);
  }
}

template <typename T, WeightType WT>
void FpAIntBGPTQGemmWrapper<T, WT>::Gemm(void* output, const void* input, const void* weight, const void* scales,
                                         void* ws, size_t m, size_t n, size_t k, size_t groupsize,
                                         cudaStream_t stream) {
  if constexpr (std::is_same_v<T, float>) {
    throw std::runtime_error("Not supported activation data type == float.");
  } else {
    using weight_type = typename WeightTypeSelector<WT>::type;

    auto runner = std::make_shared<llm_kernels::nvidia::CutlassFpAIntBGemmRunner<
        T, weight_type, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
    auto& gemm = *runner;

    size_t ws_bytes = gemm.GetWorkspaceSize(m, n, k);

    auto configs = GetConfigs(gemm, k);
    auto best_config = configs[0];

    gemm.Gemm(reinterpret_cast<const T*>(input), reinterpret_cast<const weight_type*>(weight),
              reinterpret_cast<const T*>(scales),
              nullptr,  // no zeros
              nullptr,  // no bias
              reinterpret_cast<T*>(output), m, n, k, groupsize, best_config, reinterpret_cast<char*>(ws), ws_bytes,
              stream);
  }
}

template class FpAIntBGPTQGemmWrapper<float, INT4>;
template class FpAIntBGPTQGemmWrapper<float, INT8>;

template class FpAIntBGPTQGemmWrapper<half, INT4>;
template class FpAIntBGPTQGemmWrapper<half, INT8>;

#ifdef ENABLE_BF16
template class FpAIntBGPTQGemmWrapper<__nv_bfloat16, INT4>;
template class FpAIntBGPTQGemmWrapper<__nv_bfloat16, INT8>;
#endif

}  // namespace nvidia
}  // namespace llm_kernels