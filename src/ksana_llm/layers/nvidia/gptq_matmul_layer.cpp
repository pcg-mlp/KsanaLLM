/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/gptq_matmul_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T, DataType WT>
Status GPTQMatMulLayer<T, WT>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context,
                                    int rank) {
  context_ = context;
  rank_ = rank;

  int parameter_index = 0;
  max_m = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_n = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  max_k = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  groupsize = std::any_cast<const size_t>(parameters[parameter_index++]);

  return Status();
}

template <typename T, DataType WT>
size_t GPTQMatMulLayer<T, WT>::GetWorkSpaceSize() {
  size_t max_ws_bytes;
  if constexpr (WT == TYPE_I4_G128) {
    GetFpAIntBGPTQGemmWorkspaceSize<T, llm_kernels::nvidia::WeightType::INT4>(max_m, max_n, max_k, max_ws_bytes);
  } else {
    throw std::runtime_error("Not supported weight data type.");
  }

  return max_ws_bytes;
}

template <typename T, DataType WT>
Status GPTQMatMulLayer<T, WT>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if constexpr (WT == TYPE_I4_G128) {
    const Tensor& weight_tensor = input_tensors[1];
    const void* p_qweight_tensor = weight_tensor.GetPtr<void>();
    const void* p_scales_tensor = weight_tensor.scales->GetPtr<void>();

    size_t m = input_tensors[0].shape[0];
    size_t n = weight_tensor.scales->shape[1];
    size_t k = input_tensors[0].shape[1];

    InvokeFpAIntBGPTQGemm<T, llm_kernels::nvidia::WeightType::INT4>(
        output_tensors[0].GetPtr<void>(), input_tensors[0].GetPtr<void>(), p_qweight_tensor, p_scales_tensor,
        workspace_buffer_.GetPtr<void>(), m, n, k, groupsize, context_->GetComputeStreams()[rank_].Get());
    output_tensors[0].shape = {m, n};
    output_tensors[0].dtype = input_tensors[0].dtype;
    return Status();
  } else {
    return Status(RET_RUNTIME, fmt::format("Not supported weight data type."));
  }
}

template class GPTQMatMulLayer<float, TYPE_I4_G128>;
template class GPTQMatMulLayer<half, TYPE_I4_G128>;
#ifdef ENABLE_BFLOAT16
template class GPTQMatMulLayer<__nv_bfloat16, TYPE_I4_G128>;
#endif

}  // namespace ksana_llm
