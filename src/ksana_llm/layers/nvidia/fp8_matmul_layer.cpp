/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_FP8
#  include "ksana_llm/layers/fp8_matmul_layer.h"

#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status Fp8MatMulLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  STATUS_CHECK_FAILURE(BaseLayer::Init(parameters, context, rank));
  int parameter_index = 0;
  max_m_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_k_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  return Status();
}

template <typename T>
size_t Fp8MatMulLayer<T>::GetWorkSpaceSize(const int m, const int k) {
  size_t input_size = m * k * GetTypeSize(TYPE_FP8_E4M3);
  size_t scale_size = GetTypeSize(TYPE_FP32);
  size_t workspace_size = input_size + scale_size;
  return workspace_size;
}

template <typename T>
size_t Fp8MatMulLayer<T>::GetWorkSpaceSize() {
  return GetWorkSpaceSize(max_m_, max_k_);
}

template <typename T>
Status Fp8MatMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int m = input_tensors[0].shape[0];
  int k = input_tensors[0].shape[1];
  int n = input_tensors[1].shape[0];
  const T* input = static_cast<const T*>(input_tensors[0].GetPtr<void>());
  size_t workspace_size = GetWorkSpaceSize(m, k);
  if (workspace_size > workspace_buffer_->GetTotalBytes()) {
    KLLM_LOG_ERROR << fmt::format("workspace size {} > buffer size {}", workspace_size,
                                  workspace_buffer_->GetTotalBytes());
    throw std::runtime_error(
        fmt::format("workspace size {} > buffer size {}", workspace_size, workspace_buffer_->GetTotalBytes()));
  }
  void* input_quant = workspace_buffer_->GetPtr<void>();
  float* input_scale = static_cast<float*>(input_quant + GetTypeSize(TYPE_FP8_E4M3) * m * k);
  const void* weight_quant = input_tensors[1].GetPtr<const void>();
  const void* weight_scale = input_tensors[1].scales->GetPtr<const void>();
  if (weight_scale == nullptr) {
    KLLM_LOG_ERROR << "weight_scale is nullptr.";
    throw std::runtime_error(fmt::format("weight_scale is nullptr."));
  }
  T* output = static_cast<T*>(output_tensors[0].GetPtr<void>());
  output_tensors[0].shape = {static_cast<size_t>(m), static_cast<size_t>(n)};
  output_tensors[0].dtype = input_tensors[0].dtype;
  Fp8DynamicQuantize<T>(1, m * k, input, input_quant, input_scale, context_->GetComputeStreams()[rank_].Get());
  Fp8QuantizedMatMul<T>(context_->ext->GetCublasHandles()[rank_], context_->ext->GetCublasLtHandles()[rank_], m, n, k,
                        input_quant, input_scale, weight_quant, weight_scale, output,
                        context_->GetComputeStreams()[rank_].Get());
  return Status();
}

template class Fp8MatMulLayer<float>;
template class Fp8MatMulLayer<half>;
#  ifdef ENABLE_BFLOAT16
template class Fp8MatMulLayer<__nv_bfloat16>;
#  endif

}  // namespace ksana_llm
#endif
