/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/layernorm_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status LayernormLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  NLLM_LOG_DEBUG << fmt::format("rms_norm_eps {}", rms_norm_eps_);
  return Status();
}

template <typename T>
Status LayernormLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  InvokeLayerNorm<T>(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), rms_norm_eps_,
                     input_tensors[0].shape[0], input_tensors[0].shape[1], output_tensors[0].GetPtr<void>(),
                     context_->GetComputeStreams()[rank_].Get());
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template class LayernormLayer<float>;
template class LayernormLayer<half>;
#ifdef ENABLE_BFLOAT16
template class LayernormLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
