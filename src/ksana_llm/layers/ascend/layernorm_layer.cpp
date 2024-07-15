/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/layernorm_layer.h"

#include <cstdint>

#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/kernels/ascend/rmsnorm/rmsnorm.h"
#include "csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

template <typename T>
Status LayernormLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  int parameter_index = 0;
  context_ = context;
  rank_ = rank;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  return Status();
}

template <typename T>
Status LayernormLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int64_t total_seq_len = input_tensors[0].shape[0];
  int64_t hidden_size = input_tensors[0].shape[1];
  void* lm_input_tensor_buf_ptr = input_tensors[0].GetPtr<void>();
  void* lm_weight_tensor_buf_ptr = input_tensors[1].GetPtr<void>();
  void* lm_output_tensor_buf_ptr = output_tensors[0].GetPtr<void>();
  // TODO(karlluo): support beta
  T* beta_ptr = nullptr;
  llm_kernels::ascend::InvokeRmsLayerNorm<T>(
      reinterpret_cast<T*>(lm_output_tensor_buf_ptr), reinterpret_cast<T*>(lm_input_tensor_buf_ptr),
      reinterpret_cast<T*>(lm_weight_tensor_buf_ptr), beta_ptr, rms_norm_eps_, total_seq_len, hidden_size,
      context_->GetComputeStreams()[rank_].Get(), GetWorkSpaceFunc());

  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}
template class LayernormLayer<float>;
template class LayernormLayer<float16>;
}  // namespace ksana_llm
