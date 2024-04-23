/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/activation_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status ActivationLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status();
}

template class ActivationLayer<float>;
template class ActivationLayer<float16>;
#ifdef ENABLE_BFLOAT16
template class ActivationLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
