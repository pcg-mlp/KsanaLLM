/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/silu_mul_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status SiluMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  InvokeSiluActivation<T>(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                          reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>()),
                          static_cast<int>(input_tensors[0].shape[0]), static_cast<int>(input_tensors[0].shape[1]),
                          output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_].Get());
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template class SiluMulLayer<float>;
template class SiluMulLayer<half>;
#ifdef ENABLE_BFLOAT16
template class SiluMulLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
