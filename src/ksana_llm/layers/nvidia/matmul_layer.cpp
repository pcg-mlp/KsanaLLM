/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/matmul_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status MatMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  InvokeMatMul<T>(context_->ext->GetCublasHandles()[rank_], context_->ext->GetCublasLtHandles()[rank_],
               static_cast<int>(input_tensors[0].shape[0]), static_cast<int>(input_tensors[1].shape[1]),
               static_cast<int>(input_tensors[0].shape[1]),
               reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
               reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>()), output_tensors[0].GetPtr<void>(),
               context_->GetComputeStreams()[rank_].Get());
  output_tensors[0].shape = {static_cast<int>(input_tensors[0].shape[0]), static_cast<int>(input_tensors[1].shape[1])};
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}

template class MatMulLayer<float>;
template class MatMulLayer<half>;
#ifdef ENABLE_BFLOAT16
template class MatMulLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
