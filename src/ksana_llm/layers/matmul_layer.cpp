/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/matmul_layer.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

#ifdef ENABLE_ACL
#  include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#endif

namespace ksana_llm {

Status MatMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
#ifdef ENABLE_CUDA
  InvokeMatMul(context_->GetCublasHandles()[rank_], context_->GetCublasLtHandles()[rank_],
               static_cast<int>(input_tensors[0].shape[0]), static_cast<int>(input_tensors[1].shape[1]),
               static_cast<int>(input_tensors[0].shape[1]),
               reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
               reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>()), output_tensors[0].GetPtr<void>(),
               context_->GetComputeStreams()[rank_]);
#endif
  output_tensors[0].shape = {static_cast<int>(input_tensors[0].shape[0]), static_cast<int>(input_tensors[1].shape[1])};
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}
}  // namespace ksana_llm
