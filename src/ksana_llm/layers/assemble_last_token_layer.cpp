/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/assemble_last_token_layer.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

#ifdef ENABLE_ACL
#  include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#endif

namespace ksana_llm {

Status AssembleLastTokenLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int batch_size = input_tensors[1].shape[0] - 1;
#ifdef ENABLE_CUDA
  AssembleLastToken(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                    reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>()), batch_size,
                    input_tensors[0].shape[1], reinterpret_cast<void*>(output_tensors[0].GetPtr<void>()),
                    context_->GetComputeStreams()[rank_].GetStreamIns());
#endif
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].shape[0] = batch_size;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm
