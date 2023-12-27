/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/add_layer.h"

#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"

namespace numerous_llm {

Status AddLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  InvokeAddBiasResidual(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                        reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>()),
                        static_cast<int>(input_tensors[0].shape[0]), static_cast<int>(input_tensors[0].shape[1]),
                        output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_]);
  return Status();
}
}  // namespace numerous_llm
