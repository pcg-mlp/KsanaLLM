/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/assemble_last_token_layer.h"

#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"

namespace numerous_llm {

Status AssembleLastTokenLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int bs = 1;

  AssembleLastToken(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                    reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>()), bs, input_tensors[0].shape[1],
                    reinterpret_cast<void*>(output_tensors[0].GetPtr<void>()), context_->GetComputeStreams()[rank_]);

  return Status();
}

}  // namespace numerous_llm