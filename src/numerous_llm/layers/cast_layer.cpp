/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/cast_layer.h"

#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"

namespace numerous_llm {

Status CastLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  HalfToFloat(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()), input_tensors[0].GetElementNumber(),
              output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_]);
  output_tensors[0].shape = input_tensors[0].shape;
  return Status();
}
}  // namespace numerous_llm
