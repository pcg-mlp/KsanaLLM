/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/cast_layer.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

#ifdef ENABLE_ACL
#  include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#endif

namespace ksana_llm {

Status CastLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
#ifdef ENABLE_CUDA
  HalfToFloat(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()), input_tensors[0].GetElementNumber(),
              output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_].Get());
#endif
  output_tensors[0].shape = input_tensors[0].shape;
  return Status();
}
}  // namespace ksana_llm
