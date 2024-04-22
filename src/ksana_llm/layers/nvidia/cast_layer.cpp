/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/cast_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status CastLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DataToFloat<T>(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()), input_tensors[0].GetElementNumber(),
                 output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_].Get());
  output_tensors[0].shape = input_tensors[0].shape;
  return Status();
}

template class CastLayer<float>;
template class CastLayer<half>;
#ifdef ENABLE_BFLOAT16
template class CastLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
