/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/cast_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename SRC_DTYPE>
Status CastLayer<SRC_DTYPE>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  void* output_ptr = output_tensors[0].GetPtr<void>();
  if (input_tensors.size() > 1) {
    // When the number of input_tensors is greater than 1, perform a cast operation with an offset.
    // Set output_offset to the value of the first dimension of input_tensors[1].
    size_t output_offset = input_tensors[1].shape[0];
    output_ptr += output_offset;
  }
  DataToFloat<SRC_DTYPE>(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                         input_tensors[0].GetElementNumber(), output_ptr, context_->GetComputeStreams()[rank_].Get());
  if (input_tensors.size() == 1) {
    output_tensors[0].shape = input_tensors[0].shape;
  }
  output_tensors[0].dtype = DataType::TYPE_FP32;
  return Status();
}

template class CastLayer<float>;
template class CastLayer<half>;
#ifdef ENABLE_BFLOAT16
template class CastLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
