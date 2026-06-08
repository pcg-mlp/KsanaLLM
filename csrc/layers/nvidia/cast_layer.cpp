/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/cast_layer.h"

#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename SRC_DTYPE>
Status CastLayer::ForwardT(const Tensor* input, size_t byte_offset, size_t vocab_size, size_t vocab_size_pad,
                           Tensor* output) {
  void* output_ptr = output->GetPtr<void>();
  output_ptr += byte_offset;
  DataToFloat<SRC_DTYPE>(reinterpret_cast<const void*>(input->GetPtr<void>()), input->GetElementNumber(), vocab_size,
                         vocab_size_pad, output_ptr, context_->device->GetPrimaryComputeStream().Get());
  output->dtype = DataType::TYPE_FP32;
  output->shape = input->shape;
  return Status();
}

Status CastLayer::Forward(const Tensor* input, size_t byte_offset, size_t vocab_size, size_t vocab_size_pad,
                          Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input, byte_offset, vocab_size, vocab_size_pad, output);
}

}  // namespace ksana_llm
