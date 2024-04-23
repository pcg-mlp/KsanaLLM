/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status PagedAttentionLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status();
}
template class PagedAttentionLayer<float>;
template class PagedAttentionLayer<float16>;

}  // namespace ksana_llm
