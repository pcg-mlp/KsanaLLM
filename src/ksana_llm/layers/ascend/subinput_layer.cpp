/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/subinput_layer.h"

namespace ksana_llm {

template <typename T>
Status SubinputLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "SubinputLayer not supported.");
}
template class SubinputLayer<float>;
template class SubinputLayer<float16>;
}  // namespace ksana_llm
