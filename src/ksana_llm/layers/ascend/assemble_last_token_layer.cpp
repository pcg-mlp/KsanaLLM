/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/assemble_last_token_layer.h"

#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status AssembleLastTokenLayer<T>::Forward(const std::vector<Tensor>& input_tensors,
                                          std::vector<Tensor>& output_tensors) {
  return Status();
}
template class AssembleLastTokenLayer<float>;
template class AssembleLastTokenLayer<float16>;

}  // namespace ksana_llm
