/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/assemble_last_token_layer.h"
#include <cstdlib>

#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

template <typename T>
Status AssembleLastTokenLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context,
                                       int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "AssembleLastTokenLayer not supported.");
}

template <typename T>
Status AssembleLastTokenLayer<T>::Forward(const std::vector<Tensor>& input_tensors,
                                          std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "AssembleLastTokenLayer not supported.");
}
template class AssembleLastTokenLayer<float>;
template class AssembleLastTokenLayer<float16>;
#ifdef ENABLE_BFLOAT16
template class AssembleLastTokenLayer<bfloat16>;
#endif

}  // namespace ksana_llm
