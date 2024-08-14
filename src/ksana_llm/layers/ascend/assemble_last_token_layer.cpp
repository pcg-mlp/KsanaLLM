/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/assemble_last_token_layer.h"
#include <cstdlib>

#include "3rdparty/LLM_kernels/csrc/kernels/ascend/assemble_last_token/assemble_last_token.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

template <typename T>
Status AssembleLastTokenLayer<T>::Forward(const std::vector<Tensor>& input_tensors,
                                          std::vector<Tensor>& output_tensors) {
  size_t batch_size = input_tensors[1].shape[0] - 1;
  size_t hidden_size = input_tensors[0].shape[1];

  llm_kernels::ascend::InvokeAssembleLastToken<T>(input_tensors[0].GetPtr<T>(), input_tensors[1].GetPtr<size_t>(),
                                                  nullptr, batch_size, hidden_size, output_tensors[0].GetPtr<T>(),
                                                  context_->GetComputeStreams()[rank_].Get(), GetWorkSpaceFunc());
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].shape[0] = batch_size;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}
template class AssembleLastTokenLayer<float>;
template class AssembleLastTokenLayer<float16>;

}  // namespace ksana_llm
