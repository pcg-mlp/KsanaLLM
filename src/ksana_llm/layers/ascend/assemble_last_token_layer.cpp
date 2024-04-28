/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/assemble_last_token_layer.h"

#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status AssembleLastTokenLayer<T>::Forward(const std::vector<Tensor>& input_tensors,
                                          std::vector<Tensor>& output_tensors) {
  size_t batch_size = input_tensors[0].shape[0];
  size_t seq_len = input_tensors[0].shape[1];
  size_t hidden_size = input_tensors[0].shape[2];

  void* input_ptr = input_tensors[0].GetPtr<void>();
  void* output_ptr = output_tensors[0].GetPtr<void>();

  for (size_t i = 0; i < batch_size; ++i) {
    size_t batch_offset = i * seq_len * hidden_size * sizeof(T);

    size_t offset = batch_offset  + (seq_len - 1) * hidden_size * sizeof(T);
    Memcpy(output_ptr + (i * hidden_size * sizeof(T)), input_ptr + offset, hidden_size * sizeof(T),
           MEMCPY_DEVICE_TO_DEVICE);
  }

  output_tensors[0].shape = {batch_size, 1, hidden_size};

  return Status();
}
template class AssembleLastTokenLayer<float>;
template class AssembleLastTokenLayer<float16>;

}  // namespace ksana_llm
