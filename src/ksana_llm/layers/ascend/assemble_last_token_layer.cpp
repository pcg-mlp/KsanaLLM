/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/assemble_last_token_layer.h"
#include <cstdlib>

#include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

template <typename T>
Status AssembleLastTokenLayer<T>::Forward(const std::vector<Tensor>& input_tensors,
                                          std::vector<Tensor>& output_tensors) {
  size_t batch_size = input_tensors[1].shape[0] - 1;
  size_t hidden_size = input_tensors[0].shape[1];

  void* input_ptr = input_tensors[0].GetPtr<void>();
  void* output_ptr = output_tensors[0].GetPtr<void>();

  void* seq_len_offset = malloc((batch_size + 1) * sizeof(uint64_t));
  MemcpyAsync(seq_len_offset, input_tensors[1].GetPtr<void>(), (batch_size + 1) * sizeof(uint64_t),
              MEMCPY_DEVICE_TO_HOST, context_->GetComputeStreams()[rank_]);
  StreamSynchronize(context_->GetComputeStreams()[rank_]);

  size_t total_batch_offset = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    uint64_t* seq_len_ptr = reinterpret_cast<uint64_t*>(seq_len_offset);
    size_t cur_seq_len = seq_len_ptr[i + 1] - seq_len_ptr[i];
    size_t batch_offset = total_batch_offset * hidden_size * sizeof(T);

    size_t offset = batch_offset + (cur_seq_len - 1) * hidden_size * sizeof(T);
    MemcpyAsync(output_ptr + (i * hidden_size * sizeof(T)), input_ptr + offset, hidden_size * sizeof(T),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);

    total_batch_offset += cur_seq_len;
  }

  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].shape[0] = batch_size;
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}
template class AssembleLastTokenLayer<float>;
template class AssembleLastTokenLayer<float16>;

}  // namespace ksana_llm
