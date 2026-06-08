/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/emb_lookup_layer.h"

namespace ksana_llm {

Status CpuEmbLookupLayer::Forward(const Tensor* input_ids, const Tensor* emb_table, const Tensor* emb_weight,
                                  Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_ids, emb_table, emb_weight, output);
}

// input_ids [token_num]
// emb_table [token_num, hidden_units]
// emb_weight [vocab_size, hidden_units]
// output [token_num, hidden_units]
template <typename T>
Status CpuEmbLookupLayer::ForwardT(const Tensor* input_ids, const Tensor* emb_table, const Tensor* emb_weight,
                                   Tensor* output) {
  size_t token_num = input_ids->shape[0];
  size_t hidden_units = emb_weight->shape[1];

  int* cpu_input_tokens = reinterpret_cast<int*>(input_ids->GetPtr<void>());
  T* cpu_data_ptr = reinterpret_cast<T*>(emb_table->GetPtr<void>());
  T* emb_weight_ptr = reinterpret_cast<T*>(emb_weight->GetPtr<void>());
  for (size_t i = 0; i < token_num; i++) {
    memcpy(cpu_data_ptr + i * hidden_units, emb_weight_ptr + cpu_input_tokens[i] * hidden_units,
           hidden_units * sizeof(T));
  }
  MemcpyAsync(output->GetPtr<void>(), cpu_data_ptr, token_num * hidden_units * sizeof(T), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetPrimaryComputeStream());

  output->dtype = emb_weight->dtype;
  output->shape = {token_num, hidden_units};
  return Status();
}

}  // namespace ksana_llm
