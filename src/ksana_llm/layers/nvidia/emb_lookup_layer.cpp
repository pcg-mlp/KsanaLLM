/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/emb_lookup_layer.h"
#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status EmbLookupLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // weigth_shape = input_tensors[2].
  // input_tensors:
  //   0: input_ids
  //   1: offset
  //   2: emb_weight
  //   3: pos
  // output_tensors:
  //   0: emb_output
  int vocab_size = input_tensors[2].shape[0];
  int hidden_units = input_tensors[2].shape[1];
  int bs = input_tensors[1].shape[0] - 1;
  int total_seq_len = input_tensors[0].shape[0];
  int step = 1;
  int vocab_id = 0;

  if (input_tensors.size() > 3) {
    LookupEmbedding<T>(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
                    input_tensors[3].GetPtr<void>(), output_tensors[0].GetPtr<void>(), vocab_size, hidden_units, bs,
                    step, vocab_id, context_->GetComputeStreams()[rank_].Get());
  } else {
    LookupEmbedding<T>(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
                    nullptr, output_tensors[0].GetPtr<void>(), vocab_size, hidden_units, bs, step, vocab_id,
                    context_->GetComputeStreams()[rank_].Get());
  }
  output_tensors[0].shape = {static_cast<size_t>(total_seq_len), static_cast<size_t>(hidden_units)};
  output_tensors[0].dtype = input_tensors[2].dtype;
  return Status();
}

template class EmbLookupLayer<float>;
template class EmbLookupLayer<half>;
#ifdef ENABLE_BFLOAT16
template class EmbLookupLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
