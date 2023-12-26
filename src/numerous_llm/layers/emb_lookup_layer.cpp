/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/emb_lookup_layer.h"
#include "numerous_llm/kernels/nvidia/remove_me_later.h"

namespace numerous_llm {


Status EmbLookupLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  //weigth_shape = input_tensors[2].
  int vocab_size = 1;
  int hidden_units = 1;
  int bs = 1;
  int step = 1;
  int vocab_id = 0;
  if (input_tensors.size() > 3) {
    emb_lookup(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(), input_tensors[3].GetPtr<void>(), output_tensors[0].GetPtr<void>(), vocab_size, hidden_units, bs, step, vocab_id, stream_);
  } else {
    emb_lookup(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(), nullptr, output_tensors[0].GetPtr<void>(), vocab_size, hidden_units, bs, step, vocab_id, stream_);
  }
  return Status();
}
}  // namespace numerous_llm
