/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/emb_lookup_layer.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/kernels/ascend/embedding/embedding.h"

namespace ksana_llm {

template <typename T>
Status EmbLookupLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // weigth_shape = input_tensors[2].
  // input_tensors:
  //   0: input_ids
  //   1: offset
  //   2: prefix (not be used)
  //   3: emb_weight
  //   4: pos
  // output_tensors:
  //   0: emb_output
  SetDevice(rank_);

  int total_seq_len = input_tensors[0].shape[0];
  int hidden_units = input_tensors[3].shape[1];

  Tensor input_ids = input_tensors[0];
  Tensor embedding_table = input_tensors[3];

  aclTensor* input_tensor = input_ids.ResetDeviceTensor(DataType::TYPE_INT32, {static_cast<int64_t>(total_seq_len)});

  aclTensor* embedding_tensor = embedding_table.ResetDeviceTensor(
      DataType::TYPE_FP16,
      {static_cast<int64_t>(embedding_table.shape[0]), static_cast<int64_t>(embedding_table.shape[1])});

  aclTensor* output_tensor = output_tensors[0].ResetDeviceTensor(
      DataType::TYPE_FP16, {static_cast<int64_t>(total_seq_len), static_cast<int64_t>(hidden_units)});

  if (input_tensors.size() > 4) {
    Tensor position_table = input_tensors[4];
    aclTensor* position_tensor = position_table.ResetDeviceTensor(
        DataType::TYPE_FP16,
        {static_cast<int64_t>(position_table.shape[0]), static_cast<int64_t>(position_table.shape[1])});
    LookupEmbedding(input_tensor, position_tensor, position_table.GetDeviceTensor(), output_tensor,
                    context_->GetComputeStreams()[rank_].Get(), GetWorkSpaceFunc());
  } else {
    LookupEmbedding(input_tensor, embedding_tensor, nullptr, output_tensor, context_->GetComputeStreams()[rank_].Get(),
                    GetWorkSpaceFunc());
  }

  output_tensors[0].shape = {static_cast<size_t>(total_seq_len), static_cast<size_t>(hidden_units)};
  output_tensors[0].dtype = input_tensors[3].dtype;
  return Status();
}
template class EmbLookupLayer<float>;
template class EmbLookupLayer<float16>;
}  // namespace ksana_llm
