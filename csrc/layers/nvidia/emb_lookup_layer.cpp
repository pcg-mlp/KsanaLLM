/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/emb_lookup_layer.h"
#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status EmbLookupLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  size_t parameter_index = 0ul;
  if (parameter_index < parameters.size()) {
    use_emb_scale_ = std::any_cast<bool>(parameters[parameter_index++]);
  }
  if (parameter_index < parameters.size()) {
    emb_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  }
  if (parameter_index < parameters.size()) {
    pos_weight_ = std::any_cast<void*>(parameters[parameter_index++]);
  }
  return Status();
}

// input_ids [token_num]
// ids_offsets [batch_size + 1]
// prefix_offsets [batch_size + 1]
// emb_weight [vocab_size, hidden_units]
// steps [token_num] (optional)
// emb_output [token_num, hidden_units]
template <typename T>
Status EmbLookupLayer::ForwardT(const Tensor* input_ids, const Tensor* ids_offsets, const Tensor* prefix_offsets,
                                const Tensor* emb_weight, Tensor* output, const Tensor* steps) {
  int vocab_size = emb_weight->shape[0];
  int hidden_units = emb_weight->shape[1];
  int bs = ids_offsets->shape[0] - 1;
  int token_num = input_ids->shape[0];
  int vocab_id = 0;
  const void* steps_ptr = steps != nullptr ? steps->GetPtr<void>() : nullptr;

  LookupEmbedding<T>(input_ids->GetPtr<void>(), ids_offsets->GetPtr<void>(), prefix_offsets->GetPtr<void>(),
                     emb_weight->GetPtr<void>(), pos_weight_, steps_ptr, output->GetPtr<void>(), use_emb_scale_,
                     emb_scale_, vocab_size, hidden_units, bs, vocab_id,
                     context_->device->GetPrimaryComputeStream().Get());
  output->shape = {static_cast<size_t>(token_num), static_cast<size_t>(hidden_units)};
  output->dtype = emb_weight->dtype;
  return Status();
}

Status EmbLookupLayer::Forward(const Tensor* input_ids, const Tensor* ids_offsets, const Tensor* prefix_offsets,
                               const Tensor* emb_weight, Tensor* output, const Tensor* steps,
                               const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_ids, ids_offsets, prefix_offsets, emb_weight, output, steps);
}

}  // namespace ksana_llm
