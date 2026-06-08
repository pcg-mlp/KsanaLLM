/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/emb_lookup_layer.h"
#include "kernels/ascend/kernel_wrapper.h"

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
  atb::infer::GatherParam gather_param;
  atb_op_executor_.Init(rank, gather_param);
  return Status();
}

// input_ids
// offset
// prefix (not be used)
// emb_weight
// pos (optional)
// emb_output
Status EmbLookupLayer::Forward(const Tensor* input_ids, const Tensor* ids_offsets, const Tensor* prefix_offsets,
                               const Tensor* emb_weight, Tensor* output, const Tensor* steps,
                               const LayerExecContext& exec_ctx) {
  if (steps != nullptr) {
    KLLM_THROW("Not supported position embedding on NPU.");
  }

  int total_seq_len = input_ids->shape[0];
  int hidden_units = emb_weight->shape[1];
  output->shape = {static_cast<size_t>(total_seq_len), static_cast<size_t>(hidden_units)};
  output->dtype = emb_weight->dtype;

  reinterpret_cast<atb::Context*>(GetRuntimeContext())->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_op_executor_.ResetVariantPack();
  atb_op_executor_.SetInputTensor(emb_weight->GetPtr<void>(), emb_weight->shape,
                                  static_cast<aclDataType>(DataType(emb_weight->dtype)));
  atb_op_executor_.SetInputTensor(input_ids->GetPtr<void>(), input_ids->shape,
                                  static_cast<aclDataType>(DataType(input_ids->dtype)));
  atb_op_executor_.SetOutputTensor(output->GetPtr<void>(), output->shape,
                                   static_cast<aclDataType>(DataType(output->dtype)));
  atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext()), GetWorkSpaceFunc());

  return Status();
}

}  // namespace ksana_llm
