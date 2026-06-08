/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/bmm.h"

#include "layers/batched_matmul_layer.h"

namespace ksana_llm {

Bmm::Bmm(const std::string& weight_name, const LayerCreationContext& creation_context,
         const LinearComputeBackend& linear_compute_backend) {
  bmm_layer_ = creation_context.matmul_layer_factory->AutoCreateLayer(
      creation_context.base_weight, "", TYPE_VOID, creation_context.input_type, creation_context.output_type,
      linear_compute_backend, {});

  bmm_layer_->SetWorkspaceBuffer(creation_context.workspace_mgr->GetWorkspace(bmm_layer_->GetWorkspaceSize()));
  bmm_layer_->Preprocess(creation_context.model_config, creation_context.runtime_config);

  context_ = creation_context.context;
  rank_ = creation_context.rank;

  weight_ = creation_context.base_weight->GetModelWeights(weight_name);
}

Status Bmm::Forward(const Tensor& input, Tensor& output) {
  if (input.shape[1] == weight_.shape[0] && input.shape[2] == weight_.shape[1]) {
    return bmm_layer_->Forward(&input, &weight_, &output);
  }
  KLLM_THROW("The input shapes: " + Vector2Str(std::vector<size_t>(input.shape)) + " and weight shapes: " +
             Vector2Str(std::vector<size_t>(weight_.shape)) + " of bmm that have not been implemented yet.");
  return Status();
}

}  // namespace ksana_llm
