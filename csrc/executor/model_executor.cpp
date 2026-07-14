/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/model_executor.h"

namespace ksana_llm {

void ModelExecutor::SetModelInstance(std::shared_ptr<ModelInstance> model_instance) {
  model_instance_ = model_instance;
}

void ModelExecutor::SetModelWeight(std::shared_ptr<WeightInstanceInterface> model_weight) {
  model_weight_ = model_weight;
}

void ModelExecutor::SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager) {
  block_allocator_manager_ = block_allocator_manager;
}

Status ModelExecutor::Forward(std::shared_ptr<NewModelInput> model_input, HiddenUnitDeviceBuffer* hidden_unit,
                              std::shared_ptr<NewModelOutput> model_output) {
  model_output->schedule_output = model_input->schedule_output;
  model_output->run_mode = model_input->run_mode;

  Status result_status = Status();
#ifdef ENABLE_CUDA
  result_status = model_instance_->Forward(model_weight_->GetWeight(), InferStage::kDecode, model_input->reqs,
                                           hidden_unit, model_input->run_mode);
#else
  for (const auto& stage : {InferStage::kContext, InferStage::kDecode}) {
    std::vector<ForwardRequest> stage_reqs;
    stage_reqs.reserve(model_input->reqs.size());
    for (const auto& req : model_input->reqs) {
      if (req.infer_stage == stage) {
        stage_reqs.emplace_back(req);
      }
    }
    const Status status =
        model_instance_->Forward(model_weight_->GetWeight(), stage, stage_reqs, hidden_unit, model_input->run_mode);
    if (!status.OK()) {
      result_status = status;
    }
  }
#endif
  return result_status;
}

}  // namespace ksana_llm
