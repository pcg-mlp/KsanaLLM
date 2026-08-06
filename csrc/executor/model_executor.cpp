/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/model_executor.h"

#include "models/base/base_model.h"
#include "runtime/profile_metrics.h"

namespace ksana_llm {

namespace {

ForwardMetricsKind MetricsKindFromRunMode(RunMode run_mode) {
  return run_mode == RunMode::kNextN ? ForwardMetricsKind::kSpeculative : ForwardMetricsKind::kMain;
}

ProfileMetrics* ResolveProfileRoot(std::shared_ptr<NewModelInput>& model_input,
                                   std::shared_ptr<NewModelOutput>& model_output) {
  if (model_output->step_metrics != nullptr) {
    return model_output->step_metrics;
  }
  if (model_input->schedule_output && model_input->schedule_output->step_metrics != nullptr) {
    model_output->step_metrics = model_input->schedule_output->step_metrics;
    return model_output->step_metrics;
  }
  return &model_output->profile_metrics;
}

}  // namespace

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
  // Propagate run_mode so ModelSampler::Sampling can tell apart main-forward sampling from
  // MTP/DSpark/speculative-forward sampling per call, instead of relying on a persisted request field.
  model_output->run_mode = model_input->run_mode;
  // 固化本步 logits_offset: fast path 下共享 ExecutorInferRequest 可能被下一 schedule 覆写,
  model_output->frozen_logits_offsets.clear();
  model_output->frozen_logits_offsets.reserve(model_input->reqs.size());
  for (const auto& fr : model_input->reqs) {
    model_output->frozen_logits_offsets.push_back(fr.logits_offset);
  }
  const size_t slot = model_input->slot_index;
  const int64_t* prev_ring_dev = model_input->prev_ring_dev;
  model_output->slot_index = slot;

  const bool gpu_input_already_prepared = model_input->gpu_input_ready;
  const bool need_profile = model_input->schedule_output && model_input->schedule_output->enable_profile_metrics;
  ProfileMetrics* profile_root = ResolveProfileRoot(model_input, model_output);
  ForwardPerfMetrics* current_sub = nullptr;
  int64_t forward_begin_ns = 0;
  if (need_profile) {
    const bool is_first_forward = profile_root->forward_metrics.empty();
    current_sub = &profile_root->GetOrCreate(MetricsKindFromRunMode(model_input->run_mode));
    forward_begin_ns = ModelPerfNowNs();
    if (is_first_forward) {
      profile_root->schedule_forward_interval_ns =
          forward_begin_ns - model_input->schedule_output->schedule_dispatch_time_ns;
    }
    model_instance_->SetForwardProfileMetrics(profile_root, current_sub);
  }
  Status result_status = Status();
#ifdef ENABLE_CUDA
  result_status =
      model_instance_->Forward(model_weight_->GetWeight(), InferStage::kDecode, model_input->reqs, hidden_unit,
                               model_input->run_mode, slot, prev_ring_dev, gpu_input_already_prepared);
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
        model_instance_->Forward(model_weight_->GetWeight(), stage, stage_reqs, hidden_unit, model_input->run_mode,
                                 slot, prev_ring_dev, gpu_input_already_prepared);
    if (!status.OK()) {
      result_status = status;
    }
  }
#endif
  if (need_profile && current_sub != nullptr) {
    current_sub->forward_time_ns = ModelPerfNowNs() - forward_begin_ns;
    model_instance_->SetForwardProfileMetrics(nullptr, nullptr);
  }
  // 透传本步 (req_id, last_offset) 设备指针, 供 HandlePostprocess 内 WriteRing 写入 ring slot.
  // build-ahead 关闭或尚未 fill 时 GetCurBatchPairs 返回 nullptr/0, WriteRing 不会被调用.
  model_instance_->GetCurBatchPairs(slot, &model_output->cur_pairs_dev, &model_output->B_cur);
  return result_status;
}

}  // namespace ksana_llm
