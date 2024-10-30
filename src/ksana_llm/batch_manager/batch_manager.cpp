/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

#include "ksana_llm/batch_manager/batch_manager.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

BatchManager::BatchManager(std::shared_ptr<Context> context) {
  context_ = context;
  queue_waiter_ = std::make_shared<Waiter>(1);
}

Status BatchManager::RegisterModelInstance(const std::shared_ptr<ModelInstance> &model_instance) {
  KLLM_LOG_DEBUG << "register model instance " << model_instance->name << " : " << model_instance.get();
  model_instances_[model_instance->name] = model_instance;
  return Status();
}

void BatchManager::SetBatchScheduler(std::shared_ptr<BatchSchedulerInterface> batch_scheduler) {
  batch_scheduler_ = batch_scheduler;
}

void BatchManager::SetLlmRuntime(std::shared_ptr<LlmRuntime> llm_runtime) { llm_runtime_ = llm_runtime; }

Status BatchManager::Enqueue(std::shared_ptr<Request> &req) {
  KLLM_LOG_DEBUG << "batch manager enqueue req id " << req->req_id;

  Status enqueue_status = Status(RetCode::RET_SUCCESS);

  if (model_instances_.find(req->model_name) == model_instances_.end()) {
    KLLM_LOG_ERROR << "req->model_name=" << req->model_name << " not found!";
    req->finish_status = Status(RET_INVALID_ARGUMENT, fmt::format("Model {} not found.", req->model_name));
    req->waiter->Notify();
    return req->finish_status;
  }
  const std::shared_ptr<ModelInstance> &model_instance = model_instances_[req->model_name];

  // Update `stop_token_ids` based on the config of the requested model.
  std::vector<int> &stop_token_ids = req->sampling_config.stop_token_ids;
  if (req->sampling_config.ignore_eos) {  // Ignore any end ids.
    stop_token_ids.clear();
  } else {  // Supplement the end ids in model config or generation config.
    for (int end_id : model_instance->GetModelConfig().end_ids) {
      if (std::find(stop_token_ids.begin(), stop_token_ids.end(), end_id) == stop_token_ids.end()) {
        stop_token_ids.push_back(end_id);
      }
    }
  }

  std::vector<std::shared_ptr<InferRequest>> infer_request_group;
  for (size_t i = 0; i < req->output_group.size(); i++) {
    std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(req, i);
    infer_request_group.push_back(infer_req);

    infer_req->kv_cache_blocks.resize(context_->GetTensorParallelSize());
    infer_req->block_token_num = GetBlockManager()->GetBlockTokenNum();
    infer_req->model_instance = model_instance;
    infer_req->pad_id = model_instance->GetModelConfig().pad_id;
    infer_req->infer_stage = InferStage::STAGE_CONTEXT;
    infer_req->step = 0;
  }

  for (auto &infer_req : infer_request_group) {
    infer_req->SetReqGroup(infer_request_group);
  }

  enqueue_status = batch_scheduler_->AddInferRequest(infer_request_group);
  if (enqueue_status.OK()) {
    KLLM_LOG_DEBUG << "batch scheduler: added req id " << req->req_id << " and "
                   << infer_request_group[0]->input_tokens.size() << " input tokens";
  } else {
    KLLM_LOG_ERROR << "batch scheduler: add req id " << req->req_id << " and "
                   << infer_request_group[0]->input_tokens.size()
                   << " input tokens failed, message: " << enqueue_status.ToString();
    if (req->sampling_config.num_beams > 1) {
      for (auto &infer_req : infer_request_group) {
        infer_req->ClearReqGroup();
      }
    }
    return enqueue_status;
  }

  // Notify the scheduler only after the current batch of requests has been enqueued.
  if (req->last_in_batch) {
    queue_waiter_->Notify();
  }
  return Status();
}

Status BatchManager::WaitAllDone() { return Status(); }

Status BatchManager::Process() {
  GetBlockManager()->SetDeviceId(0);
  while (!terminated_) {
    std::vector<std::shared_ptr<InferRequest>> scheduled_reqs;

    scheduled_reqs = batch_scheduler_->Schedule();

    if (scheduled_reqs.empty()) {
      if (batch_scheduler_->IsIdle()) {
        queue_waiter_->Wait();
        queue_waiter_->Reset(1);
      }
      continue;
    }

    {
      Status status = llm_runtime_->Step(scheduled_reqs);
      if (!status.OK()) {
        KLLM_LOG_ERROR << status.ToString();
      }
    }
  }

  return Status();
}

Status BatchManager::Start() {
  batch_manager_thread_ = std::unique_ptr<std::thread>(new std::thread(&BatchManager::Process, this));
  return Status();
}

Status BatchManager::Stop() {
  KLLM_LOG_INFO << "Stop batch manager.";

  terminated_ = true;

  // Break process loop.
  queue_waiter_->Notify();

  if (batch_manager_thread_ && batch_manager_thread_->joinable()) {
    batch_manager_thread_->join();
  }

  KLLM_LOG_INFO << "batch manager stopped.";
  return Status();
}

}  // namespace ksana_llm
