/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_manager/batch_manager.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

namespace ksana_llm {

BatchManager::BatchManager(const BatchManagerConfig &batch_manager_config, std::shared_ptr<Context> context) {
  batch_manager_config_ = batch_manager_config;
  context_ = context;

  Initialize();
}

Status BatchManager::Initialize() {
  batch_scheduler_ = std::make_shared<BatchScheduler>(batch_manager_config_.batch_scheduler_config, context_);

  context_caching_ = std::make_shared<ContextCaching>(batch_manager_config_.context_caching_config);

  lora_coordinator_ = std::make_shared<LoraCoordinator>(batch_manager_config_.lora_coordinator_config);

  request_batching_ = std::make_shared<RequestBatching>(batch_manager_config_.request_batching_config);

  llm_runtime_ = std::make_shared<LlmRuntime>(batch_manager_config_.batch_scheduler_config, context_);

  queue_waiter_ = std::make_shared<Waiter>(1);

  return Status();
}

Status BatchManager::RegisterModelInstance(const std::shared_ptr<ModelInstance> &model_instance) {
  NLLM_LOG_DEBUG << "register model instance " << model_instance->name << " : " << model_instance.get();
  model_instances_[model_instance->name] = model_instance;
  return Status();
}

Status BatchManager::Enqueue(std::shared_ptr<Request> &req) {
  NLLM_LOG_DEBUG << "batch manager enqueue req id " << req->req_id;

  Status enqueue_status = Status(RetCode::RET_SUCCESS);

  std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(req);

  infer_req->kv_cache_blocks.resize(context_->GetTensorParallelSize());
  infer_req->block_size = GetBlockManager()->GetBlockSize();

  if (model_instances_.find(req->model_name) == model_instances_.end()) {
    NLLM_LOG_ERROR << "req->model_name=" << req->model_name << " not found!";
    req->finish_status = Status(RET_INVALID_ARGUMENT, fmt::format("Model {} not found.", req->model_name));
    req->waiter->Notify();
    return req->finish_status;
  }
  infer_req->model_instance = model_instances_[req->model_name];
  infer_req->infer_stage = InferStage::STAGE_CONTEXT;
  infer_req->step = 0;

  // check if this request qualify to use prefix cache
  if (GetBlockManager()->GetPrefixCacheBlocksNumber() > 0) {
    infer_req->is_use_prefix_cache = GetBlockManager()->CheckReqIsValidForPrefixCache(infer_req->input_tokens);
    infer_req->prefix_cache_len = GetBlockManager()->GetPrefixCacheTokensNumber();
    infer_req->prefix_cache_blocks_number = GetBlockManager()->GetPrefixCacheBlocksNumber();
    GetBlockManager()->FillPrefixCacheBlocks(infer_req->kv_cache_blocks);
    // NOTE(karlluo): preallocate prefix kv cache for infer request
    NLLM_LOG_DEBUG << "req id " << infer_req->req_id << " is use prefix cache " << infer_req->is_use_prefix_cache;
  }

  enqueue_status = batch_scheduler_->AddInferRequest(infer_req);
  if (enqueue_status.OK()) {
    NLLM_LOG_DEBUG << "batch scheduler: added req id " << req->req_id << " and " << infer_req->input_tokens.size()
                   << " input tokens";
  } else {
    NLLM_LOG_ERROR << "batch scheduler: add req id " << req->req_id << " and " << infer_req->input_tokens.size()
                   << " input tokens failed, message: " << enqueue_status.ToString();
  }

  queue_waiter_->Notify();
  return Status();
}

Status BatchManager::WaitAllDone() { return Status(); }

Status BatchManager::Process() {
  GetBlockManager()->SetDeviceId(0);
  while (!terminated_) {
    std::vector<std::shared_ptr<InferRequest>> scheduled_reqs;

    {
      REPORT_TIME_US(batch_manager_schedule_us);
      scheduled_reqs = batch_scheduler_->Schedule();
    }

    if (scheduled_reqs.empty()) {
      if (batch_scheduler_->WaitingBufferEmpty() && batch_scheduler_->SwappedQueueEmtpy()) {
        queue_waiter_->Wait();
        queue_waiter_->Reset(1);
      }
      continue;
    }

    {
      REPORT_TIME_US(batch_manager_step_us);
      llm_runtime_->Step(scheduled_reqs);
    }
  }

  return Status();
}

Status BatchManager::Start() {
  // Check config here, because the block number is determined after all models loaded.
  NLLM_CHECK_WITH_INFO((GetBlockManager()->GetDeviceFreeBlockNumber() * GetBlockManager()->GetBlockTokenNum()) >=
                          (batch_manager_config_.batch_scheduler_config.max_token_len),
                       "Total device block_num * block_token_size must large than max_token_len.");

  batch_manager_thread_ = std::unique_ptr<std::thread>(new std::thread(&BatchManager::Process, this));

  return Status();
}

Status BatchManager::Stop() {
  NLLM_LOG_DEBUG << "Stop batch manager.";

  terminated_ = true;

  // Break process loop.
  queue_waiter_->Notify();

  if (batch_manager_thread_ && batch_manager_thread_->joinable()) {
    batch_manager_thread_->join();
  }

  NLLM_LOG_DEBUG << "batch manager stopped.";
  return Status();
}

}  // namespace ksana_llm
