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

BatchManager::BatchManager(const BatchManagerConfig &batch_manager_config, std::shared_ptr<Context> context) {
  batch_manager_config_ = batch_manager_config;
  context_ = context;

  Initialize();
}

Status BatchManager::Initialize() {
  batch_scheduler_ =
      std::make_shared<BatchScheduler>(batch_manager_config_.batch_scheduler_config, context_->GetTensorParallelSize());

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

  if (model_instances_.find(req->model_name) == model_instances_.end()) {
    NLLM_LOG_ERROR << "req->model_name=" << req->model_name << " not found!";
    req->finish_status = Status(RET_INVALID_ARGUMENT, fmt::format("Model {} not found.", req->model_name));
    req->waiter->Notify();
    return req->finish_status;
  }
  std::vector<std::shared_ptr<InferRequest>> infer_request_group;
  for (int i = 0; i < req->output_group.size(); i++) {
    std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(req, i);
    infer_request_group.push_back(infer_req);

    infer_req->kv_cache_blocks.resize(context_->GetTensorParallelSize());
    infer_req->block_size = GetBlockManager()->GetBlockSize();
    infer_req->model_instance = model_instances_[req->model_name];
    infer_req->end_id = infer_req->model_instance->GetModelConfig().end_id;
    infer_req->pad_id = infer_req->model_instance->GetModelConfig().pad_id;
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
  }

  for (auto &infer_req : infer_request_group) {
    infer_req->SetReqGroup(infer_request_group);
  }

  enqueue_status = batch_scheduler_->AddInferRequest(infer_request_group);
  if (enqueue_status.OK()) {
    NLLM_LOG_DEBUG << "batch scheduler: added req id " << req->req_id << " and "
                   << infer_request_group[0]->input_tokens.size() << " input tokens";
  } else {
    NLLM_LOG_ERROR << "batch scheduler: add req id " << req->req_id << " and "
                   << infer_request_group[0]->input_tokens.size()
                   << " input tokens failed, message: " << enqueue_status.ToString();
    if (req->sampling_config.num_beams > 1) {
      for (auto &infer_req : infer_request_group) {
        infer_req->ClearReqGroup();
      }
    }
    return enqueue_status;
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
      Status status = llm_runtime_->Step(scheduled_reqs);
      if (!status.OK()) {
        NLLM_LOG_ERROR << status.ToString();
      }
    }
  }

  return Status();
}

Status BatchManager::Start() {
  // Check config here, because the block number is determined after all models loaded.
  size_t total_token_num = GetBlockManager()->GetDeviceFreeBlockNumber() * GetBlockManager()->GetBlockTokenNum();
#ifdef ENABLE_CUDA
  NLLM_CHECK_WITH_INFO(total_token_num >= (batch_manager_config_.batch_scheduler_config.max_token_len),
                       "Total device block_num * block_token_size must large than max_token_len.");
#endif

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
