/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/batch_manager/batch_manager.h"
#include "numerous_llm/runtime/infer_request.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/tensor.h"
#include "numerous_llm/utils/waiter.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

namespace numerous_llm {

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

  llm_runtime_ = std::make_shared<LlmRuntime>(context_);

  return Status();
}

void BatchManager::InitReqsWithInferReqId(const int64_t req_id, const size_t batch_size) {
  std::lock_guard<std::mutex> guard(infer_reqs_maintainer_mutex_);
  infer_reqs_maintainer_[req_id].resize(batch_size, nullptr);
}

void BatchManager::SetReqsWithInferReqId(const int64_t req_id, const size_t batch_id,
                                         std::shared_ptr<InferRequest> infer_req) {
  std::lock_guard<std::mutex> guard(infer_reqs_maintainer_mutex_);
  infer_reqs_maintainer_[req_id][batch_id] = infer_req;
}

// Get reqs by infer req id
std::vector<std::shared_ptr<InferRequest>> &BatchManager::GetReqsWithInferReqId(const int64_t req_id) {
  std::lock_guard<std::mutex> guard(infer_reqs_maintainer_mutex_);
  return infer_reqs_maintainer_[req_id];
}

void BatchManager::EraseReqsWithInferReqId(const int64_t req_id) {
  std::lock_guard<std::mutex> guard(infer_reqs_maintainer_mutex_);
  const auto &maintainer_it = infer_reqs_maintainer_.find(req_id);
  if (maintainer_it != infer_reqs_maintainer_.end()) {
    infer_reqs_maintainer_.erase(req_id);
  }
}

Status BatchManager::RegisterModelInstance(const std::shared_ptr<ModelInstance> &model_instance) {
  NLLM_LOG_INFO << "register model instance " << model_instance->name << " : " << model_instance.get();
  model_instances_[model_instance->name] = model_instance;
  return Status();
}

Status BatchManager::Enqueue(int64_t req_id, const std::vector<std::vector<int>> &tokens,
                             const std::vector<SamplingConfig> &sampling_configs) {
  NLLM_LOG_INFO << "batch manager enqueue req id " << req_id << ", batch_size " << tokens.size();

  Status enqueue_status = Status(RetCode::RET_SUCCESS);

  // Split into multiple prompt
  if (tokens.size() != sampling_configs.size()) {
    return Status(RET_INVALID_ARGUMENT, "Size of tokens and sampling_configs should be equal.");
  }

  std::shared_ptr<Waiter> waiter = std::make_shared<Waiter>(tokens.size());
  InitReqsWithInferReqId(req_id, tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>();
    infer_req->req_id = req_id;
    infer_req->input_tokens = tokens[i];
    infer_req->output_tokens = infer_req->input_tokens;
    infer_req->sampling_config = sampling_configs[i];
    infer_req->waiter = waiter;

    infer_req->kv_cache_blocks.resize(context_->GetTensorParallelSize());
    infer_req->block_size = GetBlockManager()->GetBlockSize();

    // TODO(karlluo): pass it from config or request
    infer_req->model_name = "llama";

    infer_req->model_instance = model_instances_[infer_req->model_name];
    infer_req->infer_stage = InferStage::STAGE_CONTEXT;
    infer_req->step = 0;
    SetReqsWithInferReqId(infer_req->req_id, static_cast<size_t>(i), infer_req);

    enqueue_status = batch_scheduler_->AddInferRequest(infer_req);
    if (enqueue_status.OK()) {
      NLLM_LOG_INFO << "batch schdule add req id " << req_id << " batch id " << i << " and "
                    << infer_req->input_tokens.size() << " tokens";
    } else {
      NLLM_LOG_ERROR << "batch schdule add req id " << req_id << " batch id " << i << " and "
                     << infer_req->input_tokens.size() << " tokens failed, message: " << enqueue_status.ToString();
    }
  }

  return Status();
}

Status BatchManager::WaitDone(int64_t req_id, std::vector<std::vector<int>> &tokens) {
  NLLM_LOG_INFO << "waiting req_id " << req_id << " finish.";
  auto &infer_reqs_list = GetReqsWithInferReqId(req_id);
  tokens.resize(infer_reqs_list.size());
  Status infer_status = Status(RET_SUCCESS);
  for (size_t infer_req_idx = 0; infer_req_idx < infer_reqs_list.size(); ++infer_req_idx) {
    infer_reqs_list[infer_req_idx]->waiter->Wait();
    tokens[infer_req_idx] = infer_reqs_list[infer_req_idx]->output_tokens;
  }
  EraseReqsWithInferReqId(req_id);
  return infer_status;
}

Status BatchManager::WaitAllDone() { return Status(); }

Status BatchManager::Process() {
  while (!terminated_) {
    std::vector<std::shared_ptr<InferRequest>> scheduled_reqs;
    scheduled_reqs = batch_scheduler_->Schedule();
    if (scheduled_reqs.empty()) {
      continue;
    }

    NLLM_LOG_INFO << "batch scheduler result " << scheduled_reqs.size();
    llm_runtime_->Step(scheduled_reqs);
  }

  return Status();
}

Status BatchManager::Start() {
  batch_manager_thread_ = std::unique_ptr<std::thread>(new std::thread(&BatchManager::Process, this));

  return Status();
}

Status BatchManager::Stop() {
  NLLM_LOG_INFO << "Stop batch manager.";

  terminated_ = true;

  if (batch_manager_thread_ && batch_manager_thread_->joinable()) {
    batch_manager_thread_->join();
  }

  NLLM_LOG_INFO << "batch manager stopped.";
  return Status();
}

}  // namespace numerous_llm
