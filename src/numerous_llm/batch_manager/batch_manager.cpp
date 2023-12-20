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

BatchManager::BatchManager(const BatchManagerConfig &batch_manager_config, std::shared_ptr<Context> contex) {
  batch_manager_config_ = batch_manager_config;
  contex_ = contex;

  Initialize();
}

Status BatchManager::Initialize() {
  batch_scheduler_ = std::make_shared<BatchScheduler>(batch_manager_config_.batch_scheduler_config);

  context_caching_ = std::make_shared<ContextCaching>(batch_manager_config_.context_caching_config);

  lora_coordinator_ = std::make_shared<LoraCoordinator>(batch_manager_config_.lora_coordinator_config);

  request_batching_ = std::make_shared<RequestBatching>(batch_manager_config_.request_batching_config);

  llm_runtime_ = std::make_shared<LlmRuntime>(contex_);

  return Status();
}

Status BatchManager::RegisterModelInstance(const std::shared_ptr<ModelInstance> &model_instance) {
  NLLM_LOG_INFO << "register model instance " << model_instance->name << " : " << model_instance.get();
  model_instances_[model_instance->name] = model_instance;
  return Status();
}

Status BatchManager::Enqueue(int req_id, const std::vector<std::vector<int>> &tokens,
                             const std::vector<SamplingConfig> &sampling_configs) {
  NLLM_LOG_INFO << "batch manager enqueue, batch_size:" << tokens.size();

  // Split into multiple prompt
  if (tokens.size() != sampling_configs.size()) {
    return Status(RET_INVALID_ARGUMENT, "Size of tokens and sampling_configs should be equal.");
  }

  std::shared_ptr<Waiter> waiter = std::make_shared<Waiter>(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>();
    infer_req->req_id = req_id;
    infer_req->input_tokens = tokens[i];
    infer_req->output_tokens = tokens[i];
    infer_req->sampling_config = sampling_configs[i];
    infer_req->waiter = waiter;
    infer_req->model_name = "llama";
    infer_req->model_instance = model_instances_[infer_req->model_name];
    infer_req->infer_stage = InferStage::STAGE_CONTEXT;

    NLLM_LOG_INFO << "infer_req.model_name: " << infer_req->model_name;

    batch_scheduler_->AddInferRequest(infer_req);
    NLLM_LOG_INFO << "batch schdule add request.";
  }

  waiter->Wait();
  NLLM_LOG_INFO << "batch schdule finish request.";

  return Status();
}

Status BatchManager::WaitDone(int req_id, std::vector<std::vector<int>> &tokens) {
  std::this_thread::sleep_for(std::chrono::seconds(3));
  return Status();
}

Status BatchManager::WaitAllDone() { return Status(); }

Status BatchManager::Process() {
  while (!terminated_) {
    std::vector<std::shared_ptr<InferRequest>> scheduled_reqs;
    scheduled_reqs = batch_scheduler_->Schedule();
    if (scheduled_reqs.empty()) {
      continue;
    }

    NLLM_LOG_INFO << "batch scheduler result:" << scheduled_reqs.size();
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
