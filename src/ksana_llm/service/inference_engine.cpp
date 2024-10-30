/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/service/inference_engine.h"

#include <memory>
#include <thread>

#include "ksana_llm/cache_manager/cache_manager_factory.h"
#include "ksana_llm/periphery/version_reporter.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

InferenceEngine::InferenceEngine(Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue)
    : request_queue_(request_queue) {
  Initialize();
}

InferenceEngine::~InferenceEngine() {
  model_instances_.clear();
  if (block_manager_) {
    delete block_manager_;
    block_manager_ = nullptr;
  }
}

Status InferenceEngine::Initialize() {
  std::shared_ptr<Environment> env = Singleton<Environment>::GetInstance();
  if (!env) {
    return Status(RET_INVALID_ARGUMENT, "The Environment is nullptr.");
  }

  context_.reset(new Context(env->GetTensorParallelSize(), env->GetPipeLineParallelSize()));

  // Initialize global block manager.
  BlockManagerConfig block_manager_config;
  Status status = env->GetBlockManagerConfig(block_manager_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get block manager config error:" + status.ToString());
  }
  block_manager_ = new BlockManager(block_manager_config, context_);
  SetBlockManager(block_manager_);

  ProfilerConfig profiler_config;
  status = env->GetProfilerConfig(profiler_config);
  Singleton<Profiler>::GetInstance()->Init(profiler_config);

  // Load model instances.
  std::unordered_map<std::string, ModelConfig> model_configs;
  status = env->GetModelConfigs(model_configs);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get model configs error:" + status.ToString());
  }
  KLLM_LOG_DEBUG << "Get model instance size: " << model_configs.size();

  size_t max_batch_size = 0;
  size_t max_vocab_size = 0;
  for (auto &[model_name, model_config] : model_configs) {
    max_batch_size = std::max(max_batch_size, (size_t)model_config.max_batch_size);
    max_vocab_size = std::max(max_vocab_size, (size_t)model_config.vocab_size);
  }

  // Create batch manager.
  BatchSchedulerConfig batch_scheduler_config;
  status = env->GetBatchSchedulerConfig(batch_scheduler_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get batch manager config error:" + status.ToString());
  }

  batch_scheduler_config.max_batch_size = max_batch_size;
  batch_scheduler_config.max_vocab_size = max_vocab_size;
  KLLM_LOG_DEBUG << "Batch Scheduler Config Max Batch Size = " << max_batch_size
                 << " Max Vocab Size = " << max_vocab_size;
  batch_manager_ = std::make_shared<BatchManager>(context_);

  // Register model instance.
  for (auto &[model_name, model_config] : model_configs) {
    std::shared_ptr<ModelInstance> model_instance = std::make_shared<ModelInstance>(model_config, context_);
    model_instance->Load();

    // Register model instance.
    model_instances_.push_back(model_instance);
    batch_manager_->RegisterModelInstance(model_instance);
  }

  // Create cache manager.
  CacheManagerConfig cache_manager_config;
  status = env->GetCacheManagerConfig(cache_manager_config);
  cache_manager_ = CacheManagerFactory::CreateCacheManager(cache_manager_config);

  // Create batch scheduler.
  batch_scheduler_ = std::make_shared<BatchScheduler>(batch_scheduler_config, context_->GetTensorParallelSize());
  batch_scheduler_->SetCacheManager(cache_manager_);

  // Create llm runtime
  llm_runtime_ = std::make_shared<LlmRuntime>(batch_scheduler_config, context_);

  batch_manager_->SetBatchScheduler(batch_scheduler_);
  batch_manager_->SetLlmRuntime(llm_runtime_);

  if (Singleton<Environment>::GetInstance()->IsReportVersion()) {
    VersionReporter::GetInstance().Init();
  }
  return Status();
}

Status InferenceEngine::HandleRequest(std::shared_ptr<Request> &req) {
  opentelemetry::common::KeyValueIterableView<std::unordered_map<std::string, std::string>> attributes(*req->req_ctx);
  REPORT_COUNTER(forward_req_total_num, static_cast<size_t>(1), attributes);
  REPORT_METRIC(metric_input_tokens_num, req->input_tokens.size(), attributes);

  Status handle_req_status = batch_manager_->Enqueue(req);
  if (!handle_req_status.OK()) {
    REPORT_COUNTER(forward_req_error_num, static_cast<size_t>(1), attributes);
    return handle_req_status;
  }
  return Status();
}

Status InferenceEngine::HandleLoop() {
  KLLM_LOG_DEBUG << "Start handler";

  while (!terminated_) {
    std::pair<Status, std::shared_ptr<Request>> req_pair;
    request_queue_.Read(&req_pair);
    if (terminated_) {
      break;
    }

    Status status = req_pair.first;
    if (status.GetCode() == RET_TERMINATED) {
      break;
    }

    std::shared_ptr<Request> req = req_pair.second;
    if (req) {
      HandleRequest(req);
    }
  }

  return Status();
}

Status InferenceEngine::StartHandler() {
  handle_thread_ = std::thread(&InferenceEngine::HandleLoop, this);
  return Status();
}

Status InferenceEngine::DoWarmupRun() {
  pybind11::gil_scoped_release release;
  KLLM_LOG_INFO << "Start to do warmup run";
  auto warmup_run_input = std::make_shared<KsanaPythonInput>();
  // Prepare the warm up input.
  warmup_run_input->input_tokens = std::vector<int>{1};
  // Warm up with one context and one decoding.
  warmup_run_input->sampling_config.max_new_tokens = 2;
  warmup_run_input->sampling_config.ignore_eos = true;

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto req = std::make_shared<Request>(warmup_run_input, req_ctx);
  req->waiter = std::make_shared<Waiter>(1);
  HandleRequest(req);

  // Wait the warm up.
  req->waiter->Wait();

  for (const auto &[output, req_logprobs, total_score] : req->output_group) {
    KLLM_CHECK_WITH_INFO(req->input_tokens.size() + req->padded_size < output.size(),
                         "Ksana warmup run generate empty output tokens. Warmup inference run failed");
  }
  KLLM_LOG_INFO << "End to do warmup run";
  pybind11::gil_scoped_acquire acquire;
  return Status();
}

Status InferenceEngine::Start() {
  // Reset block num via device memory usage.
  block_manager_->ResetPreAllocatedBlocks();

  // Check block number, the block number is determined after all models loaded.
  BatchSchedulerConfig batch_scheduler_config;
  Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
  KLLM_CHECK_WITH_INFO((block_manager_->GetDeviceFreeBlockNumber() * block_manager_->GetBlockTokenNum()) >=
                           (batch_scheduler_config.max_token_len),
                       "Total device block_num * block_token_size must large than max_token_len.");

  // Initialize blocks from block manager.
  cache_manager_->InitializeCachedBlocks();

  // Start batch manager.
  batch_manager_->Start();

  // Start service handler.
  StartHandler();

#ifndef ENABLE_ACL
  // Start warmup run
  DoWarmupRun();
#endif

  return Status();
}

Status InferenceEngine::Stop() {
  if (terminated_) {
    return Status();
  }

  terminated_ = true;
  handle_thread_.join();

  // Wait all request done.
  KLLM_LOG_INFO << "Waiting all running request.";
  Status status = batch_manager_->WaitAllDone();
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Wait all requests done error:" << status.ToString();
  }

  // Stop the batch manger.
  KLLM_LOG_INFO << "Stop batch manager.";
  batch_manager_->Stop();
  batch_manager_ = nullptr;
  llm_runtime_ = nullptr;

  if (Singleton<Environment>::GetInstance()->IsReportVersion()) {
    VersionReporter::GetInstance().StopReporting();
    VersionReporter::GetInstance().Destroy();
  }

  KLLM_LOG_INFO << "The Inference Engine has stopped.";
  return Status();
}

}  // namespace ksana_llm
