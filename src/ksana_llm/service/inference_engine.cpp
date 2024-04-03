/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/service/inference_engine.h"
#include <memory>
#include <thread>
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

InferenceEngine::InferenceEngine(Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue)
    : request_queue_(request_queue) {
  Initialize();
}

InferenceEngine::~InferenceEngine() {
  if (block_manager_) {
    delete block_manager_;
    block_manager_ = nullptr;
  }

  if (profile_collector_) {
    delete profile_collector_;
    profile_collector_ = nullptr;
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
  profile_collector_ = new ProfileCollector(profiler_config);
  SetProfileCollector(profile_collector_);

  BatchManagerConfig batch_manager_config;
  status = env->GetBatchManagerConfig(batch_manager_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get batch manager config error:" + status.ToString());
  }

  // Load model instances.
  std::unordered_map<std::string, ModelConfig> model_configs;
  status = env->GetModelConfigs(model_configs);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get model configs error:" + status.ToString());
  }
  NLLM_LOG_DEBUG << "Get model instance size: " << model_configs.size();

  size_t max_batch_size = 0;
  size_t max_vocab_size = 0;
  for (auto &[model_name, model_config] : model_configs) {
    max_batch_size = std::max(max_batch_size, (size_t)model_config.max_batch_size);
    max_vocab_size = std::max(max_vocab_size, (size_t)model_config.vocab_size);
  }
  batch_manager_config.batch_scheduler_config.max_batch_size = max_batch_size;
  batch_manager_config.batch_scheduler_config.max_vocab_size = max_vocab_size;
  NLLM_LOG_DEBUG << "Batch Scheduler Config Max Batch Size = " << max_batch_size
                 << " Max Vocab Size = " << max_vocab_size;
  batch_manager_ = std::make_shared<BatchManager>(batch_manager_config, context_);

  for (auto &[model_name, model_config] : model_configs) {
    std::shared_ptr<ModelInstance> model_instance = std::make_shared<ModelInstance>(model_config, context_);
    model_instance->Load();

    // Register model instance.
    model_instances_.push_back(model_instance);
    batch_manager_->RegisterModelInstance(model_instance);
  }

  return Status();
}

Status InferenceEngine::HandleRequest(std::shared_ptr<Request> &req) {
  NLLM_LOG_DEBUG << "Handle request id " << req->req_id;
  Status handle_req_status = batch_manager_->Enqueue(req);
  if (!handle_req_status.OK()) {
    return handle_req_status;
  }
  return Status();
}

Status InferenceEngine::HandleLoop() {
  NLLM_LOG_DEBUG << "Start handler";

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

Status InferenceEngine::Start() {
  // Start profiler, invoked before batch manager.
  profile_collector_->Start();

  // Reset block num via device memory usage.
  block_manager_->ResetPreAllocatedBlocks();

  // Prepare prefix cache tokens if need
  block_manager_->PreparePrefixCacheBlocks();

  // Start batch manager.
  batch_manager_->Start();

  // Start service handler.
  StartHandler();

  return Status();
}

Status InferenceEngine::Stop() {
  if (terminated_) {
    return Status();
  }

  terminated_ = true;
  handle_thread_.join();

  // Wait all request done.
  NLLM_LOG_DEBUG << "Waiting all running request.";
  Status status = batch_manager_->WaitAllDone();
  if (!status.OK()) {
    NLLM_LOG_ERROR << "Wait all requests done error:" << status.ToString();
  }

  // Stop the batch manger.
  NLLM_LOG_DEBUG << "Stop batch manager.";
  batch_manager_->Stop();

  // Stop profiler, after profiler.
  profile_collector_->Stop();

  return Status();
}

}  // namespace ksana_llm
