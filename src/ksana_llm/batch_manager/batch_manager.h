/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>

#include "ksana_llm/batch_manager/batch_scheduler/batch_scheduler.h"
#include "ksana_llm/batch_manager/context_caching/context_caching.h"
#include "ksana_llm/batch_manager/lora_coordinator/lora_coordinator.h"
#include "ksana_llm/batch_manager/request_batching/request_batching.h"
#include "ksana_llm/block_manager/block_manager.h"
#include "ksana_llm/runtime/context.h"
#include "ksana_llm/runtime/llm_runtime.h"
#include "ksana_llm/runtime/model_instance.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class BatchManager {
 public:
  BatchManager(const BatchManagerConfig &batch_manager_config, std::shared_ptr<Context> context);

  // Register a model instance to current batch manager.
  Status RegisterModelInstance(const std::shared_ptr<ModelInstance> &model_instance);

  // Enqueue a request to waiting queue.
  Status Enqueue(std::shared_ptr<Request> &request);

  // Wait all requests done.
  Status WaitAllDone();

  // Process and get next running jobs.
  Status Process();

  // Start the batch manager.
  Status Start();

  // Stop the batch manager.
  Status Stop();

 private:
  // Initialize the batch manager.
  Status Initialize();

 private:
  // The config for whole batch manager.
  BatchManagerConfig batch_manager_config_;

  // The global context.
  std::shared_ptr<Context> context_;

  // The batch scheduler.
  std::shared_ptr<BatchScheduler> batch_scheduler_ = nullptr;

  // Used to cache kv cache of prefix part of input.
  std::shared_ptr<ContextCaching> context_caching_ = nullptr;

  // Batching the requests.
  std::shared_ptr<RequestBatching> request_batching_ = nullptr;

  // Used to manage lora weights, load or unload.
  std::shared_ptr<LoraCoordinator> lora_coordinator_ = nullptr;

  // The process thread.
  std::unique_ptr<std::thread> batch_manager_thread_;

  // Whether batch manager should be stopped.
  std::atomic<bool> terminated_ = false;

  // The model name to model instance.
  std::unordered_map<std::string, std::shared_ptr<ModelInstance>> model_instances_;

  // The runtime instance.
  std::shared_ptr<LlmRuntime> llm_runtime_ = nullptr;

  // To guard result maintainer.
  std::mutex infer_reqs_maintainer_mutex_;

  // A waiter used to notify scheduler.
  std::shared_ptr<Waiter> queue_waiter_;
};

}  // namespace ksana_llm
