/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>

#include "numerous_llm/batch_manager/batch_scheduler/batch_scheduler.h"
#include "numerous_llm/batch_manager/context_caching/context_caching.h"
#include "numerous_llm/batch_manager/lora_coordinator/lora_coordinator.h"
#include "numerous_llm/batch_manager/request_batching/request_batching.h"
#include "numerous_llm/block_manager/block_manager.h"
#include "numerous_llm/runtime/llm_runtime.h"
#include "numerous_llm/runtime/model_instance.h"
#include "numerous_llm/samplers/sampler.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class BatchManager {
 public:
  BatchManager(const BatchManagerConfig &batch_manager_config);

  // Register a model instance to current batch manager.
  Status RegisterModelInstance(const std::shared_ptr<ModelInstance> &model_instance);

  // Enqueue a request to waiting queue.
  Status Enqueue(int req_id, const std::vector<TensorMap> &tensor_maps,
                 const std::vector<SamplingConfig> &sampling_configs);

  // Wait request done and return output tensor maps.
  Status WaitDone(int req_id, std::vector<TensorMap> &tensor_maps);

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
  BatchManagerConfig batch_manager_config_;

  // The batch scheduler.
  std::shared_ptr<BatchScheduler> batch_scheduler_ = nullptr;

  // Used to cache kv cache of prefix part of input.
  std::shared_ptr<ContextCaching> context_caching_ = nullptr;

  // Manage the cpu&gpu memory block.
  std::shared_ptr<BlockManager> block_manager_ = nullptr;

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

  // The sampler instance.
  std::shared_ptr<Sampler> llm_sampler_ = nullptr;
};

}  // namespace numerous_llm
