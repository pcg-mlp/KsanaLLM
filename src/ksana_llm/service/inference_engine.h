/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_manager/batch_manager.h"
#include "ksana_llm/block_manager/block_manager.h"
#include "ksana_llm/cache_manager/cache_manager_interface.h"
#include "ksana_llm/profiler/collector.h"
#include "ksana_llm/runtime/llm_runtime.h"
#include "ksana_llm/utils/channel.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The serving engine define.
class InferenceEngine {
 public:
  explicit InferenceEngine(Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue);
  ~InferenceEngine();

  // Start the rpc service.
  Status Start();

  // Stop the rpc service.
  Status Stop();

  // Start the handler loop.
  Status StartHandler();

  // Handle one request.
  Status HandleRequest(std::shared_ptr<Request> &req);

 private:
  // Initialize inference engine:
  // load weights & register model instance & start rpc port.
  Status Initialize();

  // Execute the handle loop.
  Status HandleLoop();

  // Do warmup run
  Status DoWarmupRun();

 private:
  // The channel used to pass request from endpoint.
  Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue_;

  // Global context for inference
  std::shared_ptr<Context> context_ = nullptr;

  // The global block manager.
  BlockManager *block_manager_ = nullptr;

  // The profiler.
  ProfileCollector *profile_collector_ = nullptr;

  // The batch manager for the whole inference.
  std::shared_ptr<BatchManager> batch_manager_ = nullptr;

  // The batch scheduler used for inference engine.
  std::shared_ptr<BatchScheduler> batch_scheduler_ = nullptr;

  // The runtime instance used for inference engine.
  std::shared_ptr<LlmRuntime> llm_runtime_ = nullptr;

  // The cache manager inference used for inference engine.
  std::shared_ptr<CacheManagerInterface> cache_manager_ = nullptr;

  // The model instances this service support.
  std::vector<std::shared_ptr<ModelInstance>> model_instances_;

  // Whether the handle loop terminated.
  std::atomic<bool> terminated_ = false;

  // The async thread used to hanle main loop.
  std::thread handle_thread_;
};

}  // namespace ksana_llm
