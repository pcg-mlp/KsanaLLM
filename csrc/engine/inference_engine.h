/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "cache_manager/cache_manager_interface.h"
#include "distributed/distributed_control_coordinator.h"
#include "engine/model_driver.h"
#include "global_cache_connector/global_cache_connector_interface.h"
#include "ipc/ipc_func_wrapper.h"
#include "runtime/weight_instance_inferface.h"
#include "transfer/transfer_engine.h"
#include "utils/channel.h"
#include "utils/grammar_backend.h"
#include "utils/status.h"

namespace ksana_llm {

// The serving engine define.
class InferenceEngine {
 public:
  explicit InferenceEngine(Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue,
                           std::shared_ptr<Environment> env, std::shared_ptr<Context> context);
  ~InferenceEngine();

  // Initialize inference engine:
  // load weights & register model instance & start rpc port.
  Status Initialize();

  void SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper);

  // Start the rpc service.
  Status Start();

  // Stop the rpc service.
  Status Stop();

 private:
  // Do warmup run
  Status DoWarmupRun();

 private:
  std::shared_ptr<Environment> env_;
  std::shared_ptr<Context> context_;

  // The channel used to pass request from endpoint.
  Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue_;

  // Manage all engine threads and queues.
  std::shared_ptr<ModelDriver> model_driver_ = nullptr;

  // The scheduler used for inference engine.
  std::shared_ptr<ScheduleProcessorInterface> schedule_processor_ = nullptr;

  // object pool of schedule and sampling output buffer.
  std::shared_ptr<ScheduleOutputPool> schedule_output_pool_ = nullptr;
  std::shared_ptr<SamplingOutputPool> sampling_output_pool_ = nullptr;

  // Used for pipeline mode.
  std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator_ = nullptr;

  // The cache manager inference used for inference engine.
  std::vector<std::shared_ptr<CacheManagerInterface>> cache_managers_;

  // Transfer engine for layer progress handling
  std::shared_ptr<TransferEngine> transfer_engine_ = nullptr;

  // Global cache connector for supplementing local prefix cache
  std::shared_ptr<GlobalCacheConnectorInterface> global_cache_connector_ = nullptr;

  // IPC function wrapper for inter-process communication
  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_ = nullptr;

  // Whether the handle loop terminated.
  std::atomic<bool> terminated_ = false;
};

}  // namespace ksana_llm
