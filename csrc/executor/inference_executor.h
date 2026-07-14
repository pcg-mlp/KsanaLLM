/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <mutex>

#include "configure/context.h"
#include "configure/environment.h"
#include "distributed/distributed_control_coordinator.h"
#include "distributed/distributed_data_coordinator.h"
#include "executor/executor_block_manager.h"
#include "executor/model_runner.h"
#include "ipc/ipc_func_wrapper.h"
#include "runtime/draft_generator/draft_generator_controller.h"
#include "runtime/draft_generator/ptp_generator.h"
#include "runtime/draft_generator/trie_generator.h"
#include "runtime/hidden_unit_buffer_pool.h"
#include "runtime/model_instance.h"
#include "runtime/weight_instance_inferface.h"

#ifdef ENABLE_CUDA
#  include "executor/nccl_executor_handler.h"
#endif

namespace ksana_llm {

// Used to execute the real forward on device.
class InferenceExecutor {
 public:
  InferenceExecutor(int executor_id, std::shared_ptr<Environment> env, std::shared_ptr<Context> context);

  // Initialize executor, it will finish all necessary prepare work, such as
  // load model & weight, initialize device context, initialize all executor threads.
  Status Initialize();

  void SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper);

  // Block util the end of the world.
  void BlockingForever();

  // Start the executor.
  void Start();

  // Stop the executor.
  void Stop();

 private:
  int executor_id_ = -1;

  // Shared by all executor.
  std::shared_ptr<Environment> env_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;

  std::mutex mutex_;
  std::condition_variable cond_;
  std::atomic<bool> terminated_ = false;

  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_ = nullptr;

  // Used for pipeline mode.
  std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator_ = nullptr;

  // Used for report result to master.
  std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator_ = nullptr;
  std::shared_ptr<SamplingOutputPool> sampling_output_pool_ = nullptr;

  // manage hidden_unit_buffer, for both standalone and distributed mode.
  std::shared_ptr<HiddenUnitBufferPool> hidden_unit_buffer_pool_ = nullptr;

  // The model and weight instances, do not support multiple model or lora.
  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
  std::shared_ptr<WeightInstanceInterface> model_weight_ = nullptr;

  // The block allocator manager, used to manage device memory and kv block.
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager_ = nullptr;
  std::shared_ptr<ExecutorBlockManager> executor_block_manager_ = nullptr;

  // The executor runtime object.
  std::shared_ptr<ExecutorRuntime> executor_runtime_ = nullptr;

  std::shared_ptr<ModelRunner> model_runner_ = nullptr;

#ifdef ENABLE_CUDA
  // NCCL executor handler for managing NCCL operations on this executor's GPU
  std::shared_ptr<NcclExecutorHandler> nccl_executor_handler_ = nullptr;
#endif
};

}  // namespace ksana_llm
