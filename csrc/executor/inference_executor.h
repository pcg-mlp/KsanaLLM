/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <mutex>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/context.h"
#include "configure/environment.h"
#include "distributed/distributed_control_coordinator.h"
#include "distributed/distributed_data_coordinator.h"
#include "executor/model_runner.h"
#include "global_cache_connector/global_cache_connector_interface.h"
#include "ipc/ipc_func_wrapper.h"
#include "pd_v2/prefill/pd_v2_kv_barrier.h"
#include "pd_v2/prefill/pd_v2_kv_dispatcher.h"
#include "pd_v2/prefill/pd_v2_layer_event_tracker.h"
#include "pd_v2/public/pd_v2_connector.h"
#include "runtime/hidden_unit_buffer_pool.h"
#include "runtime/model_instance.h"
#include "runtime/weight_instance_inferface.h"

#ifdef ENABLE_CUDA
#  include "executor/nccl_executor_handler.h"
#endif

namespace ksana_llm {

struct PdV2RuntimeConfig;

// Used to execute the real forward on device.
class InferenceExecutor {
 public:
  InferenceExecutor(int executor_id, std::shared_ptr<Environment> env, std::shared_ptr<Context> context);
  ~InferenceExecutor();

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
  // pd_v2 引导：委托 pd_v2 层完成 dlopen + connector 创建/初始化
  // （pd_v2::BuildExecutorConnectorConfig + CreateAndInitConnector），随后注册
  // KV-pool 回调；Prefill 角色再建立 per-layer KV-write 基础设施。非 pd_v2 进程为 no-op。
  void BootstrapPdV2Connector();

  // pd_v2 Prefill-only：搭建 per-layer KV-write 基础设施
  // （PdV2LayerEventTracker + PdV2KvDispatcher + PdV2KvBarrier 及其回调接线）。
  void SetupPdV2PrefillKvWrite(const PdV2RuntimeConfig& pd_v2_rt);

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
  // The SWA block allocator manager; nullptr when SWA is disabled.
  std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager_ = nullptr;

  // The executor runtime object.
  std::shared_ptr<ExecutorRuntime> executor_runtime_ = nullptr;

  // The model-level cache layout (holds all block cache layouts keyed by type).
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;

  std::shared_ptr<ModelRunner> model_runner_ = nullptr;

#ifdef ENABLE_CUDA
  // NCCL executor handler for managing NCCL operations on this executor's GPU
  std::shared_ptr<NcclExecutorHandler> nccl_executor_handler_ = nullptr;
#endif

  // Global cache connector for fetching KV cache from global store
  std::shared_ptr<GlobalCacheConnectorInterface> global_cache_connector_ = nullptr;

  // pd_v2 executor-side connector. Owned by InferenceExecutor; the dlopen
  // handle is closed on destruction. Only populated when ConnectorConfig
  // enables pd_v2.
  std::shared_ptr<pd_v2::PdV2ConnectorInterface> pd_v2_connector_ = nullptr;
  void* pd_v2_dlopen_handle_ = nullptr;

  // pd_v2 Prefill-only per-rank: per-layer event tracker + KV dispatcher
  // + post-forward barrier. Stays null on Decode role / when pd_v2 is
  // disabled. The tracker is also registered as the
  // Singleton<PdV2LayerEventTracker> instance so attention layers can
  // call IsActive()-gated RecordLayerProgress.
  std::shared_ptr<pd_v2::PdV2LayerEventTracker> pd_v2_layer_tracker_ = nullptr;
  std::unique_ptr<pd_v2::PdV2KvDispatcher> pd_v2_kv_dispatcher_ = nullptr;
  std::unique_ptr<pd_v2::PdV2KvBarrier> pd_v2_kv_barrier_ = nullptr;
};

}  // namespace ksana_llm
