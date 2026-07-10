/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "batch_scheduler/event_driven_scheduler/model_perf/model_perf_backend.h"
#include "cache_manager/cache_layout/base_cache_layout.h"
#include "cache_manager/cache_manager_interface.h"
#include "distributed/distributed_control_coordinator.h"
#include "engine/model_driver.h"
#include "global_cache_connector/global_cache_connector_interface.h"
#include "ipc/ipc_func_wrapper.h"
#include "pd_v2/public/pd_v2_connector.h"
#include "runtime/weight_instance_inferface.h"
#include "transfer/transfer_engine.h"
#include "utils/channel.h"
#include "utils/grammar_backend.h"
#include "utils/status.h"

namespace ksana_llm {

// Forward declarations
class EventDrivenScheduleProcessor;
class ScheduleHookManager;
struct BatchSchedulerConfig;
struct ConnectorConfig;
struct PdV2RuntimeConfig;
class LocalEndpoint;

namespace pd_v2 {
class PdV2PrefillEndpoint;
}  // namespace pd_v2

// The serving engine define.
class InferenceEngine {
 public:
  explicit InferenceEngine(Channel<std::pair<Status, std::shared_ptr<Request>>>& request_queue,
                           std::shared_ptr<Environment> env, std::shared_ptr<Context> context);
  ~InferenceEngine();

  // Initialize inference engine:
  // load weights & register model instance & start rpc port.
  Status Initialize();

  void SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper);

  // Must be called before Initialize() on Prefill engines using pd_v2.
  void SetLocalEndpoint(std::shared_ptr<LocalEndpoint> local_endpoint);

  // Start the rpc service.
  Status Start();

  // Stop the rpc service.
  Status Stop();

  // Health: whether this (Decode) engine's connector currently sees at least
  // one healthy pd_v2 Prefill peer. False when this process is not a pd_v2
  // Decode node (no connector loaded, or Prefill role). Surfaced up through
  // the service layer to the Python /health endpoint.
  bool HasHealthyPrefillPeer() const;

 private:
  // Do warmup run.
  Status DoWarmupRun();

  // 初始化 ModelPerfBackend（仅在环境变量 ENABLE_MODEL_PERF=1 + MODEL_PERF_CONFIG 有效时调用）。
  // 返回是否成功激活了 ModelPerfBackend。
  bool InitModelPerfBackend(std::shared_ptr<EventDrivenScheduleProcessor> event_driven_schedule_processor,
                            const BatchSchedulerConfig& batch_scheduler_config);

  // 根据 pd_v2 / V1 连接器角色，推导 GlobalCacheFetcher 的 role 字符串。
  //   pd_v2 Prefill -> "pd_v2_prefill"，pd_v2 Decode -> "pd_v2_decode"，
  //   V1 Prefill    -> "prefill"，     V1 Decode     -> "decode"，其余 -> ""。
  std::string DetermineGlobalCacheFetcherRole(const ConnectorConfig& connector_config, bool connector_config_ok);

  // pd_v2 cross-node Prefill/Decode 引导入口：委托 pd_v2 层完成
  // dlopen + connector 创建/初始化（pd_v2::BuildEngineConnectorConfig +
  // CreateAndInitConnector），随后按角色把 connector 接入本引擎的调度器 hook 链 /
  // Prefill endpoint。当本进程不是 pd_v2 Prefill/Decode 节点时为 no-op。
  void BootstrapPdV2Connector(const std::shared_ptr<EventDrivenScheduleProcessor>& schedule_processor,
                              const std::shared_ptr<ScheduleHookManager>& hook_manager);

  // 注册 pd_v2 Decode 侧 hook 到调度器 hook 链（内部构造 LinUCB 所需的 load 快照回调）。
  void RegisterPdV2DecodeHook(const std::shared_ptr<EventDrivenScheduleProcessor>& schedule_processor,
                              const std::shared_ptr<ScheduleHookManager>& hook_manager,
                              const pd_v2::PdV2ConnectorConfig& pd_cfg);

  // 注册 pd_v2 Prefill 侧 endpoint 及 snapshot / barrier 回调。
  void RegisterPdV2PrefillEndpoint(const std::shared_ptr<EventDrivenScheduleProcessor>& schedule_processor,
                                   const pd_v2::PdV2ConnectorConfig& pd_cfg);

  // pd_v2 Decode warmup 前置：轮询等待本地 PeerSelector 出现健康的 Prefill peer。
  // 仅在本进程为 pd_v2 Decode 节点时生效，其余场景为 no-op。
  void WaitForPdV2DecodePeer();

 private:
  std::shared_ptr<Environment> env_;
  std::shared_ptr<Context> context_;

  // The channel used to pass request from endpoint.
  Channel<std::pair<Status, std::shared_ptr<Request>>>& request_queue_;

  // Manage all engine threads and queues.
  std::shared_ptr<ModelDriver> model_driver_ = nullptr;

  // The scheduler used for inference engine.
  std::shared_ptr<ScheduleProcessorInterface> schedule_processor_ = nullptr;

  // 仅在 ENABLE_MODEL_PERF=1 时非空；用于在 Start() 中跳过 warmup。
  std::shared_ptr<ModelPerfBackend> model_perf_backend_ = nullptr;

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

  // The model-level cache layout (holds all block cache layouts keyed by type).
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;

  // pd_v2 connector (loaded via dlopen on engine-side when ConnectorConfig
  // sets is_pd_v2 = true and group_role = DECODE). Kept alive for the
  // engine's lifetime; the dlopen handle is released on destruction.
  std::shared_ptr<pd_v2::PdV2ConnectorInterface> pd_v2_connector_ = nullptr;
  void* pd_v2_dlopen_handle_ = nullptr;
  std::shared_ptr<LocalEndpoint> local_endpoint_ = nullptr;
  // pd_v2 Prefill side: connector → PdV2PrefillEndpoint → request_queue_.
  std::shared_ptr<pd_v2::PdV2PrefillEndpoint> pd_v2_prefill_endpoint_ = nullptr;

  // Whether the handle loop terminated.
  std::atomic<bool> terminated_ = false;
};

}  // namespace ksana_llm
