/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <functional>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/environment.h"
#include "distributed/control_message.h"
#include "distributed/distributed_control_coordinator.h"
#include "distributed/distributed_data_coordinator.h"
#include "executor/executor_runtime.h"
#include "ipc/core/ipc_func_proxy.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/functions/report_layer_progress.h"
#include "runtime/sampling_output.h"
#include "runtime/schedule_output.h"
#include "utils/status.h"

namespace ksana_llm {

class GlobalCacheFetcher;
class GlobalCacheConnectorInterface;
class GlobalCachePutTracker;

enum {
  IPC_FUNC_ADD_EXECUTOR = 0,
  IPC_FUNC_LOAD_WEIGHT,
  IPC_FUNC_LOAD_MODEL,

  IPC_FUNC_RESET_PIPELINE_CONFIG,
  IPC_FUNC_CREATE_BLOCK_ALLOCATOR,
  IPC_FUNC_CREATE_EXECUTOR_RUNTIME,

  IPC_FUNC_DISPATCH_SCHEDULE_OUTPUT,
  IPC_FUNC_REPORT_SAMPLING_OUTPUT,
  IPC_FUNC_REPORT_LAYER_PROGRESS,

  IPC_FUNC_INIT_DEVICE_COMMUNICATION,
  IPC_FUNC_INIT_CUSTOM_COMMUNICATION,
  IPC_FUNC_INIT_MULTICAST_COMMUNICATION,
  IPC_FUNC_LISTEN_TCP_DATA_CHANNEL,
  IPC_FUNC_CONNECT_TCP_DATA_CHANNEL,
  IPC_FUNC_CONNECT_NCCL_DATA_CHANNEL,
  IPC_FUNC_INIT_REPORTER_CHANNEL,
  IPC_FUNC_GET_DEVICE_MEMORY,

  IPC_FUNC_CONNECTOR_INIT_NCCL_COMM_GROUP,
  IPC_FUNC_CONNECTOR_NCCL_SEND_DATA,
  IPC_FUNC_CONNECTOR_NCCL_RECV_DATA,
  IPC_FUNC_CONNECTOR_NCCL_SEND_BLOCKS,
  IPC_FUNC_CONNECTOR_NCCL_RECV_BLOCKS,
  IPC_FUNC_CONNECTOR_GET_NCCL_COMM_INFO,

  IPC_FUNC_GET_EP_COMMUNICATION_CONFIG,
  IPC_FUNC_SET_EP_COMMUNICATION_CONFIG,

  IPC_FUNC_GLOBAL_CACHE_QUERY,
  IPC_FUNC_GLOBAL_CACHE_QUERY_DONE,
  IPC_FUNC_GLOBAL_CACHE_PUT_DONE,

  // pd_v2: engine asks executor[rank] for its Mooncake-registered KV
  // pool base VA + shape. One round-trip per rank at first SubmitPrefill
  // call, cached afterwards.
  IPC_FUNC_PD_V2_GET_KV_POOL_BASE_VA,

  // pd_v2: executor → engine async report that all in-flight Mooncake
  // submitTransfer batches for a request have drained (sender-side
  // barrier). Engine then sends PrefillRequestComplete back to peer.
  IPC_FUNC_PD_V2_BARRIER_DONE,
};

// Used to call some pre-defined functions in different process.
class IpcFuncWrapper {
 public:
  IpcFuncWrapper(size_t executor_num, bool is_engine, int executor_id);
  ~IpcFuncWrapper();

  // Engine to executors.
  Status SetDeviceCommunicationContext(std::shared_ptr<Context> context);
  Status InitDeviceCommunication();

  // Exchange between executor and engine: broadcast IPC pointers for custom communication
  Status SetCustomCommunicationContext(std::shared_ptr<Context> context);
  Status InitCustomCommunication();

  Status SetMulticastCommunicationContext(std::shared_ptr<Context> context);
  Status InitMulticastCommunication();

  // Engine to executor.
  Status GetDeviceMemory(size_t* device_free, size_t* device_total);

  // Engine to executors.
  Status ResetPipelineConfig(const PipelineConfig& pipeline_config);

  // Engine to executors.
  void SetTcpDistributedDataCoordinator(std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator);
  Status ListenTcpDataChannel(std::vector<HostPort>& executor_host_ports);
  Status ConnectTcpDataChannel(const std::vector<HostPort>& executor_downstream_host_ports);

  // Engine to executors: Get EP communicator config from all executors.
  Status GetEPCommunicationConfig(std::vector<EPCommunicationConfig>& executor_comm_configs);
  // Engine to executors: Set aggregated EP communicator config to all executors.
  Status SetEPCommunicationConfig(const std::vector<EPCommunicationConfig>& aggregated_comm_configs);

#ifdef ENABLE_CUDA
  void SetNcclDistributedDataCoordinator(std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator);
  Status ConnectNcclDataChannel(ncclUniqueId nccl_unique_id);
#endif

  void SetReporterDistributedControlCoordinator(
      std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator);
  Status ConnectResultReporterChannel();

  // Set the cache layout for specific model instance.
  void SetBlockCacheLayout(std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

  // Engine to executor, blocking.
  Status SetWeightInstance(std::shared_ptr<WeightInstanceInterface> model_weight);
  Status LoadWeight();

  // Engine to executor, blocking.
  Status SetModelInstance(std::shared_ptr<ModelInstance> model_instance);
  Status LoadModel();

  // Engine to executor, blocking.
  Status SetEnvironment(std::shared_ptr<Environment> env);

  // Engine to executor, blocking.
  Status SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager,
                                  int dp_group_id);
  // Register the executor-side SWA block allocator manager so the callable can initialize it.
  Status SetSwaBlockAllocatorManager(std::shared_ptr<BlockAllocatorManagerInterface> swa_block_allocator_manager);
  Status CreateBlockAllocatorManager(const BlockAllocatorManagerConfig& block_allocator_manager_config,
                                     const BlockAllocatorManagerConfig& swa_block_allocator_manager_config = {});

  // Engine to executor, blocking.
  Status SetExecutorRuntime(std::shared_ptr<ExecutorRuntime> executor_runtime);
  Status SetExecutorEnvironment(std::shared_ptr<Environment> env);
  Status CreateExecutorRuntime(const VocabInfo& vocab_info);

  // Engine to executor.
  Status SetScheduleOutputRecvQueue(
      std::shared_ptr<BlockingQueue<std::shared_ptr<ScheduleOutput>>> schedule_output_queue);
  Status DispatchScheduleOutput(std::shared_ptr<ScheduleOutput> schedule_output);
  Status DispatchScheduleOutput(Packet* serialized_packet);

  // Executor to engine.
  Status SetSamplingOutputRecvQueue(
      std::shared_ptr<BlockingQueue<std::shared_ptr<SamplingOutput>>> sampling_output_queue);
  Status ReportSamplingOutput(std::shared_ptr<SamplingOutput> sampling_output);

  // Executor to engine - 报告层进度
  Status SetLayerProgressRecvQueue(std::shared_ptr<BlockingQueue<LayerProgressInfo>> layer_progress_queue);
  Status ReportLayerProgress(int device_id, int layer_index);

  // Engine → Executor: 向指定 executor 发送全局 cache 查询请求（异步，不阻塞）
  Status DispatchGlobalCacheQuery(int executor_id, int64_t req_id, const std::vector<int>& token_ids,
                                  size_t local_cached_len, int block_size, const std::vector<int>& kv_cache_block_ids);

  // Executor → Engine: 报告全局 cache 查询完成（异步，不阻塞）
  Status ReportGlobalCacheQueryDone(int64_t req_id, int device_id, const std::vector<int>& matched_ranges, bool success,
                                    const std::vector<int>& cached_tokens = {}, bool has_cached_tokens = false);

  // 设置全局 cache 查询相关的 IPC callable 依赖
  void SetGlobalCacheQueryCallableDeps(std::shared_ptr<GlobalCacheConnectorInterface> connector,
                                       std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager,
                                       int dp_group_id, std::shared_ptr<Environment> env, int device_id,
                                       const std::string& role = "", int timeout_ms = 5000);
  void SetGlobalCacheQueryDoneCallableFetcher(std::shared_ptr<GlobalCacheFetcher> fetcher);

  // Executor → Engine: 报告单个请求的 KV cache put 完成（异步，不阻塞）
  Status ReportGlobalCachePutDone(int device_id, int64_t req_id);

  // 设置 GlobalCachePutDone callable 的 tracker 依赖
  void SetGlobalCachePutDoneCallableTracker(std::shared_ptr<GlobalCachePutTracker> tracker);

  // pd_v2: engine queries executor[rank] for its Mooncake-registered KV
  // pool base VA + shape. Wired by MooncakeEngineConnector::GetKvBufferVa.
  Status PdV2GetKvPoolBaseVa(int rank, uint64_t* kv_pool_base_va, uint64_t* total_bytes, uint64_t* block_size,
                             uint64_t* num_blocks, uint64_t* num_layers);
  // Bind the executor-side answerer (called by the pd_v2 executor
  // connector during its SetIpcFuncWrapper override). Engine-side
  // process must NOT call this.
  void SetPdV2KvPoolQueryFn(std::function<bool(uint64_t*, uint64_t*, uint64_t*, uint64_t*, uint64_t*)> fn);

  // pd_v2: executor → engine async barrier-done report. Status: 0 = kOk,
  // 1 = kFailed (any in-flight batch terminated FAILED/TIMEOUT/CANCELED).
  // Called by PdV2KvBarrier post-forward when all batches have drained.
  // Fire-and-forget (is_async=true, is_oneway=true).
  Status ReportPdV2BarrierDone(int64_t req_id, uint8_t status);

  // Engine-side: register the callback that PdV2BarrierDoneCallable
  // invokes when an executor reports barrier-done. Wired in
  // inference_engine.cpp from PdV2PrefillIntakeWrapper.
  void SetPdV2BarrierDoneCallback(std::function<void(int64_t /*req_id*/, uint8_t /*status*/)> cb);

#ifdef ENABLE_CUDA
  // Connector NCCL operations through IPC - Engine to executor
  // Initialize Connector's NCCL communication group on executor side
  Status ConnectorInitNcclCommGroup(const std::string& group_key, const ncclUniqueId& comm_id, int rank, int world_size,
                                    int executor_id);
  // Connector send data content (host memory) - data will be copied to GPU on executor side
  // Stream is created and managed on executor side
  Status ConnectorNcclSendData(const std::string& group_key, int peer_rank, const std::vector<uint8_t>& data,
                               DataType dtype, int executor_id);
  // Connector recv data content (host memory) - data will be copied from GPU on executor side
  // Stream is created and managed on executor side
  Status ConnectorNcclRecvData(const std::string& group_key, int peer_rank, size_t count, DataType dtype,
                               int executor_id, std::vector<uint8_t>& data);
  // Connector send blocks using block_id and offset instead of GPU pointers
  // Stream is created and managed on executor side
  Status ConnectorNcclSendBlocks(const std::string& group_key, int peer_rank, const std::vector<int>& block_ids,
                                 const std::vector<int64_t>& offsets, const std::vector<size_t>& counts, DataType dtype,
                                 int executor_id);
  // Connector recv blocks using block_id and offset instead of GPU pointers
  // Stream is created and managed on executor side
  Status ConnectorNcclRecvBlocks(const std::string& group_key, int peer_rank, const std::vector<int>& block_ids,
                                 const std::vector<int64_t>& offsets, const std::vector<size_t>& counts, DataType dtype,
                                 int executor_id);
  // Get Connector's NCCL communication group info
  Status ConnectorGetNcclCommInfo(const std::string& group_key, int executor_id, int& rank, int& world_size);
#endif

 public:
  // Wait until all ipc executors are ready.
  Status WaitAllExecutorsReady(std::vector<int64_t>& executor_pids);

  // Report current executor to engine.
  Status AddExecutor();

  // Get IPC function proxy for accessing callables
  std::shared_ptr<IpcFuncProxy> GetIpcFuncProxy() { return ipc_func_proxy_; }

  // Stop all channels and join threads without destroying; use before waitpid in
  // tests so the child can exit while mutexes are still valid.
  void Shutdown();

 private:
  void SetIpcFunctions();

 private:
  int executor_id_ = -1;
  size_t executor_num_ = 0;

  // The func proxy.
  std::shared_ptr<IpcFuncProxy> ipc_func_proxy_ = nullptr;
};

}  // namespace ksana_llm
