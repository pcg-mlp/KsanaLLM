/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/environment.h"
#include "distributed/control_message.h"
#include "distributed/distributed_control_coordinator.h"
#include "distributed/distributed_data_coordinator.h"
#include "executor/executor_runtime.h"
#include "ipc/core/ipc_func_proxy.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/functions/batch_swap.h"
#include "ipc/functions/report_layer_progress.h"
#include "runtime/sampling_output.h"
#include "runtime/schedule_output.h"
#include "utils/status.h"

namespace ksana_llm {

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

  IPC_FUNC_BATCH_SWAP,
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
  Status CreateBlockAllocatorManager(const BlockAllocatorManagerConfig& block_allocator_manager_config);

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

  // Send BatchSwap request to executor and wait for completion (engine side, blocking)
  Status BatchSwap(int executor_id, bool is_swapin, const std::vector<int>& device_block_ids,
                   const std::vector<int>& host_block_ids);

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
