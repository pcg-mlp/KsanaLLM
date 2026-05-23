/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/ipc_func_wrapper.h"

#include <unistd.h>

#include "batch_scheduler/global_cache_fetcher.h"
#include "ipc/core/ipc_func_types.h"
#include "profiler/profile_event.h"
#include "ipc/functions/add_executor.h"
#include "ipc/functions/batch_swap.h"
#include "ipc/functions/connect_nccl_data_channel.h"
#include "ipc/functions/connect_tcp_data_channel.h"
#include "ipc/functions/create_block_manager.h"
#include "ipc/functions/create_executor_runtime.h"
#include "ipc/functions/dispatch_schdule_output.h"
#include "ipc/functions/get_device_memory.h"
#include "ipc/functions/get_ep_communication_config.h"
#include "ipc/functions/global_cache_put_done.h"
#include "ipc/functions/global_cache_query.h"
#include "ipc/functions/global_cache_query_done.h"
#include "ipc/functions/init_custom_communication.h"
#include "ipc/functions/init_device_communication.h"
#include "ipc/functions/init_multicast_communication.h"
#include "ipc/functions/init_reporter_channel.h"
#include "ipc/functions/listen_tcp_data_channel.h"
#include "ipc/functions/load_model.h"
#include "ipc/functions/load_weight.h"
#include "ipc/functions/nccl_operations.h"
#include "ipc/functions/report_layer_progress.h"
#include "ipc/functions/report_sampling_output.h"
#include "ipc/functions/reset_pipeline_config.h"
#include "ipc/functions/set_ep_communication_config.h"
#include "utils/random_generator.h"
#include "utils/status.h"

namespace ksana_llm {

IpcFuncWrapper::IpcFuncWrapper(size_t executor_num, bool is_engine, int executor_id) {
  executor_id_ = executor_id;
  executor_num_ = executor_num;
  ipc_func_proxy_ = std::make_shared<IpcFuncProxy>(executor_num, is_engine, executor_id);
  SetIpcFunctions();
}

IpcFuncWrapper::~IpcFuncWrapper() {
  if (ipc_func_proxy_) {
    ipc_func_proxy_->Destroy();
  }
}

void IpcFuncWrapper::SetIpcFunctions() {
  // add ipc executors.
  ipc_func_proxy_->SetFuncInfo<AddExecutorCallable, AddExecutorRequest, AddExecutorResponse, AddExecutorCallback>(
      IPC_FUNC_ADD_EXECUTOR);

  // broadcast init device communications.
  ipc_func_proxy_->SetFuncInfo<InitDeviceCommunicationCallable, InitDeviceCommunicationRequest,
                               InitDeviceCommunicationResponse, InitDeviceCommunicationCallback>(
      IPC_FUNC_INIT_DEVICE_COMMUNICATION);

  // broadcast init custom communications.
  ipc_func_proxy_->SetFuncInfo<InitCustomCommunicationCallable, InitCustomCommunicationRequest,
                               InitCustomCommunicationResponse, InitCustomCommunicationCallback>(
      IPC_FUNC_INIT_CUSTOM_COMMUNICATION);

  // broadcast init mutlicast communications.
  ipc_func_proxy_->SetFuncInfo<InitMulticastCommunicationCallable, InitMulticastCommunicationRequest,
                               InitMulticastCommunicationResponse, InitMulticastCommunicationCallback>(
      IPC_FUNC_INIT_MULTICAST_COMMUNICATION);

  // get device info.
  ipc_func_proxy_
      ->SetFuncInfo<GetDeviceMemoryCallable, GetDeviceMemoryRequest, GetDeviceMemoryResponse, GetDeviceMemoryCallback>(
          IPC_FUNC_GET_DEVICE_MEMORY);

  // tcp data channel
  ipc_func_proxy_->SetFuncInfo<ListenTcpDataChannelCallable, ListenTcpDataChannelRequest, ListenTcpDataChannelResponse,
                               ListenTcpDataChannelCallback>(IPC_FUNC_LISTEN_TCP_DATA_CHANNEL);
  ipc_func_proxy_->SetFuncInfo<ConnectTcpDataChannelCallable, ConnectTcpDataChannelRequest,
                               ConnectTcpDataChannelResponse, ConnectTcpDataChannelCallback>(
      IPC_FUNC_CONNECT_TCP_DATA_CHANNEL);

  // nccl data channel
  ipc_func_proxy_->SetFuncInfo<ConnectNcclDataChannelCallable, ConnectNcclDataChannelRequest,
                               ConnectNcclDataChannelResponse, ConnectNcclDataChannelCallback>(
      IPC_FUNC_CONNECT_NCCL_DATA_CHANNEL);

  // result reporter channel
  ipc_func_proxy_->SetFuncInfo<InitReporterChannelCallable, InitReporterChannelRequest, InitReporterChannelResponse,
                               InitReporterChannelCallback>(IPC_FUNC_INIT_REPORTER_CHANNEL);

  // broadcast reset pipeline config.
  ipc_func_proxy_->SetFuncInfo<ResetPipelineConfigCallable, ResetPipelineConfigRequest, ResetPipelineConfigResponse,
                               ResetPipelineConfigCallback>(IPC_FUNC_RESET_PIPELINE_CONFIG);

  // broadcast load weights.
  ipc_func_proxy_->SetFuncInfo<LoadWeightCallable, LoadWeightRequest, LoadWeightResponse, LoadWeightCallback>(
      IPC_FUNC_LOAD_WEIGHT);

  // broadcast load model.
  ipc_func_proxy_->SetFuncInfo<LoadModelCallable, LoadModelRequest, LoadModelResponse, LoadModelCallback>(
      IPC_FUNC_LOAD_MODEL);

  // broadcast create block allocator manager.
  ipc_func_proxy_->SetFuncInfo<CreateBlockManagerCallable, CreateBlockManagerRequest, CreateBlockManagerResponse,
                               CreateBlockManagerCallback>(IPC_FUNC_CREATE_BLOCK_ALLOCATOR);

  // broadcast create llm runtime.
  ipc_func_proxy_->SetFuncInfo<CreateExecutorRuntimeCallable, CreateExecutorRuntimeRequest,
                               CreateExecutorRuntimeResponse, CreateExecutorRuntimeCallback>(
      IPC_FUNC_CREATE_EXECUTOR_RUNTIME);

  // broadcast schedule output.
  ipc_func_proxy_->SetFuncInfo<DispatchScheduleOutputCallable, DispatchScheduleOutputRequest,
                               DispatchScheduleOutputResponse, DispatchScheduleOutputCallback>(
      IPC_FUNC_DISPATCH_SCHEDULE_OUTPUT);

  // report sampling output.
  ipc_func_proxy_->SetFuncInfo<ReportSamplingOutputCallable, ReportSamplingOutputRequest, ReportSamplingOutputResponse,
                               ReportSamplingOutputCallback>(IPC_FUNC_REPORT_SAMPLING_OUTPUT);

  // report layer progress.
  ipc_func_proxy_->SetFuncInfo<ReportLayerProgressCallable, ReportLayerProgressRequest, ReportLayerProgressResponse,
                               ReportLayerProgressCallback>(IPC_FUNC_REPORT_LAYER_PROGRESS);

  // batch swap: engine to executor, executor executes swap operation.
  ipc_func_proxy_->SetFuncInfo<BatchSwapCallable, BatchSwapRequest, BatchSwapResponse, BatchSwapCallback>(
      IPC_FUNC_BATCH_SWAP);

  // 全局 cache 查询: engine 广播到 executor（异步）
  ipc_func_proxy_->SetFuncInfo<GlobalCacheQueryCallable, GlobalCacheQueryRequest, GlobalCacheQueryResponse,
                               GlobalCacheQueryCallback>(IPC_FUNC_GLOBAL_CACHE_QUERY);

  // 全局 cache 查询完成: executor 报告给 engine（异步）
  ipc_func_proxy_->SetFuncInfo<GlobalCacheQueryDoneCallable, GlobalCacheQueryDoneRequest, GlobalCacheQueryDoneResponse,
                               GlobalCacheQueryDoneCallback>(IPC_FUNC_GLOBAL_CACHE_QUERY_DONE);

  // KV cache put 完成: executor 报告给 engine（异步）
  ipc_func_proxy_->SetFuncInfo<GlobalCachePutDoneCallable, GlobalCachePutDoneRequest, GlobalCachePutDoneResponse,
                               GlobalCachePutDoneCallback>(IPC_FUNC_GLOBAL_CACHE_PUT_DONE);

#ifdef ENABLE_CUDA
  // Connector NCCL operations
  ipc_func_proxy_->SetFuncInfo<ConnectorInitNcclCommGroupCallable, ConnectorInitNcclCommGroupRequest,
                               ConnectorInitNcclCommGroupResponse, ConnectorInitNcclCommGroupCallback>(
      IPC_FUNC_CONNECTOR_INIT_NCCL_COMM_GROUP);

  ipc_func_proxy_->SetFuncInfo<ConnectorNcclSendDataCallable, ConnectorNcclSendDataRequest,
                               ConnectorNcclSendDataResponse, ConnectorNcclSendDataCallback>(
      IPC_FUNC_CONNECTOR_NCCL_SEND_DATA);

  ipc_func_proxy_->SetFuncInfo<ConnectorNcclRecvDataCallable, ConnectorNcclRecvDataRequest,
                               ConnectorNcclRecvDataResponse, ConnectorNcclRecvDataCallback>(
      IPC_FUNC_CONNECTOR_NCCL_RECV_DATA);

  ipc_func_proxy_->SetFuncInfo<ConnectorNcclSendBlocksCallable, ConnectorNcclSendBlocksRequest,
                               ConnectorNcclSendBlocksResponse, ConnectorNcclSendBlocksCallback>(
      IPC_FUNC_CONNECTOR_NCCL_SEND_BLOCKS);

  ipc_func_proxy_->SetFuncInfo<ConnectorNcclRecvBlocksCallable, ConnectorNcclRecvBlocksRequest,
                               ConnectorNcclRecvBlocksResponse, ConnectorNcclRecvBlocksCallback>(
      IPC_FUNC_CONNECTOR_NCCL_RECV_BLOCKS);

  ipc_func_proxy_->SetFuncInfo<ConnectorGetNcclCommInfoCallable, ConnectorGetNcclCommInfoRequest,
                               ConnectorGetNcclCommInfoResponse, ConnectorGetNcclCommInfoCallback>(
      IPC_FUNC_CONNECTOR_GET_NCCL_COMM_INFO);

  // get EP communicator config
  ipc_func_proxy_->SetFuncInfo<GetEPCommunicationConfigCallable, GetEPCommunicationConfigRequest,
                               GetEPCommunicationConfigResponse, GetEPCommunicationConfigCallback>(
      IPC_FUNC_GET_EP_COMMUNICATION_CONFIG);

  // set EP communicator config
  ipc_func_proxy_->SetFuncInfo<SetEPCommunicationConfigCallable, SetEPCommunicationConfigRequest,
                               SetEPCommunicationConfigResponse, SetEPCommunicationConfigCallback>(
      IPC_FUNC_SET_EP_COMMUNICATION_CONFIG);
#endif
}

void IpcFuncWrapper::Shutdown() {
  if (ipc_func_proxy_) {
    ipc_func_proxy_->Destroy();
  }
}

Status IpcFuncWrapper::WaitAllExecutorsReady(std::vector<int64_t>& executor_pids) {
  auto callable = ipc_func_proxy_->GetSharedIpcFuncCallable<AddExecutorCallable>(IPC_FUNC_ADD_EXECUTOR);
  callable->Reset(executor_num_);
  callable->Wait();
  executor_pids = callable->executor_pids;

  // Make sure ipc resource removed if process crashed.
  ipc_func_proxy_->UnlinkIpcResource();
  return Status();
}

Status IpcFuncWrapper::AddExecutor() {
  std::shared_ptr<IpcContext> ipc_context =
      CreateExecutor2EngineContext(IPC_FUNC_ADD_EXECUTOR, executor_id_, true, true, true, true);
  std::shared_ptr<AddExecutorRequest> request = std::make_shared<AddExecutorRequest>();
  request->executor_pid = getpid();
  std::shared_ptr<AddExecutorResponse> response = std::make_shared<AddExecutorResponse>();
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), response.get(), ipc_context.get()));

  return Status();
}

Status IpcFuncWrapper::SetDeviceCommunicationContext(std::shared_ptr<Context> context) {
  auto callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<InitDeviceCommunicationCallable>(IPC_FUNC_INIT_DEVICE_COMMUNICATION);
  callable->SetContext(context);
  return Status();
}

Status IpcFuncWrapper::InitDeviceCommunication() {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<InitDeviceCommunicationCallback>(IPC_FUNC_INIT_DEVICE_COMMUNICATION);
  callback->Reset(executor_num_);

  std::shared_ptr<IpcContext> ipc_context =
      CreateBroadcastContext(IPC_FUNC_INIT_DEVICE_COMMUNICATION, true, false, true, true);
  std::shared_ptr<InitDeviceCommunicationRequest> request = std::make_shared<InitDeviceCommunicationRequest>();
  request->nccl_uid = GenerateNCCLUniqueID();
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));

  callback->Wait();
  return Status();
}

Status IpcFuncWrapper::SetCustomCommunicationContext(std::shared_ptr<Context> context) {
  auto callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<InitCustomCommunicationCallable>(IPC_FUNC_INIT_CUSTOM_COMMUNICATION);
  callable->SetContext(context);
  return Status();
}

Status IpcFuncWrapper::InitCustomCommunication() {
#ifdef ENABLE_CUDA
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<InitCustomCommunicationCallback>(IPC_FUNC_INIT_CUSTOM_COMMUNICATION);
  std::array<std::vector<cudaIpcMemHandle_t>, kCustomIpcHandleCount> ipc_handles;
  for (auto& ipc_handle : ipc_handles) {
    ipc_handle.resize(executor_num_);
  }
  callback->SetCustomCommunicationIpcHandles(ipc_handles.data());

  std::shared_ptr<IpcContext> ipc_context =
      CreateBroadcastContext(IPC_FUNC_INIT_CUSTOM_COMMUNICATION, true, false, true, true);
  std::shared_ptr<InitCustomCommunicationRequest> request = std::make_shared<InitCustomCommunicationRequest>();

  // First call: the executor sends its own IPC handle to the engine
  callback->Reset(executor_num_);
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  callback->Wait();

  // Second call: the engine sends all IPC handles to each executor
  callback->Reset(executor_num_);
  request->ipc_handles = std::move(ipc_handles);
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  callback->Wait();
#endif
  return Status();
}

Status IpcFuncWrapper::SetMulticastCommunicationContext(std::shared_ptr<Context> context) {
  auto callable = ipc_func_proxy_->GetSharedIpcFuncCallable<InitMulticastCommunicationCallable>(
      IPC_FUNC_INIT_MULTICAST_COMMUNICATION);
  callable->SetContext(context);
  return Status();
}

Status IpcFuncWrapper::InitMulticastCommunication() {
  auto callback = ipc_func_proxy_->GetSharedIpcFuncCallback<InitMulticastCommunicationCallback>(
      IPC_FUNC_INIT_MULTICAST_COMMUNICATION);

  std::shared_ptr<IpcContext> ipc_context =
      CreateBroadcastContext(IPC_FUNC_INIT_MULTICAST_COMMUNICATION, true, false, true, true);
  std::shared_ptr<InitMulticastCommunicationRequest> request = std::make_shared<InitMulticastCommunicationRequest>();

  static RandomGenerator<uint64_t> gen;
  // Mask the sign bit to ensure the value is always positive (gen() returns uint64_t,
  // but unique_op_id is int64_t — without masking, 50% of values wrap negative and
  // skip socket initialization, causing null-pointer crashes in the recv branch).
  request->unique_op_id = static_cast<int64_t>(gen() & INT64_MAX);
  callback->Reset(executor_num_);
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  callback->Wait();

  request->unique_op_id = -1;
  const int node_device_num = executor_num_;
  for (int broadcast_id = 0; broadcast_id <= node_device_num; broadcast_id++) {
    // Executor `broadcast_id`: Sends the IPC handle file descriptor
    request->broadcast_id = broadcast_id;
    callback->Reset(executor_num_);
    STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
    callback->Wait();

    // Other executors: receive the IPC handle file descriptor
    request->broadcast_id = -broadcast_id - 1;
    callback->Reset(executor_num_);
    STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
    callback->Wait();
  }

  // Initialize multicast memory and close the IPC socket
  request->broadcast_id = node_device_num + 1;
  callback->Reset(executor_num_);
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  callback->Wait();

  return Status();
}

Status IpcFuncWrapper::GetDeviceMemory(size_t* device_free, size_t* device_total) {
  // Get from device 0.
  constexpr int executor_id = 0;
  std::shared_ptr<IpcContext> ipc_context =
      CreateEngine2ExecutorContext(IPC_FUNC_GET_DEVICE_MEMORY, executor_id, false, false, false, false);
  std::shared_ptr<GetDeviceMemoryRequest> request = std::make_shared<GetDeviceMemoryRequest>();
  std::shared_ptr<GetDeviceMemoryResponse> response = std::make_shared<GetDeviceMemoryResponse>();
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), response.get(), ipc_context.get()));
  *device_free = response->device_free;
  *device_total = response->device_total;
  return Status();
}

Status IpcFuncWrapper::ResetPipelineConfig(const PipelineConfig& pipeline_config) {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<ResetPipelineConfigCallback>(IPC_FUNC_RESET_PIPELINE_CONFIG);
  callback->Reset(executor_num_);

  std::shared_ptr<IpcContext> ipc_context =
      CreateBroadcastContext(IPC_FUNC_RESET_PIPELINE_CONFIG, true, false, true, true);
  std::shared_ptr<ResetPipelineConfigRequest> request = std::make_shared<ResetPipelineConfigRequest>();
  request->pipeline_config = pipeline_config;
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));

  callback->Wait();
  return Status();
}

void IpcFuncWrapper::SetTcpDistributedDataCoordinator(
    std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator) {
  auto callable_listen =
      ipc_func_proxy_->GetSharedIpcFuncCallable<ListenTcpDataChannelCallable>(IPC_FUNC_LISTEN_TCP_DATA_CHANNEL);
  callable_listen->SetDistributedDataCoordinator(distributed_data_coordinator);

  auto callable_connect =
      ipc_func_proxy_->GetSharedIpcFuncCallable<ConnectTcpDataChannelCallable>(IPC_FUNC_CONNECT_TCP_DATA_CHANNEL);
  callable_connect->SetDistributedDataCoordinator(distributed_data_coordinator);
}

Status IpcFuncWrapper::ListenTcpDataChannel(std::vector<HostPort>& executor_host_ports) {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<ListenTcpDataChannelCallback>(IPC_FUNC_LISTEN_TCP_DATA_CHANNEL);
  callback->Reset(executor_num_);
  callback->executor_host_ports.resize(executor_num_);

  std::shared_ptr<IpcContext> ipc_context =
      CreateBroadcastContext(IPC_FUNC_LISTEN_TCP_DATA_CHANNEL, true, false, true, true);
  std::shared_ptr<ListenTcpDataChannelRequest> request = std::make_shared<ListenTcpDataChannelRequest>();
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));

  callback->Wait();
  executor_host_ports = callback->executor_host_ports;

  return Status();
}

Status IpcFuncWrapper::ConnectTcpDataChannel(const std::vector<HostPort>& executor_downstream_host_ports) {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<ConnectTcpDataChannelCallback>(IPC_FUNC_CONNECT_TCP_DATA_CHANNEL);
  callback->Reset(executor_num_);

  for (size_t executor_id = 0; executor_id < executor_num_; ++executor_id) {
    std::shared_ptr<IpcContext> ipc_context =
        CreateEngine2ExecutorContext(IPC_FUNC_CONNECT_TCP_DATA_CHANNEL, executor_id, true, false, true, true);
    std::shared_ptr<ConnectTcpDataChannelRequest> request = std::make_shared<ConnectTcpDataChannelRequest>();
    request->executor_downstream_host_port = executor_downstream_host_ports[executor_id];

    STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  }

  callback->Wait();
  return Status();
}

#ifdef ENABLE_CUDA
void IpcFuncWrapper::SetNcclDistributedDataCoordinator(
    std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator) {
  auto callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<ConnectNcclDataChannelCallable>(IPC_FUNC_CONNECT_NCCL_DATA_CHANNEL);
  callable->SetDistributedDataCoordinator(distributed_data_coordinator);
}
#endif

#ifdef ENABLE_CUDA
Status IpcFuncWrapper::ConnectNcclDataChannel(ncclUniqueId nccl_unique_id) {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<ConnectNcclDataChannelCallback>(IPC_FUNC_CONNECT_NCCL_DATA_CHANNEL);
  callback->Reset(executor_num_);

  std::shared_ptr<IpcContext> ipc_context =
      CreateBroadcastContext(IPC_FUNC_CONNECT_NCCL_DATA_CHANNEL, true, false, true, true);
  std::shared_ptr<ConnectNcclDataChannelRequest> request = std::make_shared<ConnectNcclDataChannelRequest>();
  request->nccl_unique_id = nccl_unique_id;
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));

  callback->Wait();
  return Status();
}
#endif

void IpcFuncWrapper::SetReporterDistributedControlCoordinator(
    std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator) {
  auto callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<InitReporterChannelCallable>(IPC_FUNC_INIT_REPORTER_CHANNEL);
  callable->SetDistributedControlCoordinator(distributed_control_coordinator);
}

Status IpcFuncWrapper::ConnectResultReporterChannel() {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<InitReporterChannelCallback>(IPC_FUNC_INIT_REPORTER_CHANNEL);
  callback->Reset(1);

  std::shared_ptr<IpcContext> ipc_context =
      CreateEngine2ExecutorContext(IPC_FUNC_INIT_REPORTER_CHANNEL, 0, true, false, true, true);
  std::shared_ptr<InitReporterChannelRequest> request = std::make_shared<InitReporterChannelRequest>();
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));

  callback->Wait();
  return callback->status;
}

Status IpcFuncWrapper::GetEPCommunicationConfig(std::vector<EPCommunicationConfig>& executor_comm_configs) {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<GetEPCommunicationConfigCallback>(IPC_FUNC_GET_EP_COMMUNICATION_CONFIG);
  callback->Reset(executor_num_);
  callback->executor_comm_configs.resize(executor_num_);

  // Call each executor individually instead of broadcast, so each has correct to_rank
  for (size_t executor_id = 0; executor_id < executor_num_; ++executor_id) {
    std::shared_ptr<IpcContext> ipc_context =
        CreateEngine2ExecutorContext(IPC_FUNC_GET_EP_COMMUNICATION_CONFIG, executor_id, true, false, true, true);
    std::shared_ptr<GetEPCommunicationConfigRequest> request = std::make_shared<GetEPCommunicationConfigRequest>();
    STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  }

  callback->Wait();
  executor_comm_configs = callback->executor_comm_configs;
  return Status();
}

Status IpcFuncWrapper::SetEPCommunicationConfig(const std::vector<EPCommunicationConfig>& aggregated_comm_configs) {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<SetEPCommunicationConfigCallback>(IPC_FUNC_SET_EP_COMMUNICATION_CONFIG);
  callback->Reset(executor_num_);

  for (size_t executor_id = 0; executor_id < executor_num_; ++executor_id) {
    std::shared_ptr<IpcContext> ipc_context =
        CreateEngine2ExecutorContext(IPC_FUNC_SET_EP_COMMUNICATION_CONFIG, executor_id, true, false, true, true);
    std::shared_ptr<SetEPCommunicationConfigRequest> request = std::make_shared<SetEPCommunicationConfigRequest>();
    request->aggregated_comm_configs = aggregated_comm_configs;

    STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  }

  callback->Wait();
  return Status();
}

void IpcFuncWrapper::SetBlockCacheLayout(std::shared_ptr<BaseModelCacheLayout> block_cache_layout) {
  auto pipeline_callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<ResetPipelineConfigCallable>(IPC_FUNC_RESET_PIPELINE_CONFIG);
  pipeline_callable->SetBlockCacheLayout(block_cache_layout);
}

Status IpcFuncWrapper::SetWeightInstance(std::shared_ptr<WeightInstanceInterface> model_weight) {
  auto callable = ipc_func_proxy_->GetSharedIpcFuncCallable<LoadWeightCallable>(IPC_FUNC_LOAD_WEIGHT);
  callable->SetWeightInstance(model_weight);
  return Status();
}

Status IpcFuncWrapper::LoadWeight() {
  auto callback = ipc_func_proxy_->GetSharedIpcFuncCallback<LoadWeightCallback>(IPC_FUNC_LOAD_WEIGHT);
  callback->Reset(executor_num_);

  std::shared_ptr<IpcContext> ipc_context = CreateBroadcastContext(IPC_FUNC_LOAD_WEIGHT, true, false, true, true);
  std::shared_ptr<LoadWeightRequest> request = std::make_shared<LoadWeightRequest>();
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));

  callback->Wait();
  return callback->load_status;
}

Status IpcFuncWrapper::SetModelInstance(std::shared_ptr<ModelInstance> model_instance) {
  auto callable = ipc_func_proxy_->GetSharedIpcFuncCallable<LoadModelCallable>(IPC_FUNC_LOAD_MODEL);
  callable->SetModelInstance(model_instance);

  // Also set model_instance for GetEPCommunicationConfigCallable
  auto get_ep_comm_callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<GetEPCommunicationConfigCallable>(IPC_FUNC_GET_EP_COMMUNICATION_CONFIG);
  get_ep_comm_callable->SetModelInstance(model_instance);

  // Also set model_instance for SetEPCommunicationConfigCallable
  auto set_ep_comm_callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<SetEPCommunicationConfigCallable>(IPC_FUNC_SET_EP_COMMUNICATION_CONFIG);
  set_ep_comm_callable->SetModelInstance(model_instance);

  return Status();
}

Status IpcFuncWrapper::SetEnvironment(std::shared_ptr<Environment> env) {
  // Set environment for ResetPipelineConfigCallable
  auto pipeline_callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<ResetPipelineConfigCallable>(IPC_FUNC_RESET_PIPELINE_CONFIG);
  pipeline_callable->SetEnvironment(env);

  // Set environment for GetEPCommunicationConfigCallable
  auto ep_comm_callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<GetEPCommunicationConfigCallable>(IPC_FUNC_GET_EP_COMMUNICATION_CONFIG);
  ep_comm_callable->SetEnvironment(env);

  return Status();
}

Status IpcFuncWrapper::LoadModel() {
  auto callback = ipc_func_proxy_->GetSharedIpcFuncCallback<LoadModelCallback>(IPC_FUNC_LOAD_MODEL);
  callback->Reset(executor_num_);

  std::shared_ptr<IpcContext> ipc_context = CreateBroadcastContext(IPC_FUNC_LOAD_MODEL, true, false, true, true);
  std::shared_ptr<LoadModelRequest> request = std::make_shared<LoadModelRequest>();
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));

  callback->Wait();
  return callback->load_status;
}

Status IpcFuncWrapper::SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager,
                                                int dp_group_id) {
  auto block_mgr_callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<CreateBlockManagerCallable>(IPC_FUNC_CREATE_BLOCK_ALLOCATOR);
  block_mgr_callable->SetBlockAllocatorManager(block_allocator_manager);

  // Set on executor side - the callable that executes the swap
  auto swap_callable = ipc_func_proxy_->GetSharedIpcFuncCallable<BatchSwapCallable>(IPC_FUNC_BATCH_SWAP);
  swap_callable->SetBlockAllocatorManager(block_allocator_manager, dp_group_id);
  return Status();
}

Status IpcFuncWrapper::SetSwaBlockAllocatorManager(
    std::shared_ptr<BlockAllocatorManagerInterface> swa_block_allocator_manager) {
  auto block_mgr_callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<CreateBlockManagerCallable>(IPC_FUNC_CREATE_BLOCK_ALLOCATOR);
  block_mgr_callable->SetSwaBlockAllocatorManager(swa_block_allocator_manager);
  return Status();
}

Status IpcFuncWrapper::CreateBlockAllocatorManager(
    const BlockAllocatorManagerConfig& block_allocator_manager_config,
    const BlockAllocatorManagerConfig& swa_block_allocator_manager_config) {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<CreateBlockManagerCallback>(IPC_FUNC_CREATE_BLOCK_ALLOCATOR);
  callback->Reset(executor_num_);

  std::shared_ptr<IpcContext> ipc_context =
      CreateBroadcastContext(IPC_FUNC_CREATE_BLOCK_ALLOCATOR, true, false, true, true);
  std::shared_ptr<CreateBlockManagerRequest> request = std::make_shared<CreateBlockManagerRequest>();
  request->block_allocator_manager_config = block_allocator_manager_config;
  request->swa_block_allocator_manager_config = swa_block_allocator_manager_config;
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));

  callback->Wait();
  return callback->load_status;
}

Status IpcFuncWrapper::SetExecutorRuntime(std::shared_ptr<ExecutorRuntime> executor_runtime) {
  auto callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<CreateExecutorRuntimeCallable>(IPC_FUNC_CREATE_EXECUTOR_RUNTIME);
  callable->SetExecutorRuntime(executor_runtime);
  return Status();
}

Status IpcFuncWrapper::SetExecutorEnvironment(std::shared_ptr<Environment> env) {
  auto callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<CreateExecutorRuntimeCallable>(IPC_FUNC_CREATE_EXECUTOR_RUNTIME);
  callable->SetEnvironment(env);
  return Status();
}

Status IpcFuncWrapper::CreateExecutorRuntime(const VocabInfo& vocab_info) {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<CreateExecutorRuntimeCallback>(IPC_FUNC_CREATE_EXECUTOR_RUNTIME);
  callback->Reset(executor_num_);

  std::shared_ptr<IpcContext> ipc_context =
      CreateBroadcastContext(IPC_FUNC_CREATE_EXECUTOR_RUNTIME, true, false, true, true);
  std::shared_ptr<CreateExecutorRuntimeRequest> request = std::make_shared<CreateExecutorRuntimeRequest>();
  request->vocab_info = vocab_info;
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));

  callback->Wait();
  return callback->create_status;
}

Status IpcFuncWrapper::SetScheduleOutputRecvQueue(
    std::shared_ptr<BlockingQueue<std::shared_ptr<ScheduleOutput>>> schedule_output_queue) {
  auto callable =
      ipc_func_proxy_->GetSharedIpcFuncCallable<DispatchScheduleOutputCallable>(IPC_FUNC_DISPATCH_SCHEDULE_OUTPUT);
  callable->SetRecvQueue(schedule_output_queue);
  return Status();
}

Status IpcFuncWrapper::DispatchScheduleOutput(std::shared_ptr<ScheduleOutput> schedule_output) {
  PROFILE_EVENT_SCOPE(dispatch_schedule_output, "Ipc.DispatchScheduleOutput");
  std::shared_ptr<IpcContext> ipc_context =
      CreateBroadcastContext(IPC_FUNC_DISPATCH_SCHEDULE_OUTPUT, true, true, true, false);
  std::shared_ptr<DispatchScheduleOutputRequest> request = std::make_shared<DispatchScheduleOutputRequest>();
  request->schedule_output = schedule_output;
  request->serialized_packet = nullptr;

  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  return Status();
}

Status IpcFuncWrapper::DispatchScheduleOutput(Packet* serialized_packet) {
  std::shared_ptr<IpcContext> ipc_context =
      CreateBroadcastContext(IPC_FUNC_DISPATCH_SCHEDULE_OUTPUT, true, true, true, false);
  std::shared_ptr<DispatchScheduleOutputRequest> request = std::make_shared<DispatchScheduleOutputRequest>();
  request->schedule_output = nullptr;
  request->serialized_packet = serialized_packet;

  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  return Status();
}

Status IpcFuncWrapper::SetSamplingOutputRecvQueue(
    std::shared_ptr<BlockingQueue<std::shared_ptr<SamplingOutput>>> sampling_output_queue) {
  // Make sure any callee could process it without lock.
  for (size_t executor_id = 0; executor_id < executor_num_; ++executor_id) {
    auto callable =
        ipc_func_proxy_->GetIpcFuncCallable<ReportSamplingOutputCallable>(IPC_FUNC_REPORT_SAMPLING_OUTPUT, executor_id);
    callable->SetRecvQueue(sampling_output_queue);
  }

  return Status();
}

Status IpcFuncWrapper::ReportSamplingOutput(std::shared_ptr<SamplingOutput> sampling_output) {
  PROFILE_EVENT_SCOPE(report_sampling_output, "Ipc.ReportSamplingOutput");
  std::shared_ptr<IpcContext> ipc_context =
      CreateExecutor2EngineContext(IPC_FUNC_REPORT_SAMPLING_OUTPUT, executor_id_, true, true, false, false);
  std::shared_ptr<ReportSamplingOutputRequest> request = std::make_shared<ReportSamplingOutputRequest>();
  request->sampling_output = sampling_output;
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  return Status();
}

Status IpcFuncWrapper::SetLayerProgressRecvQueue(
    std::shared_ptr<BlockingQueue<LayerProgressInfo>> layer_progress_queue) {
  // Make sure any callee could process it without lock.
  for (size_t executor_id = 0; executor_id < executor_num_; ++executor_id) {
    auto callable =
        ipc_func_proxy_->GetIpcFuncCallable<ReportLayerProgressCallable>(IPC_FUNC_REPORT_LAYER_PROGRESS, executor_id);
    callable->SetRecvQueue(layer_progress_queue);
  }

  return Status();
}

Status IpcFuncWrapper::ReportLayerProgress(int device_id, int layer_index) {
  std::shared_ptr<IpcContext> ipc_context =
      CreateExecutor2EngineContext(IPC_FUNC_REPORT_LAYER_PROGRESS, executor_id_, true, true, false, false);
  std::shared_ptr<ReportLayerProgressRequest> request = std::make_shared<ReportLayerProgressRequest>();
  request->device_id = device_id;
  request->layer_index = layer_index;
  KLLM_LOG_DEBUG << "ReportLayerProgress, device_id: " << device_id << ", layer_index: " << layer_index;
  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  return Status();
}

#ifdef ENABLE_CUDA
Status IpcFuncWrapper::ConnectorInitNcclCommGroup(const std::string& group_key, const ncclUniqueId& comm_id, int rank,
                                                  int world_size, int executor_id) {
  auto callback = ipc_func_proxy_->GetSharedIpcFuncCallback<ConnectorInitNcclCommGroupCallback>(
      IPC_FUNC_CONNECTOR_INIT_NCCL_COMM_GROUP);
  callback->Reset(1);

  std::shared_ptr<IpcContext> ipc_context =
      CreateEngine2ExecutorContext(IPC_FUNC_CONNECTOR_INIT_NCCL_COMM_GROUP, executor_id, true, false, true, true);
  std::shared_ptr<ConnectorInitNcclCommGroupRequest> request = std::make_shared<ConnectorInitNcclCommGroupRequest>();
  request->group_key = group_key;
  request->comm_id = comm_id;
  request->rank = rank;
  request->world_size = world_size;

  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  callback->Wait();
  return callback->status;
}

Status IpcFuncWrapper::ConnectorNcclSendData(const std::string& group_key, int peer_rank,
                                             const std::vector<uint8_t>& data, DataType dtype, int executor_id) {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<ConnectorNcclSendDataCallback>(IPC_FUNC_CONNECTOR_NCCL_SEND_DATA);
  callback->Reset(1);

  std::shared_ptr<IpcContext> ipc_context =
      CreateEngine2ExecutorContext(IPC_FUNC_CONNECTOR_NCCL_SEND_DATA, executor_id, true, false, true, true);
  std::shared_ptr<ConnectorNcclSendDataRequest> request = std::make_shared<ConnectorNcclSendDataRequest>();
  request->group_key = group_key;
  request->peer_rank = peer_rank;
  request->data = data;  // Send data content instead of pointer
  request->dtype = dtype;

  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  callback->Wait();
  return callback->status;
}

Status IpcFuncWrapper::ConnectorNcclRecvData(const std::string& group_key, int peer_rank, size_t count, DataType dtype,
                                             int executor_id, std::vector<uint8_t>& data) {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<ConnectorNcclRecvDataCallback>(IPC_FUNC_CONNECTOR_NCCL_RECV_DATA);
  callback->Reset(1);

  std::shared_ptr<IpcContext> ipc_context =
      CreateEngine2ExecutorContext(IPC_FUNC_CONNECTOR_NCCL_RECV_DATA, executor_id, true, false, true, true);
  std::shared_ptr<ConnectorNcclRecvDataRequest> request = std::make_shared<ConnectorNcclRecvDataRequest>();
  request->group_key = group_key;
  request->peer_rank = peer_rank;
  request->count = count;  // Only send count, not buffer pointer
  request->dtype = dtype;

  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  callback->Wait();

  // Get received data from callback
  if (callback->status.OK()) {
    data = callback->data;
  }
  return callback->status;
}

Status IpcFuncWrapper::ConnectorNcclSendBlocks(const std::string& group_key, int peer_rank,
                                               const std::vector<int>& block_ids, const std::vector<int64_t>& offsets,
                                               const std::vector<size_t>& counts, DataType dtype, int executor_id) {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<ConnectorNcclSendBlocksCallback>(IPC_FUNC_CONNECTOR_NCCL_SEND_BLOCKS);
  callback->Reset(1);

  std::shared_ptr<IpcContext> ipc_context =
      CreateEngine2ExecutorContext(IPC_FUNC_CONNECTOR_NCCL_SEND_BLOCKS, executor_id, true, false, true, true);
  std::shared_ptr<ConnectorNcclSendBlocksRequest> request = std::make_shared<ConnectorNcclSendBlocksRequest>();
  request->group_key = group_key;
  request->peer_rank = peer_rank;
  request->block_ids = block_ids;
  request->offsets = offsets;
  request->counts = counts;
  request->dtype = dtype;

  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  callback->Wait();
  return callback->status;
}

Status IpcFuncWrapper::ConnectorNcclRecvBlocks(const std::string& group_key, int peer_rank,
                                               const std::vector<int>& block_ids, const std::vector<int64_t>& offsets,
                                               const std::vector<size_t>& counts, DataType dtype, int executor_id) {
  auto callback =
      ipc_func_proxy_->GetSharedIpcFuncCallback<ConnectorNcclRecvBlocksCallback>(IPC_FUNC_CONNECTOR_NCCL_RECV_BLOCKS);
  callback->Reset(1);

  std::shared_ptr<IpcContext> ipc_context =
      CreateEngine2ExecutorContext(IPC_FUNC_CONNECTOR_NCCL_RECV_BLOCKS, executor_id, true, false, true, true);
  std::shared_ptr<ConnectorNcclRecvBlocksRequest> request = std::make_shared<ConnectorNcclRecvBlocksRequest>();
  request->group_key = group_key;
  request->peer_rank = peer_rank;
  request->block_ids = block_ids;
  request->offsets = offsets;
  request->counts = counts;
  request->dtype = dtype;

  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  callback->Wait();
  return callback->status;
}

Status IpcFuncWrapper::ConnectorGetNcclCommInfo(const std::string& group_key, int executor_id, int& rank,
                                                int& world_size) {
  auto callback = ipc_func_proxy_->GetSharedIpcFuncCallback<ConnectorGetNcclCommInfoCallback>(
      IPC_FUNC_CONNECTOR_GET_NCCL_COMM_INFO);
  callback->Reset(1);

  std::shared_ptr<IpcContext> ipc_context =
      CreateEngine2ExecutorContext(IPC_FUNC_CONNECTOR_GET_NCCL_COMM_INFO, executor_id, true, false, true, true);
  std::shared_ptr<ConnectorGetNcclCommInfoRequest> request = std::make_shared<ConnectorGetNcclCommInfoRequest>();
  request->group_key = group_key;

  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  callback->Wait();

  if (callback->status.OK()) {
    rank = callback->rank;
    world_size = callback->world_size;
  }
  return callback->status;
}
#endif

Status IpcFuncWrapper::BatchSwap(int executor_id, bool is_swapin, const std::vector<int>& device_block_ids,
                                 const std::vector<int>& host_block_ids) {
  std::shared_ptr<IpcContext> ipc_context =
      CreateEngine2ExecutorContext(IPC_FUNC_BATCH_SWAP, executor_id, /*is_async=*/false, /*is_oneway=*/false,
                                   /*share_callable=*/true, /*share_callback=*/true, /*is_heavy=*/true);
  std::shared_ptr<BatchSwapRequest> request = std::make_shared<BatchSwapRequest>();
  request->index = executor_id;
  request->is_swapin = is_swapin;
  request->device_block_ids = device_block_ids;
  request->host_block_ids = host_block_ids;
  std::shared_ptr<BatchSwapResponse> response = std::make_shared<BatchSwapResponse>();

  Status send_status = ipc_func_proxy_->Invoke(request.get(), response.get(), ipc_context.get());
  if (!send_status.OK()) {
    return send_status;
  }

  KLLM_LOG_SCHEDULER << "BatchSwap finished. executor_id=" << executor_id << ", is_swapin=" << is_swapin
                     << ", device_block_ids=" << device_block_ids << ", host_block_ids=" << host_block_ids
                     << ", status=" << response->swap_status.GetMessage();
  return response->swap_status;
}

Status IpcFuncWrapper::DispatchGlobalCacheQuery(int executor_id, int64_t req_id, const std::vector<int>& token_ids,
                                                size_t local_cached_len, int block_size,
                                                const std::vector<int>& kv_cache_block_ids) {
  std::shared_ptr<IpcContext> ipc_context =
      CreateEngine2ExecutorContext(IPC_FUNC_GLOBAL_CACHE_QUERY, executor_id, /*is_async=*/true, /*is_oneway=*/true,
                                   /*share_callable=*/false, /*share_callback=*/false, /*is_heavy=*/true);
  std::shared_ptr<GlobalCacheQueryRequest> request = std::make_shared<GlobalCacheQueryRequest>();
  request->req_id = req_id;
  request->token_ids = token_ids;
  request->local_cached_len = local_cached_len;
  request->block_size = block_size;
  request->kv_cache_block_ids = kv_cache_block_ids;

  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  return Status();
}

Status IpcFuncWrapper::ReportGlobalCacheQueryDone(int64_t req_id, int device_id, const std::vector<int>& matched_ranges,
                                                  bool success, const std::vector<int>& cached_tokens,
                                                  bool has_cached_tokens) {
  std::shared_ptr<IpcContext> ipc_context = CreateExecutor2EngineContext(
      IPC_FUNC_GLOBAL_CACHE_QUERY_DONE, executor_id_, /*is_async=*/true, /*is_oneway=*/true,
      /*share_callable=*/false, /*share_callback=*/false);
  std::shared_ptr<GlobalCacheQueryDoneRequest> request = std::make_shared<GlobalCacheQueryDoneRequest>();
  request->req_id = req_id;
  request->device_id = device_id;
  request->matched_ranges = matched_ranges;
  request->success = success;
  request->cached_tokens = cached_tokens;
  request->has_cached_tokens = has_cached_tokens;

  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  return Status();
}

void IpcFuncWrapper::SetGlobalCacheQueryCallableDeps(
    std::shared_ptr<GlobalCacheConnectorInterface> connector,
    std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager, int dp_group_id,
    std::shared_ptr<Environment> env, int device_id, const std::string& role, int timeout_ms) {
  auto callable = ipc_func_proxy_->GetIpcFuncCallable<GlobalCacheQueryCallable>(IPC_FUNC_GLOBAL_CACHE_QUERY, device_id);
  callable->SetGlobalCacheConnector(connector);
  callable->SetBlockAllocatorManager(block_allocator_manager, dp_group_id);
  callable->SetIpcFuncWrapper(this);
  callable->SetEnvironment(env);
  callable->SetDeviceId(device_id);
  callable->SetGlobalCacheRole(role, timeout_ms);
}

void IpcFuncWrapper::SetGlobalCacheQueryDoneCallableFetcher(std::shared_ptr<GlobalCacheFetcher> fetcher) {
  for (size_t executor_id = 0; executor_id < executor_num_; ++executor_id) {
    auto callable = ipc_func_proxy_->GetIpcFuncCallable<GlobalCacheQueryDoneCallable>(IPC_FUNC_GLOBAL_CACHE_QUERY_DONE,
                                                                                      executor_id);
    callable->SetGlobalCacheFetcher(fetcher);
  }
}

Status IpcFuncWrapper::ReportGlobalCachePutDone(int device_id, int64_t req_id) {
  std::shared_ptr<IpcContext> ipc_context =
      CreateExecutor2EngineContext(IPC_FUNC_GLOBAL_CACHE_PUT_DONE, executor_id_, /*is_async=*/true, /*is_oneway=*/true,
                                   /*share_callable=*/false, /*share_callback=*/false);
  std::shared_ptr<GlobalCachePutDoneRequest> request = std::make_shared<GlobalCachePutDoneRequest>();
  request->device_id = device_id;
  request->req_id = req_id;

  STATUS_CHECK_RETURN(ipc_func_proxy_->Invoke(request.get(), nullptr, ipc_context.get()));
  return Status();
}

void IpcFuncWrapper::SetGlobalCachePutDoneCallableTracker(std::shared_ptr<GlobalCachePutTracker> tracker) {
  for (size_t executor_id = 0; executor_id < executor_num_; ++executor_id) {
    auto callable =
        ipc_func_proxy_->GetIpcFuncCallable<GlobalCachePutDoneCallable>(IPC_FUNC_GLOBAL_CACHE_PUT_DONE, executor_id);
    callable->SetGlobalCachePutTracker(tracker);
  }
}

}  // namespace ksana_llm
