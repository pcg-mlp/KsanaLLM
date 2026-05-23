/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/inference_executor.h"

#include <unistd.h>

#include <condition_variable>
#include <memory>
#include <mutex>

#include "cache_manager/cache_layout/cache_layout_factory.h"
#include "executor/global_cache_put_worker.h"
#include "executor/layer_progress_tracker.h"
#include "global_cache_connector/global_cache_connector_factory.h"
#include "profiler/profiler.h"
#include "runtime/draft_generator/asr_custom_generator.h"
#include "runtime/draft_generator/ptp_generator.h"
#include "runtime/draft_generator/trie_generator.h"
#include "runtime/weight_instance.h"
#include "utils/host_memory_pool.h"
#include "utils/memory_allocator.h"

#ifdef ENABLE_CUDA
#  include "ipc/functions/nccl_operations.h"
#endif

namespace ksana_llm {

InferenceExecutor::InferenceExecutor(int executor_id, std::shared_ptr<Environment> env,
                                     std::shared_ptr<Context> context)
    : executor_id_(executor_id), env_(env), context_(context) {
  terminated_.store(false);

  // Set distributed config.
  PipelineConfig pipeline_config;
  env_->GetPipelineConfig(pipeline_config);
  pipeline_config.SetDistributeRelatedConfig();
  env_->SetPipelineConfig(pipeline_config);

  // Initialize EP config from environment variables first
  env_->InitializeExpertParallelConfig();

  // Then initialize distributed config which reads EP config
  context_->global->InitDistributedConfig();
  KLLM_LOG_DEBUG << "Set distributed config, is_standalone:" << context_->global->IsStandalone()
                 << ", is_distributed_master::" << context_->global->IsDistributedMaster();

  SetDevice(executor_id);
  int node_device_num = context_->global->GetTensorParallelSize();
  context_->device = std::make_shared<DeviceContext>(node_device_num);

  model_runner_ = std::make_shared<ModelRunner>(context_);
}

void InferenceExecutor::SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper) {
  ipc_func_wrapper_ = ipc_func_wrapper;
  model_runner_->SetIpcFuncWrapper(ipc_func_wrapper_);
}

void InferenceExecutor::Start() {
  KLLM_LOG_INFO << "InferenceExecutor::Start invoked.";

  // NOTE(jinxcwu)
  // 在处理请求之前初始化 host 内存池。
  // 这确保了在初始化阶段 host tensor 使用动态分配，
  // 而在请求处理阶段 host tensor 使用内存池分配。
  BlockManagerConfig block_mgr_cfg;
  env_->GetBlockManagerConfig(block_mgr_cfg);
  HostMemoryPool::GetInstance()->Initialize(block_mgr_cfg.host_pool_size_gb);

  STATUS_CHECK_FAILURE(model_runner_->Start());
  KLLM_LOG_INFO << "InferenceExecutor::Start finished.";
}

void InferenceExecutor::Stop() {
  DestroyProcessProfiler();
  std::lock_guard<std::mutex> lock(mutex_);
  terminated_.store(true);
  cond_.notify_all();
}

void InferenceExecutor::BlockingForever() {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this] { return terminated_.load(); });

  // Force exit, should not return.
  _exit(EXIT_FAILURE);
}

Status InferenceExecutor::Initialize() {
  KLLM_LOG_INFO << "InferenceExecutor::Initialize invoked.";

  ModelConfig model_config;
  STATUS_CHECK_RETURN(env_->GetModelConfig(model_config));

  RuntimeConfig runtime_config;
  STATUS_CHECK_RETURN(env_->GetRuntimeConfig(runtime_config));

  size_t max_batch_size = runtime_config.max_batch_size;
  size_t max_vocab_size = model_config.vocab_size;
  BatchSchedulerConfig batch_scheduler_config;
  STATUS_CHECK_RETURN(env_->GetBatchSchedulerConfig(batch_scheduler_config));

  batch_scheduler_config.max_batch_size = max_batch_size;
  batch_scheduler_config.max_vocab_size = max_vocab_size;
  env_->SetBatchSchedulerConfig(batch_scheduler_config);
  KLLM_LOG_DEBUG << "Reset batch_scheduler_config, max_batch_size: " << max_batch_size
                 << ", max_vocab_size: " << max_vocab_size;

  model_weight_ = std::make_shared<WeightInstance>(model_config, runtime_config, context_);
  ipc_func_wrapper_->SetWeightInstance(model_weight_);

  model_instance_ = std::make_shared<ModelInstance>(model_config, runtime_config, context_, model_weight_);
  ipc_func_wrapper_->SetModelInstance(model_instance_);

  // Create cache layout.
  block_cache_layout_ = CacheLayoutFactory::CreateModelCacheLayout(model_config);
  model_instance_->SetBlockCacheLayout(block_cache_layout_);
  ipc_func_wrapper_->SetBlockCacheLayout(block_cache_layout_);

  ipc_func_wrapper_->SetEnvironment(env_);
  ipc_func_wrapper_->SetDeviceCommunicationContext(context_);
  ipc_func_wrapper_->SetCustomCommunicationContext(context_);
  ipc_func_wrapper_->SetMulticastCommunicationContext(context_);

#ifdef ENABLE_CUDA
  // Create NCCL executor handler for this executor's GPU
  ConnectorConfig connector_config;
  env_->GetConnectorConfigs(connector_config);
  nccl_executor_handler_ = std::make_shared<NcclExecutorHandler>(context_->device->GetDeviceId(), connector_config);

  // Set NCCL handler to all Connector NCCL IPC callables
  auto init_callable =
      ipc_func_wrapper_->GetIpcFuncProxy()->GetSharedIpcFuncCallable<ConnectorInitNcclCommGroupCallable>(
          IPC_FUNC_CONNECTOR_INIT_NCCL_COMM_GROUP);
  init_callable->SetNcclExecutorHandler(nccl_executor_handler_);

  auto send_callable = ipc_func_wrapper_->GetIpcFuncProxy()->GetSharedIpcFuncCallable<ConnectorNcclSendDataCallable>(
      IPC_FUNC_CONNECTOR_NCCL_SEND_DATA);
  send_callable->SetNcclExecutorHandler(nccl_executor_handler_);

  auto recv_callable = ipc_func_wrapper_->GetIpcFuncProxy()->GetSharedIpcFuncCallable<ConnectorNcclRecvDataCallable>(
      IPC_FUNC_CONNECTOR_NCCL_RECV_DATA);
  recv_callable->SetNcclExecutorHandler(nccl_executor_handler_);

  auto send_blocks_callable =
      ipc_func_wrapper_->GetIpcFuncProxy()->GetSharedIpcFuncCallable<ConnectorNcclSendBlocksCallable>(
          IPC_FUNC_CONNECTOR_NCCL_SEND_BLOCKS);
  send_blocks_callable->SetNcclExecutorHandler(nccl_executor_handler_);

  auto recv_blocks_callable =
      ipc_func_wrapper_->GetIpcFuncProxy()->GetSharedIpcFuncCallable<ConnectorNcclRecvBlocksCallable>(
          IPC_FUNC_CONNECTOR_NCCL_RECV_BLOCKS);
  recv_blocks_callable->SetNcclExecutorHandler(nccl_executor_handler_);

  auto info_callable = ipc_func_wrapper_->GetIpcFuncProxy()->GetSharedIpcFuncCallable<ConnectorGetNcclCommInfoCallable>(
      IPC_FUNC_CONNECTOR_GET_NCCL_COMM_INFO);
  info_callable->SetNcclExecutorHandler(nccl_executor_handler_);

  KLLM_LOG_INFO << "Created NcclExecutorHandler for device " << context_->device->GetDeviceId();
#endif

  // Set block manager, created in ipc function.
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator = std::make_shared<MemoryAllocator>();
  block_allocator_manager_ = std::make_shared<BlockAllocatorManager>(false, memory_allocator, context_);
  int dp_group_id = context_->device->GetDeviceId() / context_->global->GetAttentionTensorParallelSize();
  int device_idx_in_dp_group = context_->device->GetDeviceId() % context_->global->GetAttentionTensorParallelSize();
  ipc_func_wrapper_->SetBlockAllocatorManager(block_allocator_manager_, dp_group_id);

  // Create and register the SWA block allocator manager only when SWA is enabled,
  // indicated by model_config.dsv4_attn_config.sliding_window > 0.
  if (model_config.dsv4_attn_config.sliding_window > 0) {
    KLLM_LOG_INFO << "InferenceExecutor create SWA block allocator manager, sliding_window="
                  << model_config.dsv4_attn_config.sliding_window;
    swa_block_allocator_manager_ = std::make_shared<BlockAllocatorManager>(false, memory_allocator, context_);
    ipc_func_wrapper_->SetSwaBlockAllocatorManager(swa_block_allocator_manager_);
  }

  if (context_->global->IsDistributedWorker()) {
    CacheManagerConfig cache_manager_config;
    STATUS_CHECK_RETURN(env_->GetCacheManagerConfig(cache_manager_config));
    executor_block_manager_ = std::make_shared<ExecutorBlockManager>(
        dp_group_id, device_idx_in_dp_group, block_allocator_manager_, cache_manager_config.swap_threadpool_size);
    model_runner_->SetExecutorBlockManager(executor_block_manager_);
  }

#ifdef ENABLE_CUDA
  // Set BlockAllocatorManager to NcclExecutorHandler for pointer resolution
  if (nccl_executor_handler_) {
    int rank = context_->device->GetDeviceId();
    nccl_executor_handler_->SetBlockAllocatorManager(block_allocator_manager_, rank, runtime_config);
    KLLM_LOG_INFO << "Set BlockAllocatorManager to NcclExecutorHandler with rank=" << rank;
  }
#endif

  // Set executor runtime, created in ipc function.
  executor_runtime_ = std::make_shared<ExecutorRuntime>(env_, context_);
  ipc_func_wrapper_->SetExecutorRuntime(executor_runtime_);

  // Set environment for executor to receive vocab info from Engine
  ipc_func_wrapper_->SetExecutorEnvironment(env_);

  model_runner_->SetExecutorRuntime(executor_runtime_);
  model_runner_->SetModelInstance(model_instance_);
  model_runner_->SetModelWeight(model_weight_);
  model_runner_->SetBlockAllocatorManager(block_allocator_manager_);
  if (swa_block_allocator_manager_) {
    model_runner_->SetSwaBlockAllocatorManager(swa_block_allocator_manager_);
  }

  // Only for distributed mode.
  if (!context_->global->IsStandalone()) {
    distributed_data_coordinator_ = std::make_shared<DistributedDataCoordinator>(GetPacketObject, env_, context_);

    const char* use_tcp_data_channel = std::getenv("USE_TCP_DATA_CHANNEL");
    if (use_tcp_data_channel && strcmp(use_tcp_data_channel, "1") == 0) {
      distributed_data_coordinator_->InitializeTcpDataChannel();
      ipc_func_wrapper_->SetTcpDistributedDataCoordinator(distributed_data_coordinator_);
#ifdef ENABLE_CUDA
    } else {
      distributed_data_coordinator_->InitializeNcclDataChannl();
      ipc_func_wrapper_->SetNcclDistributedDataCoordinator(distributed_data_coordinator_);
#endif
    }

    // Initialize result report, on executor 0 of last worker only.
    if (context_->global->IsDistributedResultReporter() && context_->device->GetDeviceId() == 0) {
      sampling_output_pool_ = std::make_shared<SamplingOutputPool>();
      distributed_control_coordinator_ =
          std::make_shared<DistributedControlCoordinator>(GetPacketObject, nullptr, sampling_output_pool_, context_);
      ipc_func_wrapper_->SetReporterDistributedControlCoordinator(distributed_control_coordinator_);
      model_runner_->SetSamplingOutputPool(sampling_output_pool_);
    }

    KLLM_LOG_INFO << "InferenceExecutor::Initialize distributed mode, initialize data coordinator succeed.";
  } else {
    KLLM_LOG_INFO << "InferenceExecutor::Initialize standalone mode, skip distributed coordinator.";
  }

  // Create draft generator controller.
  // Trie/AsrCustomGenerator and PtpGenerator are independent draft generators that can be enabled separately:
  //   - Trie/AsrCustomGenerator: controlled by enable_speculative_decoding, predicts draft tokens via specific method.
  //   - PtpGenerator: controlled by ptp_step_num (from model config ptp_step_num), does NOT require
  //                   enable_speculative_decoding.
  auto draft_generator_controller = std::make_shared<DraftGeneratorController>();
  if (batch_scheduler_config.enable_speculative_decoding) {
    if (batch_scheduler_config.speculative_method == "trie") {
      draft_generator_controller->AppendDraftGenerator(std::make_shared<TrieGenerator>());
      KLLM_LOG_INFO << "Added TrieGenerator to DraftGeneratorController";
    } else {  // batch_scheduler_config.speculative_method == "asr_custom"
      draft_generator_controller->AppendDraftGenerator(std::make_shared<AsrCustomGenerator>());
      KLLM_LOG_INFO << "Added AsrCustomGenerator to DraftGeneratorController";
    }
  }
  if (batch_scheduler_config.ptp_step_num > 0) {
    draft_generator_controller->AppendDraftGenerator(
        std::make_shared<PtpGenerator>(batch_scheduler_config.ptp_step_num, batch_scheduler_config.ptp_token_id));
    KLLM_LOG_INFO << "Added PtpGenerator to DraftGeneratorController with ptp_step_num="
                  << batch_scheduler_config.ptp_step_num << ", ptp_token_id=" << batch_scheduler_config.ptp_token_id;
  }
  model_runner_->SetDraftGeneratorController(draft_generator_controller);

  // For standalone mode, the first argument is nullptr.
  hidden_unit_buffer_pool_ = std::make_shared<HiddenUnitBufferPool>(distributed_data_coordinator_, env_, context_);
  model_runner_->SetHiddenUnitBufferPool(hidden_unit_buffer_pool_);
  Singleton<LayerProgressTracker>::GetInstance()->Initialize(
      context_->device->GetDeviceId(), model_config.num_layer + context_->global->GetMtpStepNum());
  Singleton<LayerProgressTracker>::GetInstance()->SetIpcFuncWrapper(ipc_func_wrapper_);

  // Initialize global cache connector for this executor's device.
  GlobalCacheConnectorConfig global_cache_config;
  Status global_cache_status = env_->GetGlobalCacheConnectorConfig(global_cache_config);
  if (global_cache_status.OK()) {
    global_cache_config.device_id = context_->device->GetDeviceId();
    global_cache_config.device_count = context_->global->GetAttentionTensorParallelSize();
    global_cache_config.block_byte_size = runtime_config.attn_backend_config.block_size;
    global_cache_connector_ = GlobalCacheConnectorFactory::CreateConnector(global_cache_config);
    if (global_cache_connector_) {
      int device_id = context_->device->GetDeviceId();
      ipc_func_wrapper_->SetGlobalCacheQueryCallableDeps(global_cache_connector_, block_allocator_manager_, dp_group_id,
                                                         env_, device_id, global_cache_config.role,
                                                         global_cache_config.timeout_ms);

      // 创建 GlobalCachePutWorker 并注入到 ModelRunner
      auto global_cache_put_worker =
          std::make_shared<GlobalCachePutWorker>(global_cache_connector_, block_allocator_manager_, dp_group_id,
                                                 device_id, ipc_func_wrapper_, global_cache_config.role);
      global_cache_put_worker->Start();
      model_runner_->SetGlobalCachePutWorker(global_cache_put_worker);

      KLLM_LOG_INFO << "GlobalCacheConnector and GlobalCachePutWorker created in executor for device " << device_id;
    }
  }

  // Initialize per-process profiler for this executor.
  ProfilerConfig profiler_config;
  if (env_->GetProfilerConfig(profiler_config).OK()) {
    profiler_config.process_role = "executor";
    profiler_config.executor_id = executor_id_;
    InitProcessProfiler(profiler_config);
  }

  KLLM_LOG_INFO << "InferenceExecutor::Initialize finished.";
  return Status();
}

}  // namespace ksana_llm
