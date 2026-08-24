/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/inference_executor.h"

#include <dlfcn.h>
#include <unistd.h>

#include <condition_variable>
#include <memory>
#include <mutex>

#include "cache_manager/cache_layout/cache_layout_factory.h"
#include "connector/config.h"
#include "executor/global_cache_put_worker.h"
#include "executor/layer_progress_tracker.h"
#include "global_cache_connector/global_cache_connector_factory.h"
#include "ipc/functions/create_block_manager.h"
#include "pd_v2/configure/pd_v2_connector_config_builder.h"
#include "pd_v2/public/pd_v2_connector_bootstrap.h"
#include "profiler/profiler.h"
#include "runtime/fast_path_controller.h"
#include "runtime/model_instance.h"
#include "runtime/speculative/common/speculative_resource_profile.h"
#include "runtime/speculative/orchestration/composite_speculative_decoder.h"
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

InferenceExecutor::~InferenceExecutor() {
  if (pd_v2_connector_) {
    pd_v2_connector_->Shutdown().GetMessage();
    pd_v2_connector_.reset();
  }
  if (pd_v2_dlopen_handle_ != nullptr) {
    dlclose(pd_v2_dlopen_handle_);
    pd_v2_dlopen_handle_ = nullptr;
  }
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

  // 与 Engine 相同：在构造 ModelInstance / Sampler 之前完成启动期 gate 判定。
  FastPathController::GetInstance().InitializeAtStartup(batch_scheduler_config, runtime_config, model_config);

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

  if (model_config.use_linear_attn) {
    mamba_block_allocator_manager_ = std::make_shared<BlockAllocatorManager>(false, memory_allocator, context_);
    ipc_func_wrapper_->SetMambaBlockAllocatorManager(mamba_block_allocator_manager_);
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

    const char *use_tcp_data_channel = std::getenv("USE_TCP_DATA_CHANNEL");
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

  // Register speculative decoder adapters via SpeculativeRegistry factory.
  // Registration order: Trie/ASR first, then PTP, then MTP.
  auto composite_decoder = std::make_shared<CompositeSpeculativeDecoder>();

  if (batch_scheduler_config.enable_speculative_decoding) {
    // Unified decoder factory lookup via SpeculativeRegistry.
    // Each adapter registers its factory during static init.
    auto result =
        SpeculativeRegistry::Get().CreateDecoder(batch_scheduler_config.speculative_method, batch_scheduler_config);
    if (result.adapter) {
      composite_decoder->Register(std::move(result.adapter));
      KLLM_LOG_INFO << "Registered speculative decoder: " << batch_scheduler_config.speculative_method;
    } else {
      KLLM_LOG_WARNING << "Unknown speculative_method: " << batch_scheduler_config.speculative_method
                       << ", speculative decoding disabled.";
      batch_scheduler_config.enable_speculative_decoding = false;
    }
  }
  if (batch_scheduler_config.ptp_step_num > 0) {
    auto result = SpeculativeRegistry::Get().CreateDecoder("ptp", batch_scheduler_config);
    if (result.adapter) {
      composite_decoder->Register(std::move(result.adapter));
      KLLM_LOG_INFO << "Registered speculative decoder: ptp"
                    << ", ptp_step_num=" << batch_scheduler_config.ptp_step_num
                    << ", ptp_token_id=" << batch_scheduler_config.ptp_token_id;
    }
  }
  model_runner_->SetCompositeSpeculativeDecoder(composite_decoder);

  executor_runtime_->SetCompositeSpeculativeDecoder(composite_decoder);

  // Execute post-init setup callbacks registered by speculative decoder modules.
  // Each decoder (e.g. MTP) registers a static PostInitSetup that binds their
  // lazy-creation logic to ExecutorRuntime::Initialize() lifecycle.
  SpeculativeSetupContext setup_ctx{composite_decoder, model_instance_, context_, executor_runtime_.get()};
  SpeculativeRegistry::Get().RunPostInitSetups(setup_ctx);

  // Inject GenerationController so CompositeSpeculativeDecoder can delegate
  // CreateGenerators/RestoreGenerators/UpdateGenerationState/RemoveGenerator.
  composite_decoder->SetGenerationController(executor_runtime_->GetGenerationController().get());

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
    // SWA block 字节数与主 KV 不同，connector 需要它建独立的 staging pool。
    if (swa_block_allocator_manager_) {
      auto swa_group = swa_block_allocator_manager_->GetBlockAllocatorGroup(dp_group_id);
      if (swa_group) {
        global_cache_config.swa_block_byte_size = swa_group->GetBlockAllocatorGroupConfig().block_size;
      }
    }
    global_cache_connector_ = GlobalCacheConnectorFactory::CreateConnector(global_cache_config);
    KLLM_CHECK_WITH_INFO(
        global_cache_connector_ != nullptr,
        FormatStr("Failed to create/initialize global cache connector for device %d", context_->device->GetDeviceId()));
    if (global_cache_connector_) {
      int device_id = context_->device->GetDeviceId();
      // pd_v2 节点自动覆盖 role: 与 engine 侧 fetcher_role 保持一致, 让 QueryCallable /
      // PutWorker 进 pd_v2 分支 (单次查询、不 reject、不强制 PutFirstTokens).
      std::string effective_role = global_cache_config.role;
      PdV2RuntimeConfig pd_v2_rt;
      if (env_->GetPdV2RuntimeConfig(pd_v2_rt).OK() && pd_v2_rt.is_pd_v2) {
        if (pd_v2_rt.group_role == GroupRole::PREFILL) {
          effective_role = "pd_v2_prefill";
        } else if (pd_v2_rt.group_role == GroupRole::DECODE) {
          effective_role = "pd_v2_decode";
        }
      }
      // decode 角色会在 connector 内部循环重试直到全命中，这与 SWA 的「命中边界必须对齐到
      // 有 SWA 的位置」冲突：重试期间边界会变，拉到的 SWA 与最终命中长度不再对应。
      // 不静默降级，直接报错退出。
      if (swa_block_allocator_manager_) {
        KLLM_CHECK_WITH_INFO(effective_role != "decode" && effective_role != "pd_v2_decode",
                             FormatStr("global cache + SWA is not supported for role '%s'.", effective_role.c_str()));
        if (context_->global->GetWorldSize() > 1) {
          KLLM_LOG_WARNING << "global cache + SWA with world_size > 1: block key has no pipeline rank, "
                              "multiple pipeline nodes will overwrite each other's entries.";
        }
      }

      ipc_func_wrapper_->SetGlobalCacheQueryCallableDeps(global_cache_connector_, block_allocator_manager_, dp_group_id,
                                                         env_, device_id, effective_role,
                                                         global_cache_config.timeout_ms, swa_block_allocator_manager_);

      // 创建 GlobalCachePutWorker 并注入到 ModelRunner
      auto global_cache_put_worker = std::make_shared<GlobalCachePutWorker>(
          global_cache_connector_, block_allocator_manager_, dp_group_id, device_id, ipc_func_wrapper_, effective_role,
          global_cache_config.fetch_thread_num);
      if (swa_block_allocator_manager_) {
        global_cache_put_worker->SetSwaConfig(swa_block_allocator_manager_, global_cache_config.swa_tail_block_num,
                                              global_cache_config.swa_front_equal_parts);
      }
      global_cache_put_worker->Start();
      model_runner_->SetGlobalCachePutWorker(global_cache_put_worker);

      KLLM_LOG_INFO << "GlobalCacheConnector and GlobalCachePutWorker created in executor for device " << device_id;
    }
  }

  // pd_v2: dlopen the backend .so, create the executor connector, bind it to
  // IpcFuncWrapper (so the engine's PdV2GetKvPoolBaseVa RPC resolves back to
  // GetLocalKvPool), register the KV-pool callback, and — Prefill only — stand
  // up per-layer KV-write infrastructure. Details live in
  // BootstrapPdV2Connector; non-pd_v2 processes no-op. The actual RegisterKvPool
  // (KV pool VA + shape → mooncake registerLocalMemory) fires later when the
  // contiguous KV pool is available. See TODO(C-impl exec KV pool register).
  BootstrapPdV2Connector();

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

void InferenceExecutor::BootstrapPdV2Connector() {
  PdV2RuntimeConfig pd_v2_rt;
  if (!env_->GetPdV2RuntimeConfig(pd_v2_rt).OK() || !pd_v2_rt.is_pd_v2) {
    return;
  }

  // Config-mapping (per-rank identity + RPC port) and dlopen/create/init are
  // shared with inference_engine through the pd_v2 layer
  // (BuildExecutorConnectorConfig + CreateAndInitConnector); the executor only
  // keeps the KV-pool / per-layer wiring below.
  const pd_v2::PdV2ConnectorConfig pd_cfg = pd_v2::BuildExecutorConnectorConfig(pd_v2_rt, executor_id_);
  pd_v2_connector_ = pd_v2::CreateAndInitConnector(pd_cfg, pd_v2_rt.redis_password, ipc_func_wrapper_.get(),
                                                   &pd_v2_dlopen_handle_, /*context_hint=*/"executor");

  // Wire the connector into CreateBlockManagerCallable so RegisterKvPool
  // fires automatically after block_allocator_manager_->Initialize.
  auto cbm_callable = ipc_func_wrapper_->GetIpcFuncProxy()->GetSharedIpcFuncCallable<CreateBlockManagerCallable>(
      IPC_FUNC_CREATE_BLOCK_ALLOCATOR);
  if (cbm_callable) {
    // Each executor process owns one (dp_group_id, device_idx_in_dp_group)
    // tuple inside the global TP/DP topology — same coordinates the
    // block_allocator_manager_ setup used. Forward both so pd_v2 RegisterKvPool
    // can locate THIS process's DeviceBlockAllocator under attn_DP > 1.
    const int attn_tp = static_cast<int>(context_->global->GetAttentionTensorParallelSize());
    const int safe_attn_tp = attn_tp > 0 ? attn_tp : 1;
    const int dp_group_id = executor_id_ / safe_attn_tp;
    const int device_idx_in_dp_group = executor_id_ % safe_attn_tp;
    cbm_callable->SetPdV2Connector(pd_v2_connector_, env_, dp_group_id, device_idx_in_dp_group);
  }

  if (pd_v2_rt.group_role == GroupRole::PREFILL) {
    SetupPdV2PrefillKvWrite(pd_v2_rt);
  }
}

void InferenceExecutor::SetupPdV2PrefillKvWrite(const PdV2RuntimeConfig &pd_v2_rt) {
  ModelConfig pd_v2_model_config;
  env_->GetModelConfig(pd_v2_model_config);
  const uint32_t num_layers = pd_v2_model_config.num_layer;
  const uint32_t prefill_tp = static_cast<uint32_t>(context_->global->GetAttentionTensorParallelSize());
  const uint32_t chunk_size = pd_v2_rt.transfer_layer_chunk_size;

  pd_v2_layer_tracker_ = Singleton<pd_v2::PdV2LayerEventTracker>::GetInstance();
  pd_v2_layer_tracker_->Initialize(executor_id_, static_cast<int>(num_layers));

  pd_v2::PdV2KvBarrier::Deps barrier_deps;
  barrier_deps.connector = pd_v2_connector_.get();
  barrier_deps.ipc_func_wrapper = ipc_func_wrapper_.get();
  pd_v2_kv_barrier_ = std::make_unique<pd_v2::PdV2KvBarrier>(barrier_deps);

  pd_v2::PdV2KvDispatcher::Deps disp_deps;
  disp_deps.connector = pd_v2_connector_.get();
  disp_deps.request_manager = executor_runtime_->GetExecutorRequestManager();
  disp_deps.transfer_layer_chunk_size = chunk_size;
  disp_deps.num_layers = num_layers;
  disp_deps.prefill_tp = prefill_tp;
  // attn_TP per Prefill DP group; dispatcher needs this to compute each rank's
  // local DeviceBlockAllocator slot. Already equals
  // GetAttentionTensorParallelSize() (= TP / DP).
  disp_deps.prefill_attn_tp_per_dp = prefill_tp;
  disp_deps.use_mla = pd_v2_model_config.use_mla;
  disp_deps.barrier = pd_v2_kv_barrier_.get();
  pd_v2_kv_dispatcher_ = std::make_unique<pd_v2::PdV2KvDispatcher>(disp_deps);

  auto *dispatcher_raw = pd_v2_kv_dispatcher_.get();
  pd_v2_layer_tracker_->RegisterCallback([dispatcher_raw](int rank, uint64_t epoch_id, int layer_index) {
    dispatcher_raw->OnLayerComplete(rank, epoch_id, layer_index);
  });
  // Guaranteed per-epoch cleanup: ResetState fires this for every epoch it
  // seals (even aborted forwards that never reached the final layer), so
  // dispatch_plans_by_epoch_ can't leak strong request refs and starve the
  // BlockAllocator.
  pd_v2_layer_tracker_->RegisterEpochEndCallback(
      [dispatcher_raw](int rank, uint64_t epoch_id) { dispatcher_raw->OnEpochEnd(rank, epoch_id); });
  // Pin batch reqs at BeginBatch for PdV2KvDispatcher.
  model_runner_->SetEpochReqCapture(
      [dispatcher_raw](uint64_t epoch_id, const std::vector<std::shared_ptr<ExecutorInferRequest>> &reqs) {
        dispatcher_raw->CaptureEpochReqs(epoch_id, reqs);
      });
  KLLM_LOG_INFO << "pd_v2 Prefill executor: layer_tracker + kv_dispatcher + "
                << "kv_barrier wired (num_layers=" << num_layers << " prefill_tp=" << prefill_tp
                << " chunk_size=" << chunk_size << " use_mla=" << pd_v2_model_config.use_mla << ")";
}

}  // namespace ksana_llm
