/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "engine/inference_engine.h"

#include <dlfcn.h>
#include <unistd.h>

#include <iostream>

#include "batch_scheduler/batch_scheduler.h"
#include "batch_scheduler/event_driven_scheduler/continuous_batching_backend.h"
#include "batch_scheduler/event_driven_scheduler/event_driven_schedule_processor.h"
#include "batch_scheduler/event_driven_scheduler/global_cache_query_hook.h"
#include "batch_scheduler/event_driven_scheduler/model_perf/model_perf_backend.h"
#include "batch_scheduler/event_driven_scheduler/model_perf/model_perf_workload_balancer.h"
#include "batch_scheduler/event_driven_scheduler/schedule_hook_manager.h"
#include "batch_scheduler/event_driven_scheduler/workload_balance/basic_workload_balancer.h"
#include "batch_scheduler/event_driven_scheduler/workload_balance/pp_batch_distributor.h"
#include "batch_scheduler/global_cache_fetcher.h"
#include "batch_scheduler/global_cache_put_tracker.h"
#include "batch_scheduler/schedule_processor.h"
#include "cache_manager/cache_layout/cache_layout_factory.h"
#include "cache_manager/cache_manager_factory.h"
#include "configure/environment.h"
#include "connector/config.h"
#include "connector/connector.h"
#include "endpoints/pd_v2/pd_v2_prefill_endpoint.h"
#include "engine/model_driver.h"
#include "executor/layer_progress_tracker.h"
#include "global_cache_connector/global_cache_connector_factory.h"
#include "ipc/functions/report_layer_progress.h"
#include "pd_v2/configure/pd_v2_connector_config_builder.h"
#include "pd_v2/configure/pd_v2_runtime_config.h"
#include "pd_v2/decode/pd_v2_decode_hook.h"
#include "pd_v2/public/pd_v2_connector.h"
#include "pd_v2/public/pd_v2_connector_bootstrap.h"
#include "periphery/version_reporter.h"
#include "profiler/reporter.h"
#include "profiler/timer.h"
#include "runtime/draft_generator/trie_generator.h"
#include "runtime/fast_path_controller.h"
#include "runtime/infer_request.h"
#include "runtime/weight_instance.h"
#include "transfer/transfer_engine.h"
#include "utils/blocking_queue.h"
#include "utils/host_memory_pool.h"
#include "utils/memory_allocator.h"
#include "utils/ret_code.h"
#include "utils/singleton.h"
#include "utils/status.h"
#include "utils/tensor.h"
#include "utils/tokenizer.h"
#include "utils/waiter.h"

namespace ksana_llm {

InferenceEngine::InferenceEngine(Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue,
                                 std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : env_(env), context_(context), request_queue_(request_queue) {
  schedule_output_pool_ = std::make_shared<ScheduleOutputPool>();
  if (!context_->global->IsDistributedWorker()) {
    sampling_output_pool_ = std::make_shared<SamplingOutputPool>();
  }

  model_driver_ = std::make_shared<ModelDriver>(request_queue_, schedule_output_pool_, sampling_output_pool_, context_);
}

InferenceEngine::~InferenceEngine() {
  // pd_v2: drop shared_ptr before unloading the .so (vtable lives there).
  if (pd_v2_prefill_endpoint_) {
    pd_v2_prefill_endpoint_.reset();
  }
  if (pd_v2_connector_) {
    pd_v2_connector_->Shutdown().GetMessage();
    pd_v2_connector_.reset();
  }
  if (pd_v2_dlopen_handle_ != nullptr) {
    dlclose(pd_v2_dlopen_handle_);
    pd_v2_dlopen_handle_ = nullptr;
  }
  HostMemoryPool::GetInstance()->Destroy();
  KLLM_LOG_DEBUG << "InferenceEngine destroyed.";
}

void InferenceEngine::SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper) {
  ipc_func_wrapper_ = ipc_func_wrapper;
  model_driver_->SetIpcFuncWrapper(ipc_func_wrapper_);
}

void InferenceEngine::SetLocalEndpoint(std::shared_ptr<LocalEndpoint> local_endpoint) {
  local_endpoint_ = std::move(local_endpoint);
}

Status InferenceEngine::Initialize() {
  // Initialize device communications.
  STATUS_CHECK_RETURN(model_driver_->InitDeviceCommunication());

  RuntimeConfig runtime_config;
  STATUS_CHECK_RETURN(env_->GetRuntimeConfig(runtime_config));

  ModelConfig model_config;
  STATUS_CHECK_RETURN(env_->GetModelConfig(model_config));

  PipelineConfig pipeline_config;
  STATUS_CHECK_RETURN(env_->GetPipelineConfig(pipeline_config));

  // Only for distributed mode.
  if (!context_->global->IsStandalone() || !context_->global->IsExpertParallelStandalone()) {
    distributed_control_coordinator_ = std::make_shared<DistributedControlCoordinator>(
        GetPacketObject, schedule_output_pool_, sampling_output_pool_, context_);

    KLLM_LOG_INFO << "InferenceEngine::Initialize distributed mode, start to initialize control cluster.";
    distributed_control_coordinator_->InitializeCluster();
    KLLM_LOG_INFO << "Initialize control cluster succeed.";
  }

  if (!context_->global->IsStandalone()) {
    const char *sampling_node_offload_layer_num_env = std::getenv("SAMPLING_NODE_OFFLOAD_LAYER_NUM");
    size_t sampling_node_offload_layer_num =
        sampling_node_offload_layer_num_env ? std::stoi(sampling_node_offload_layer_num_env) : 0;

    std::vector<int16_t> layer_nums = {static_cast<int16_t>(model_config.num_layer),
                                       static_cast<int16_t>(model_config.num_nextn_predict_layers)};

    KLLM_LOG_INFO << "Start to synchronize node layers, sampling_node_offload_layer_num:"
                  << sampling_node_offload_layer_num;
    std::vector<int16_t> layer_indexes;
    distributed_control_coordinator_->SynchronizeNodeLayers(layer_nums, runtime_config.mtp_step_num,
                                                            sampling_node_offload_layer_num, layer_indexes);
    pipeline_config.lower_layer_idx = layer_indexes[0];
    pipeline_config.upper_layer_idx = layer_indexes[1];
    pipeline_config.lower_nextn_layer_idx = layer_indexes[2];
    pipeline_config.upper_nextn_layer_idx = layer_indexes[3];
    env_->SetPipelineConfig(pipeline_config);

    KLLM_LOG_INFO << "InferenceEngine Synchronize layer range:[" << pipeline_config.lower_layer_idx << ", "
                  << pipeline_config.upper_layer_idx << ", " << pipeline_config.lower_nextn_layer_idx << ", "
                  << pipeline_config.upper_nextn_layer_idx << "].";

  } else {
    KLLM_LOG_INFO << "InferenceEngine::Initialize standalone mode, skip distributed coordinator.";

    pipeline_config.lower_layer_idx = 0;
    pipeline_config.upper_layer_idx = model_config.num_layer - 1;
    if (model_config.num_nextn_predict_layers != 0 && runtime_config.mtp_step_num > 0) {
      pipeline_config.lower_nextn_layer_idx = model_config.num_layer;
      pipeline_config.upper_nextn_layer_idx = model_config.num_layer + model_config.num_nextn_predict_layers - 1;
    }
    env_->SetPipelineConfig(pipeline_config);
    KLLM_LOG_INFO << "InferenceEngine set layer range:[" << pipeline_config.lower_layer_idx << ", "
                  << pipeline_config.upper_layer_idx << ", " << pipeline_config.lower_nextn_layer_idx << ", "
                  << pipeline_config.upper_nextn_layer_idx << "].";
  }

  // Create cache layout.
  block_cache_layout_ = CacheLayoutFactory::CreateModelCacheLayout(model_config, pipeline_config, runtime_config);

  // Get block size config of specific layers, runtime_config is changed, get it again.
  STATUS_CHECK_RETURN(env_->InitializeBlockManagerConfig(block_cache_layout_));
  STATUS_CHECK_RETURN(env_->GetRuntimeConfig(runtime_config));

  // set layer range to executors.
  KLLM_LOG_INFO << "InferenceEngine start reset pipeline config.";
  STATUS_CHECK_RETURN(model_driver_->ResetPipelineConfig(pipeline_config));
  KLLM_LOG_INFO << "InferenceEngine reset pipeline config finished.";

  // Build data communications between executors.
  if (!context_->global->IsStandalone()) {
    KLLM_LOG_INFO << "InferenceEngine start connect data channels.";
    STATUS_CHECK_RETURN(model_driver_->ConnectDataChannels(distributed_control_coordinator_));
    KLLM_LOG_INFO << "InferenceEngine connect data channels finished.";

    KLLM_LOG_INFO << "InferenceEngine start connect result reporter channel.";
    STATUS_CHECK_RETURN(model_driver_->InitializeResultReporter(distributed_control_coordinator_));
    KLLM_LOG_INFO << "InferenceEngine connect result reporter channel finished.";
  }

  // Load weight.
  KLLM_LOG_INFO << "InferenceEngine start load weight.";
  STATUS_CHECK_RETURN(model_driver_->LoadWeight());
  KLLM_LOG_INFO << "InferenceEngine load weight finished.";

  // Load model.
  KLLM_LOG_INFO << "InferenceEngine start load model.";
  STATUS_CHECK_RETURN(model_driver_->LoadModel());
  KLLM_LOG_INFO << "InferenceEngine load model finished.";

  // Initialize custom/multicast communications after loading model.
  STATUS_CHECK_RETURN(model_driver_->InitCustomCommunication());
  STATUS_CHECK_RETURN(model_driver_->InitMulticastCommunication());

  // Synchronize EP communicator config.
  if (!context_->global->IsExpertParallelStandalone()) {
    STATUS_CHECK_RETURN(model_driver_->SynchronizeEPCommunicationConfig(distributed_control_coordinator_));
  }

  ProfilerConfig profiler_config;
  STATUS_CHECK_RETURN(env_->GetProfilerConfig(profiler_config));
  profiler_config.process_role = "engine";
  InitProcessProfiler(profiler_config);

  // Create batch scheduler.
  // NOTE: BatchSchedulerConfig is also needed on distributed workers (e.g. rank1 in PP mode) to
  // correctly detect flags like enable_xgrammar, otherwise the vocab info loading below would be
  // skipped on workers and the executor process would fail with "Vocab info not initialized".
  BatchSchedulerConfig batch_scheduler_config;
  STATUS_CHECK_RETURN(env_->GetBatchSchedulerConfig(batch_scheduler_config));
  if (!context_->global->IsDistributedWorker()) {
    batch_scheduler_config.max_batch_size = runtime_config.max_batch_size;
    batch_scheduler_config.max_vocab_size = model_config.vocab_size;
    KLLM_LOG_DEBUG << "Batch Scheduler Config Max Batch Size= " << batch_scheduler_config.max_batch_size
                   << ", Max Vocab Size= " << batch_scheduler_config.max_vocab_size
                   << ", max_pp_batch_num=" << batch_scheduler_config.max_pp_batch_num;

    // 启动期硬条件 gate：yaml 总开关 + async/调度器/并行度/高级特性等约束。
    FastPathController::GetInstance().InitializeAtStartup(batch_scheduler_config, runtime_config, model_config);
  }

  // Calc block number after model loaded.
  KLLM_LOG_INFO << "InferenceEngine calc block number.";
  size_t device_free, device_total;
  STATUS_CHECK_RETURN(model_driver_->GetDeviceMemory(&device_free, &device_total));
  KLLM_LOG_INFO << "InferenceEngine Get device memory, free:" << device_free << ", total:" << device_total;
  STATUS_CHECK_RETURN(env_->CalculateBlockNumber(device_free, device_total));

  if (!context_->global->IsStandalone()) {
    KLLM_LOG_INFO << "Start to synchronize cache block num.";

    std::vector<size_t> node_block_nums;
    std::vector<size_t> block_nums = {env_->GetTotalDeviceBlockNum(), env_->GetTotalHostBlockNum()};
    distributed_control_coordinator_->SynchronizeCacheBlockNum(block_nums, node_block_nums);

    pipeline_config.device_block_num = node_block_nums[0];
    pipeline_config.host_block_num = node_block_nums[1];
    KLLM_LOG_INFO << "InferenceEngine Synchronize device block num " << pipeline_config.device_block_num
                  << ", host block_num " << pipeline_config.host_block_num << ".";

    env_->SetPipelineConfig(pipeline_config);
    env_->ResetPipelineBlockNumber();
  }

  KLLM_LOG_INFO << "InferenceEngine create block allocator group.";
  BlockAllocatorManagerConfig block_allocator_manager_config;
  for (size_t dp_id = 0; dp_id < context_->global->GetAttentionDataParallelSize(); ++dp_id) {
    BlockAllocatorGroupConfig dp_group_config;
    dp_group_config.devices = env_->GetDataParaGroupDevices(dp_id);
    dp_group_config.device_block_num = env_->GetTotalDeviceBlockNum();
    dp_group_config.host_block_num = env_->GetTotalHostBlockNum();
    dp_group_config.block_size = runtime_config.attn_backend_config.block_size;

    block_allocator_manager_config[dp_id] = dp_group_config;
  }

  // Build SWA block allocator manager config when SWA is enabled
  // (indicated by model_config.dsv4_attn_config.sliding_window > 0).
  BlockManagerConfig block_manager_config;
  STATUS_CHECK_RETURN(env_->GetBlockManagerConfig(block_manager_config));
  const size_t swa_block_num = block_manager_config.swa_device_allocator_config.blocks_num;
  const size_t swa_block_size = block_manager_config.swa_device_allocator_config.block_size;
  const size_t main_block_num = block_manager_config.device_allocator_config.blocks_num;
  const size_t main_block_size = block_manager_config.device_allocator_config.block_size;
  BatchSchedulerConfig swa_log_bs_cfg;
  STATUS_CHECK_RETURN(env_->GetBatchSchedulerConfig(swa_log_bs_cfg));
  const float swa_block_num_ratio = swa_log_bs_cfg.swa_block_num_ratio;
  BlockAllocatorManagerConfig swa_block_allocator_manager_config;
  if (model_config.dsv4_attn_config.sliding_window > 0) {
    KLLM_LOG_INFO << "InferenceEngine create SWA block allocator group, sliding_window="
                  << model_config.dsv4_attn_config.sliding_window << ", swa_block_num_ratio=" << swa_block_num_ratio
                  << ", swa_block_num=" << swa_block_num << ", swa_block_size=" << swa_block_size
                  << ", main_block_num=" << main_block_num << ", main_block_size=" << main_block_size;
    for (size_t dp_id = 0; dp_id < context_->global->GetAttentionDataParallelSize(); ++dp_id) {
      BlockAllocatorGroupConfig swa_dp_group_config;
      swa_dp_group_config.devices = env_->GetDataParaGroupDevices(dp_id);
      swa_dp_group_config.device_block_num = swa_block_num;
      swa_dp_group_config.host_block_num = 0;
      swa_dp_group_config.block_size = swa_block_size;
      swa_block_allocator_manager_config[dp_id] = swa_dp_group_config;
    }
  }

  // Create cache manager.
  if (!context_->global->IsDistributedWorker()) {
    std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = std::make_shared<MemoryAllocator>();
    BlockAllocatorManager block_allocator_manager(true, block_allocator_manager_config, memory_allocator_, context_);

    // Create SWA block allocator manager when SWA is enabled.
    std::unique_ptr<BlockAllocatorManager> swa_block_allocator_manager;
    if (!swa_block_allocator_manager_config.empty()) {
      swa_block_allocator_manager = std::make_unique<BlockAllocatorManager>(true, swa_block_allocator_manager_config,
                                                                            memory_allocator_, context_);
    }

    CacheManagerConfig cache_manager_config;
    STATUS_CHECK_RETURN(env_->GetCacheManagerConfig(cache_manager_config));

    KLLM_LOG_INFO << "InferenceEngine create cache manager.";
    for (size_t dp_id = 0; dp_id < context_->global->GetAttentionDataParallelSize(); ++dp_id) {
      std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group =
          block_allocator_manager.GetBlockAllocatorGroup(dp_id);
      std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group;
      if (swa_block_allocator_manager) {
        swa_block_allocator_group = swa_block_allocator_manager->GetBlockAllocatorGroup(dp_id);
      }
      cache_managers_.emplace_back(CacheManagerFactory::CreateCacheManager(cache_manager_config, block_allocator_group,
                                                                           swa_block_allocator_group));
    }
  }

  // Create block manager on executors (KV + optional SWA).
  STATUS_CHECK_RETURN(
      model_driver_->CreateBlockManager(block_allocator_manager_config, swa_block_allocator_manager_config));

  if (!context_->global->IsStandalone()) {
    distributed_control_coordinator_->Barrier();
  }

  // Initialize tokenzier
  Singleton<Tokenizer>::GetInstance()->InitTokenizer(model_config.path);

  // Get vocab info and store in Environment for Executor to use
  if (batch_scheduler_config.enable_xgrammar) {
    VocabInfo vocab_info;
    auto tokenizer = Singleton<Tokenizer>::GetInstance();

    Status vocab_status = tokenizer->GetVocabInfo(vocab_info.vocab, vocab_info.vocab_size, vocab_info.stop_token_ids);
    if (vocab_status.OK()) {
      vocab_info.initialized = true;
      // Detect tokenizer metadata (vocab_type, add_prefix_space)
      // This is done in Engine process where Python API is safe to use
      Status metadata_status = tokenizer->DetectTokenizerMetadata(vocab_info);
      if (!metadata_status.OK()) {
        KLLM_LOG_WARNING << "Failed to detect tokenizer metadata: " << metadata_status.GetMessage()
                         << ", using defaults";
      }

      // Get think_end_token_id from Environment (set by InitService earlier)
      vocab_info.think_end_token_id = env_->GetThinkEndTokenId();

      env_->SetVocabInfo(vocab_info);
      KLLM_LOG_INFO << "Vocab info stored in Environment, vocab_size: " << vocab_info.vocab_size
                    << ", stop_token_ids size: " << vocab_info.stop_token_ids.size()
                    << ", vocab_type: " << vocab_info.vocab_type
                    << ", add_prefix_space: " << vocab_info.add_prefix_space
                    << ", think_end_token_id: " << vocab_info.think_end_token_id;
    } else {
      KLLM_LOG_WARNING << "Failed to get vocab info: " << vocab_status.GetMessage();
    }
  }

  // Create TransferEngine instance first (before batch scheduler).
  // pd_v2 has its own data plane (Mooncake tent + libpd_v2_connector_mooncake.so)
  // and does NOT use the V1 TransferEngine / Connector. Skipping V1 here:
  //   - removes "Invalid request_id: -1" / "Task address is empty" log noise,
  //   - reclaims a few seconds of boot time,
  //   - prevents two stacks from racing on the same notify channels.
  // We still set up the layer_progress_queue unconditionally because the
  // V1 LayerProgressTracker singleton in the executor pushes to it
  // regardless of whether V1 transfer is enabled — null queue → crash.
  ConnectorConfig connector_config;
  Status status = env_->GetConnectorConfigs(connector_config);
  if (status.OK() && connector_config.group_role != GroupRole::NONE) {
    auto layer_progress_queue = std::make_shared<BlockingQueue<LayerProgressInfo>>();
    ipc_func_wrapper_->SetLayerProgressRecvQueue(layer_progress_queue);
    if (!connector_config.is_pd_v2) {
      transfer_engine_ = std::make_shared<TransferEngine>();
      // Set IpcFuncWrapper to TransferEngine (which will set it to Connector)
      transfer_engine_->SetIpcFuncWrapper(ipc_func_wrapper_);
      transfer_engine_->Initialize(connector_config.group_role);
      transfer_engine_->SetLayerProgressRecvQueue(layer_progress_queue);
      KLLM_LOG_INFO << "TransferEngine created and initialized.";
    } else {
      KLLM_LOG_INFO << "pd_v2 enabled: skipping V1 TransferEngine / Connector setup "
                    << "(layer_progress_queue still wired so executor's V1 LayerProgressTracker "
                    << "has somewhere to push).";
    }
  }

  // Create global cache connector if configured.
  GlobalCacheConnectorConfig global_cache_config;
  Status global_cache_status = env_->GetGlobalCacheConnectorConfig(global_cache_config);
  if (global_cache_status.OK()) {
    global_cache_config.device_count = context_->global->GetAttentionTensorParallelSize();
    global_cache_config.block_byte_size = runtime_config.attn_backend_config.block_size;

    global_cache_connector_ = GlobalCacheConnectorFactory::CreateConnector(global_cache_config);
    if (global_cache_connector_) {
      KLLM_LOG_INFO << "GlobalCacheConnector created and initialized in engine.";
    } else {
      KLLM_LOG_WARNING << "Failed to create GlobalCacheConnector, global cache will be unavailable.";
    }
  }

  // Create batch scheduler.
  if (!context_->global->IsDistributedWorker()) {
    KLLM_LOG_INFO << "InferenceEngine create batch scheduler.";

    if (batch_scheduler_config.scheduler_type == SchedulerType::EVENT_DRIVEN) {
      const size_t attn_dp_size = context_->global->GetAttentionDataParallelSize();
      auto event_driven_schedule_processor =
          std::make_shared<EventDrivenScheduleProcessor>(attn_dp_size, batch_scheduler_config.max_pp_batch_num);

      const bool use_model_perf_backend = InitModelPerfBackend(event_driven_schedule_processor, batch_scheduler_config);
      if (!use_model_perf_backend) {
        event_driven_schedule_processor->SetScheduleBackend(
            std::make_shared<ContinuousBatchingBackend>(batch_scheduler_config));
      }
      event_driven_schedule_processor->SetCacheManagers(cache_managers_);

      auto hook_manager = std::make_shared<ScheduleHookManager>();

      // 创建 GlobalCacheFetcher 并注入到 GlobalCacheQueryHook
      {
        int device_count = context_->global->GetAttentionTensorParallelSize();
        std::shared_ptr<IpcFuncWrapper> fetcher_ipc = global_cache_connector_ ? ipc_func_wrapper_ : nullptr;
        // role 与节点身份一一对应:
        //   V1 standalone -> ""，V1 Prefill -> "prefill"，V1 Decode -> "decode"，
        //   pd_v2 Prefill -> "pd_v2_prefill"，pd_v2 Decode -> "pd_v2_decode"。
        const std::string fetcher_role = DetermineGlobalCacheFetcherRole(connector_config, status.OK());
        auto global_cache_fetcher = std::make_shared<GlobalCacheFetcher>(device_count, fetcher_ipc, fetcher_role);

        if (global_cache_connector_) {
          ipc_func_wrapper_->SetGlobalCacheQueryDoneCallableFetcher(global_cache_fetcher);
        }
        global_cache_fetcher->Start();

        // 创建全局 cache 查询 Hook（hook_idx=0，第一个添加的 hook）。
        // pd_v2 Decode 角色下让本 Hook 成为 block 分配的单一所有者: alloc 失败时挂起并等
        // block 释放后重试, 而非回退到 backend (下游 PdV2DecodeHook 假定 req 已分好 block)。
        const bool defer_admission_on_block_full = (fetcher_role == "pd_v2_decode");
        // master 引入 ENABLE_GLOBAL_CACHE_HOOK 开关，用于非 pd_v2 场景按需启用全局 cache hook；
        // pd_v2 Decode 必须创建该 hook（block 分配的唯一所有者），不受该开关影响。
        const char *enable_global_cache_hook_env = std::getenv("ENABLE_GLOBAL_CACHE_HOOK");
        const bool enable_global_cache_hook =
            (enable_global_cache_hook_env != nullptr && std::string(enable_global_cache_hook_env) == "1");
        if (enable_global_cache_hook || defer_admission_on_block_full) {
          auto global_cache_hook =
              std::make_shared<GlobalCacheQueryHook>(global_cache_fetcher, batch_scheduler_config,
                                                     /*hook_idx=*/0, defer_admission_on_block_full);
          hook_manager->AddHook(global_cache_hook);

          // PD decode 下 GlobalCacheQueryHook 是 block 分配的唯一所有者，recompute 必须经
          // hook 链重新分配而非直接回灌 prefilling_reqs（详见 HandleRecomputeRequest）。
          event_driven_schedule_processor->SetPdV2DecodeMode(defer_admission_on_block_full);

          KLLM_LOG_INFO << "GlobalCacheQueryHook created and added to event-driven scheduler, device_count="
                        << device_count << ", global_cache_enabled=" << (global_cache_connector_ != nullptr)
                        << ", defer_admission_on_block_full=" << defer_admission_on_block_full;
        } else {
          KLLM_LOG_INFO << "GlobalCacheQueryHook disabled (ENABLE_GLOBAL_CACHE_HOOK="
                        << (enable_global_cache_hook_env ? enable_global_cache_hook_env : "not set") << ")";
        }
      }

      // pd_v2 cross-node Prefill/Decode bootstrap.
      // 详见 BootstrapPdV2Connector：dlopen 后端 .so、创建 connector 并按角色注册
      // Decode hook / Prefill endpoint。非 pd_v2 Prefill/Decode 节点为 no-op。
      BootstrapPdV2Connector(event_driven_schedule_processor, hook_manager);

      // 创建 GlobalCachePutTracker，防止 put 进行中释放 block 导致 use-after-free
      if (global_cache_connector_) {
        int device_count = context_->global->GetAttentionTensorParallelSize();
        auto put_tracker = std::make_shared<GlobalCachePutTracker>(device_count, cache_managers_[0]);
        put_tracker->SetDpCacheManagers(cache_managers_);
        event_driven_schedule_processor->SetGlobalCachePutTracker(put_tracker);
        ipc_func_wrapper_->SetGlobalCachePutDoneCallableTracker(put_tracker);
        KLLM_LOG_INFO << "GlobalCachePutTracker created for event-driven scheduler, device_count=" << device_count;
      }

      event_driven_schedule_processor->SetScheduleHookManager(hook_manager);

      if (!use_model_perf_backend) {
        event_driven_schedule_processor->SetWorkloadBalancer(std::make_shared<BasicWorkloadBalancer>(
            attn_dp_size, static_cast<size_t>(batch_scheduler_config.max_batch_size),
            static_cast<size_t>(batch_scheduler_config.max_decode_tokens_per_req),
            static_cast<size_t>(batch_scheduler_config.max_step_token_num)));
      }

      // Create PpBatchDistributor for PP multi-batch request distribution.
      // ModelPerfBackend 不使用 PP 分发（自行控制 multi_batch_id=0）
      if (!use_model_perf_backend && batch_scheduler_config.max_pp_batch_num > 1) {
        auto distributor = std::make_shared<PpBatchDistributor>(batch_scheduler_config.pp_multibatch_wb_strategy);
        event_driven_schedule_processor->SetPpBatchDistributor(distributor);
        KLLM_LOG_INFO << "PpBatchDistributor created for event-driven scheduler, max_pp_batch_num="
                      << batch_scheduler_config.max_pp_batch_num
                      << ", strategy=" << static_cast<int>(batch_scheduler_config.pp_multibatch_wb_strategy);
      }

      schedule_processor_ = event_driven_schedule_processor;
      schedule_processor_->Initialize(nullptr);
    } else {
      auto batch_scheduler =
          std::make_shared<BatchScheduler>(batch_scheduler_config, runtime_config, schedule_output_pool_);
      for (size_t dp_id = 0; dp_id < context_->global->GetAttentionDataParallelSize(); ++dp_id) {
        batch_scheduler->SetCacheManager(cache_managers_[dp_id], dp_id);
      }

      // Set TransferEngine for batch scheduler if it exists
      if (transfer_engine_) {
        batch_scheduler->SetTransferEngine(transfer_engine_);
        KLLM_LOG_INFO << "TransferEngine set for BatchScheduler.";
      }

      // 创建 GlobalCachePutTracker 并注入 BatchScheduler 和 IPC callable。
      if (global_cache_connector_) {
        int device_count = context_->global->GetAttentionTensorParallelSize();
        auto put_tracker = std::make_shared<GlobalCachePutTracker>(device_count, cache_managers_[0]);
        std::dynamic_pointer_cast<BatchScheduler>(batch_scheduler)->SetGlobalCachePutTracker(put_tracker);
        ipc_func_wrapper_->SetGlobalCachePutDoneCallableTracker(put_tracker);
        KLLM_LOG_INFO << "GlobalCachePutTracker created and injected, device_count=" << device_count;
      }

      schedule_processor_ = std::make_shared<ScheduleProcessor>(batch_scheduler_config.enable_async,
                                                                batch_scheduler_config.max_pp_batch_num);
      schedule_processor_->Initialize(batch_scheduler);
    }

    if (transfer_engine_) {
      model_driver_->SetTransferEngine(transfer_engine_);
    }
    model_driver_->SetModelConfigAndRuntimeConfig(model_config, runtime_config);
    model_driver_->SetScheduleProcessor(schedule_processor_);

    // PD 分离：将 GlobalCache 角色传递给 ModelDriver，用于在调度时发送 prefill 通知
    if (global_cache_status.OK()) {
      model_driver_->SetGlobalCacheRole(global_cache_config.role);
    }

    // For locking with scheduler thread.
    ScheduleOutputParser::SetScheduleProcessor(schedule_processor_);
  }

  if (env_->IsReportVersion()) {
    VersionReporter::GetInstance().Init();
  }

  // Create executor runtime on executor process.
  // Get vocab info to pass to executor (already set in Environment earlier)
  VocabInfo vocab_info;
  Status vocab_status = env_->GetVocabInfo(vocab_info);
  if (!vocab_status.OK()) {
    KLLM_LOG_WARNING << "Failed to get vocab info for executor: " << vocab_status.GetMessage();
  } else {
    KLLM_LOG_INFO << "Passing vocab info to executor, vocab_size: " << vocab_info.vocab_size
                  << ", think_end_token_id: " << vocab_info.think_end_token_id;
  }
  STATUS_CHECK_RETURN(model_driver_->CreateExecutorRuntime(vocab_info));

  return Status();
}

Status InferenceEngine::DoWarmupRun() {
  if (std::getenv("DISABLE_WARMUP") != nullptr) {
    KLLM_LOG_DEBUG << "warmup is disabled";
    return Status();
  }

  // pd_v2 Decode 需先等到一个健康的 Prefill peer，否则 warmup 请求会因 CanAccept 失败而
  // 产生空输出（详见 WaitForPdV2DecodePeer）。非 pd_v2 Decode 节点为 no-op。
  WaitForPdV2DecodePeer();

  ModelConfig model_config;
  Status status = env_->GetModelConfig(model_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "No model config found for warmup run. " + status.ToString());
  }
  const bool is_embedding = model_config.is_embedding_model;

  RuntimeConfig runtime_config;
  env_->GetRuntimeConfig(runtime_config);
  int max_warmup_input_length = std::min(static_cast<int>(runtime_config.max_seq_len), 2048);
  if (std::getenv("MAX_WARMUP_INPUT_LENGTH") != nullptr) {
    max_warmup_input_length = std::stoi(std::getenv("MAX_WARMUP_INPUT_LENGTH"));
  }

  const int warmup_input_length_step = 64;
  std::vector<int> warmup_input_lengths;
  // Use <= to include the boundary length, ensuring warmup covers the full max input length.
  for (int i = warmup_input_length_step; i <= max_warmup_input_length; i += warmup_input_length_step) {
    warmup_input_lengths.push_back(i);
  }

  pybind11::gil_scoped_release release;
  KLLM_LOG_INFO << "Start to do warmup run, warmup req number=" << warmup_input_lengths.size();

  for (int input_length : warmup_input_lengths) {
    auto warmup_run_input = std::make_shared<KsanaPythonInput>();

    // Fill dummy tokens.
    std::vector<int> input_tokens(input_length, 0);
    for (int i = 0; i < input_length; ++i) {
      input_tokens[i] = i % 100;  // Fill with some dummy tokens.
    }
    warmup_run_input->input_tokens = input_tokens;
    warmup_run_input->sampling_config.ignore_eos = true;

    if (is_embedding) {
      // Embedding: only one prefill forward, no decode step.
      warmup_run_input->sampling_config.max_new_tokens = 1;
      // Set request_target with "layernorm" to trigger the embedding LmHead path.
      TargetDescribe target_desc;
      target_desc.token_reduce_mode = TokenReduceMode::GATHER_ALL;
      warmup_run_input->request_target["layernorm"] = target_desc;
    } else {
      // LLM: one context + one decode.
      warmup_run_input->sampling_config.max_new_tokens = 2;
    }

    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto req = std::make_shared<Request>(warmup_run_input, req_ctx);

    ConnectorConfig connector_config;
    env_->GetConnectorConfigs(connector_config);
    // NOTE(winminkong): To prevent PD disaggregate waiting in a loop for the configuration to be transmitted.
    if (connector_config.group_role == GroupRole::PREFILL) {
      req->req_id = -1;
      req->req_ids[0] = -1;
    }
    PdV2RuntimeConfig pd_v2_rt_warmup;
    if (env_->GetPdV2RuntimeConfig(pd_v2_rt_warmup).OK() && pd_v2_rt_warmup.is_pd_v2 &&
        pd_v2_rt_warmup.group_role == GroupRole::DECODE) {
      req->req_id = -1;
      req->req_ids[0] = -1;
    }

    req->waiter = std::make_shared<Waiter>(1);
    request_queue_.Write({Status(), req});
    req->waiter->Wait();

    // Verification.
    if (is_embedding) {
      if (req->response.find("layernorm") == req->response.end()) {
        KLLM_LOG_WARNING << "Embedding warmup (len=" << input_length
                         << ") did not produce 'layernorm' response, skipping verification";
      }
    } else {
      for (const auto &[output, req_logprobs, total_score] : req->output_group) {
        KLLM_CHECK_WITH_INFO(req->input_tokens.size() < output.size(),
                             "Warmup run generated empty output tokens. Inference run failed");
      }
    }
  }

  KLLM_LOG_INFO << "End to do warmup run";
  pybind11::gil_scoped_acquire acquire;
  return Status();
}

bool InferenceEngine::InitModelPerfBackend(
    std::shared_ptr<EventDrivenScheduleProcessor> event_driven_schedule_processor,
    const BatchSchedulerConfig &batch_scheduler_config) {
  const char *enable_model_perf_env = std::getenv("ENABLE_MODEL_PERF");
  const char *model_perf_config_env = std::getenv("MODEL_PERF_CONFIG");
  const bool use_model_perf = (enable_model_perf_env != nullptr && std::string(enable_model_perf_env) == "1") &&
                              (model_perf_config_env != nullptr && std::string(model_perf_config_env).size() > 0);

  if (!use_model_perf) {
    return false;
  }

  KLLM_LOG_INFO << "ModelPerfBackend activated. config=" << model_perf_config_env;

  ModelConfig model_config;
  Status model_config_status = env_->GetModelConfig(model_config);
  if (!model_config_status.OK()) {
    KLLM_THROW(fmt::format("ModelPerfBackend: failed to get ModelConfig: {}", model_config_status.GetMessage()));
  }
  if (model_config.vocab_size == 0) {
    KLLM_THROW("ModelPerfBackend: ModelConfig.vocab_size is 0");
  }

  RuntimeConfig runtime_config;
  Status runtime_config_status = env_->GetRuntimeConfig(runtime_config);
  if (!runtime_config_status.OK()) {
    KLLM_THROW(fmt::format("ModelPerfBackend: failed to get RuntimeConfig: {}", runtime_config_status.GetMessage()));
  }

  auto model_perf_backend =
      std::make_shared<ModelPerfBackend>(batch_scheduler_config, std::string(model_perf_config_env),
                                         model_config.vocab_size, runtime_config.attn_backend_config.block_token_num);
  event_driven_schedule_processor->SetScheduleBackend(model_perf_backend);

  // 使用专用 balancer（透传 attn_dp_group_id，不做轮询覆写）
  const size_t attn_dp_size = context_->global->GetAttentionDataParallelSize();
  auto model_perf_balancer = std::make_shared<ModelPerfWorkloadBalancer>(attn_dp_size);
  event_driven_schedule_processor->SetWorkloadBalancer(model_perf_balancer);

  // SetWorkloadBalancer 完成后，将 processor 注入 backend
  model_perf_backend->SetScheduleProcessor(event_driven_schedule_processor);
  model_perf_backend_ = model_perf_backend;
  return true;
}

std::string InferenceEngine::DetermineGlobalCacheFetcherRole(const ConnectorConfig &connector_config,
                                                             bool connector_config_ok) {
  // pd_v2 read-sites consult PdV2RuntimeConfig (process-internal) rather than
  // ConnectorConfig to keep V1 derivations from leaking — see K7.
  PdV2RuntimeConfig pd_v2_rt;
  const bool is_pd_v2 = env_->GetPdV2RuntimeConfig(pd_v2_rt).OK() && pd_v2_rt.is_pd_v2;
  if (is_pd_v2 && pd_v2_rt.group_role == GroupRole::PREFILL) {
    return "pd_v2_prefill";  // 全命中保留 1 token 走 prefill 流水线
  } else if (is_pd_v2 && pd_v2_rt.group_role == GroupRole::DECODE) {
    return "pd_v2_decode";  // 无 reject, peer Prefill 通过 RDMA 兜底
  } else if (connector_config_ok && connector_config.group_role == GroupRole::PREFILL) {
    return "prefill";  // max_new_tokens=1 + flip kDecode
  } else if (connector_config_ok && connector_config.group_role == GroupRole::DECODE) {
    return "decode";  // partial hit reject
  }
  return "";
}

void InferenceEngine::BootstrapPdV2Connector(const std::shared_ptr<EventDrivenScheduleProcessor> &schedule_processor,
                                             const std::shared_ptr<ScheduleHookManager> &hook_manager) {
  // Both Decode and Prefill engines dlopen the same backend .so; the role is
  // selected via PdV2ConnectorConfig.node_role at factory time. Decode side
  // adds PdV2DecodeHook to the EDS hook chain; Prefill side constructs
  // PdV2PrefillEndpoint and registers its incoming-request callback with the
  // connector. Executor-side RegisterKvPool lives in InferenceExecutor.
  // pd_v2 reads from PdV2RuntimeConfig (process-internal) rather than
  // ConnectorConfig to keep V1 derivations from leaking — see K7.
  PdV2RuntimeConfig pd_v2_rt;
  const bool pd_v2_rt_ok = env_->GetPdV2RuntimeConfig(pd_v2_rt).OK();
  if (!pd_v2_rt_ok || !pd_v2_rt.is_pd_v2 ||
      (pd_v2_rt.group_role != GroupRole::DECODE && pd_v2_rt.group_role != GroupRole::PREFILL)) {
    return;
  }

  // Config-mapping + dlopen/create/init are pushed down into the pd_v2 layer
  // (shared with inference_executor); the engine only keeps the scheduler /
  // endpoint wiring below, which needs engine-owned deps.
  const pd_v2::PdV2ConnectorConfig pd_cfg = pd_v2::BuildEngineConnectorConfig(
      pd_v2_rt, static_cast<uint32_t>(context_->global->GetAttentionTensorParallelSize()));
  pd_v2_connector_ = pd_v2::CreateAndInitConnector(pd_cfg, pd_v2_rt.redis_password, ipc_func_wrapper_.get(),
                                                   &pd_v2_dlopen_handle_, /*context_hint=*/"engine");

  if (pd_v2_rt.group_role == GroupRole::DECODE) {
    RegisterPdV2DecodeHook(schedule_processor, hook_manager, pd_cfg);
  } else {
    RegisterPdV2PrefillEndpoint(schedule_processor, pd_cfg);
  }
}

void InferenceEngine::RegisterPdV2DecodeHook(const std::shared_ptr<EventDrivenScheduleProcessor> &schedule_processor,
                                             const std::shared_ptr<ScheduleHookManager> &hook_manager,
                                             const pd_v2::PdV2ConnectorConfig &pd_cfg) {
  // pd_v2 Decode-side preempt path (swap removal): under memory pressure a
  // kv_from_remote req is preempted via recompute, which HandleRecomputeRequest
  // turns into a full PD restart (release local KV + re-fetch from a peer
  // Prefill through AddInferRequest). This works under any preempt_mode, so the
  // previous "must be SWAP" requirement no longer applies.
  //
  // LinUCB router (selector_policy == kLinUCB) needs this Decode engine's
  // per-step EWMA load for the LocalArmContext. Capture the EDS weakly (mirror
  // of the Prefill snapshot_fn): a call after teardown safely yields a zero
  // ScheduleLoadStat.
  std::weak_ptr<EventDrivenScheduleProcessor> weak_eds_decode = schedule_processor;
  std::function<ScheduleLoadStat()> load_stat_fn = [weak_eds_decode]() -> ScheduleLoadStat {
    // PdV2DecodeHook invokes this from inside DispatchEvent, where the EDS
    // thread already holds infer_request_mutex_. Use the unlocked reader; the
    // locking GetLocalLoad() would self-deadlock.
    if (auto eds = weak_eds_decode.lock()) return eds->GetLocalLoadUnlocked();
    return ScheduleLoadStat{};
  };
  auto pd_v2_hook =
      std::make_shared<pd_v2::PdV2DecodeHook>(pd_v2_connector_, /*hook_idx=*/1, pd_cfg.engine.local_complete_threshold,
                                              pd_cfg.engine.selector_policy, std::move(load_stat_fn));
  hook_manager->AddHook(pd_v2_hook);
  KLLM_LOG_INFO << "pd_v2 Decode hook registered, cluster=" << pd_cfg.cluster_name
                << " inference_addr=" << pd_cfg.inference_addr
                << " local_complete_threshold=" << pd_cfg.engine.local_complete_threshold
                << " selector_policy=" << static_cast<int>(pd_cfg.engine.selector_policy);
}

void InferenceEngine::RegisterPdV2PrefillEndpoint(
    const std::shared_ptr<EventDrivenScheduleProcessor> &schedule_processor, const pd_v2::PdV2ConnectorConfig &pd_cfg) {
  if (local_endpoint_ == nullptr) {
    KLLM_THROW("pd_v2 Prefill role requires InferenceEngine::SetLocalEndpoint before Initialize");
  }
  pd_v2::PdV2PrefillEndpoint::Deps endpoint_deps;
  endpoint_deps.local_endpoint = local_endpoint_;
  endpoint_deps.connector = pd_v2_connector_;
  endpoint_deps.prefill_attn_tp = static_cast<uint32_t>(context_->global->GetAttentionTensorParallelSize());

  // pd_v2 Prefill load-snapshot provider: produces a PeerHealth that reflects
  // this Prefill engine's real load. Fed to BOTH snapshot paths — the
  // per-completion reply (endpoint_deps.snapshot_fn, consumed in
  // pd_v2_prefill_completion.cpp) and the periodic heartbeat (connector
  // SetSnapshotProvider → BroadcastSnapshot). cache_managers_ is copied by
  // value (vector of shared_ptr); the EDS processor is captured weakly so a
  // snapshot firing after engine teardown is a harmless no-op (load fields
  // stay 0).
  std::weak_ptr<EventDrivenScheduleProcessor> weak_eds = schedule_processor;
  auto snapshot_cache_mgrs = cache_managers_;
  const std::string snapshot_server_name = pd_cfg.inference_addr;
  std::function<pd_v2::PeerHealth()> snapshot_fn = [weak_eds, snapshot_cache_mgrs,
                                                    snapshot_server_name]() -> pd_v2::PeerHealth {
    pd_v2::PeerHealth h;
    h.server_name = snapshot_server_name;
    h.healthy = true;
    h.last_snapshot_ms = static_cast<uint64_t>(ProfileTimer::GetCurrentTimeInMs());
    uint64_t free_blocks = 0;
    // swap removed: no swapout-in-flight blocks, so future-free is always 0.
    uint64_t future_free_blocks = 0;
    for (const auto &cm : snapshot_cache_mgrs) {
      if (cm != nullptr) {
        free_blocks += cm->GetUsableBlockNumber();
      }
    }
    h.free_kv_blocks = static_cast<uint32_t>(free_blocks);
    h.future_free_kv_blocks = static_cast<uint32_t>(future_free_blocks);
    if (auto eds = weak_eds.lock()) {
      const ScheduleLoadStat ld = eds->GetLocalLoad();
      h.waiting_reqs = ld.waiting_reqs;
      h.running_reqs = ld.running_reqs;
      h.pending_prefill_tokens = ld.pending_prefill_tokens;
      h.step_token_num = ld.step_token_num;
      h.step_latency_us = ld.step_latency_us;
    }
    return h;
  };
  endpoint_deps.snapshot_fn = snapshot_fn;

  pd_v2_prefill_endpoint_ = std::make_shared<pd_v2::PdV2PrefillEndpoint>(std::move(endpoint_deps));
  pd_v2_prefill_endpoint_->Register();
  // Periodic heartbeat path: the connector's BroadcastSnapshot uses the same
  // provider to fill PeerHealth load fields each cycle.
  pd_v2_connector_->SetSnapshotProvider(snapshot_fn);
  {
    std::weak_ptr<pd_v2::PdV2PrefillEndpoint> weak_endpoint = pd_v2_prefill_endpoint_;
    ipc_func_wrapper_->SetPdV2BarrierDoneCallback([weak_endpoint](int64_t req_id, uint8_t status) {
      if (auto p = weak_endpoint.lock()) p->OnBarrierDone(req_id, status);
    });
  }
  KLLM_LOG_INFO << "pd_v2 Prefill endpoint registered, cluster=" << pd_cfg.cluster_name
                << " inference_addr=" << pd_cfg.inference_addr;
}

void InferenceEngine::WaitForPdV2DecodePeer() {
  // pd_v2 DECODE warmup synthesizes a request that has to clear CanAccept;
  // CanAccept needs at least one healthy Prefill peer in the local
  // PeerSelector, which only populates after Prefill's heartbeat
  // (snapshot_push_interval_ms, default 1s) sends us its first PrefillSnapshot.
  // If Decode boots faster than the first snapshot arrives we hit `Warmup run
  // generated empty output tokens`. Poll CanAccept(probe_req=-2)+Release until
  // it succeeds, capped by an env-overridable timeout.
  PdV2RuntimeConfig pd_v2_rt;
  if (!(env_->GetPdV2RuntimeConfig(pd_v2_rt).OK() && pd_v2_rt.is_pd_v2 && pd_v2_rt.group_role == GroupRole::DECODE &&
        pd_v2_connector_ != nullptr)) {
    return;
  }

  int wait_ms = 30000;
  if (const char *env = std::getenv("PD_V2_WARMUP_PEER_WAIT_MS")) {
    try {
      wait_ms = std::max(0, std::stoi(env));
    } catch (...) {
    }
  }
  if (pd_v2::WaitForHealthyPeer(pd_v2_connector_, wait_ms)) {
    KLLM_LOG_INFO << "pd_v2 DECODE warmup: PeerSelector has healthy peer, proceeding";
  } else {
    KLLM_LOG_WARNING << "pd_v2 DECODE warmup: timed out (" << wait_ms
                     << "ms) waiting for first PrefillSnapshot; warmup will likely fail. "
                     << "Check Prefill side / Redis connectivity.";
  }
}

Status InferenceEngine::Start() {
  // NOTE(jinxcwu)
  // 在处理请求之前初始化 host 内存池。
  // 这确保了在初始化阶段 host tensor 使用动态分配，
  // 而在请求处理阶段 host tensor 使用内存池分配。
  BlockManagerConfig block_mgr_cfg;
  env_->GetBlockManagerConfig(block_mgr_cfg);
  HostMemoryPool::GetInstance()->Initialize(block_mgr_cfg.host_pool_size_gb);

  // Initialize cached block tree only for chief node.
  if (!context_->global->IsDistributedWorker()) {
    for (auto &cache_manager : cache_managers_) {
      cache_manager->InitializeCachedBlocks();
    }
  }

  // Start batch manager.
  model_driver_->Start();

  // Start TransferEngine if needed
  if (transfer_engine_) {
    transfer_engine_->Start();
    KLLM_LOG_INFO << "TransferEngine started for layer progress handling.";
  }

  if (!context_->global->IsStandalone()) {
    distributed_control_coordinator_->Barrier();
    distributed_control_coordinator_->Frozen();
    KLLM_LOG_INFO << "All Nodes are ready, ready to server";
  }
#ifndef ENABLE_ACL
  // ModelPerfBackend 模式：执行性能测试（阻塞直到所有 batch 完成）；否则执行正常 warmup。
  if (model_perf_backend_ != nullptr) {
    KLLM_LOG_INFO << "ModelPerfBackend active: starting performance profiling (replacing DoWarmupRun)";
    model_perf_backend_->StartPerf();
    KLLM_LOG_INFO << "ModelPerfBackend: profiling complete, exiting";
    std::cout << "ModelPerfBackend: profiling complete, exiting." << std::endl;
    _exit(0);
  } else if (!context_->global->IsDistributedWorker()) {
    DoWarmupRun();
  }
#endif

  return Status();
}

Status InferenceEngine::Stop() {
  if (terminated_) {
    return Status();
  }

  terminated_ = true;
  request_queue_.Write({Status(RET_REQUEST_TERMINATED), nullptr});

  request_queue_.Close();

  // Stop TransferEngine if needed
  if (transfer_engine_) {
    transfer_engine_->Stop();
    KLLM_LOG_INFO << "TransferEngine stopped.";
  }

  // 释放 global cache connector，shared_ptr 析构时自动调用 Shutdown 和 delete。
  if (global_cache_connector_) {
    global_cache_connector_.reset();
    KLLM_LOG_INFO << "GlobalCacheConnector released.";
  }

  // Destroy the tokenizer.
  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();
  if (!context_->global->IsStandalone()) {
    KLLM_LOG_INFO << "Destroy distributed coordinator.";
    distributed_control_coordinator_->DestroyCluster();
    distributed_control_coordinator_ = nullptr;
  }

  // Stop model driver and all its threads.
  model_driver_->Stop();

  // Clear batch scheduler
  KLLM_LOG_INFO << "Destroy batch scheduler.";

  if (env_->IsReportVersion()) {
    KLLM_LOG_INFO << "Stop version reporter.";
    VersionReporter::GetInstance().StopReporting();
    VersionReporter::GetInstance().Destroy();
  }

  KLLM_LOG_INFO << "Destroy Profiler";
  DestroyProcessProfiler();

  KLLM_LOG_INFO << "The Inference Engine has stopped.";
  return Status();
}

}  // namespace ksana_llm
