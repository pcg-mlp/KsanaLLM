/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "engine/inference_engine.h"

#include "batch_scheduler/batch_scheduler.h"
#include "batch_scheduler/event_driven_scheduler/continuous_batching_backend.h"
#include "batch_scheduler/event_driven_scheduler/event_driven_schedule_processor.h"
#include "batch_scheduler/event_driven_scheduler/global_cache_query_hook.h"
#include "batch_scheduler/event_driven_scheduler/schedule_hook_manager.h"
#include "batch_scheduler/event_driven_scheduler/workload_balance/basic_workload_balancer.h"
#include "batch_scheduler/global_cache_fetcher.h"
#include "batch_scheduler/global_cache_put_tracker.h"
#include "batch_scheduler/schedule_processor.h"
#include "cache_manager/cache_manager_factory.h"
#include "configure/environment.h"
#include "connector/connector.h"
#include "engine/model_driver.h"
#include "executor/layer_progress_tracker.h"
#include "global_cache_connector/global_cache_connector_factory.h"
#include "ipc/functions/report_layer_progress.h"
#include "periphery/version_reporter.h"
#include "profiler/reporter.h"
#include "runtime/draft_generator/trie_generator.h"
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
  HostMemoryPool::GetInstance()->Destroy();
  KLLM_LOG_DEBUG << "InferenceEngine destroyed.";
}

void InferenceEngine::SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper) {
  ipc_func_wrapper_ = ipc_func_wrapper;
  model_driver_->SetIpcFuncWrapper(ipc_func_wrapper_);
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

  // Get block size config of specific layers, runtime_config is changed, get it again.
  STATUS_CHECK_RETURN(env_->InitializeBlockManagerConfig());
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

  ProfilerConfig profiler_config;
  STATUS_CHECK_RETURN(env_->GetProfilerConfig(profiler_config));
  profiler_config.process_role = "engine";
  InitProcessProfiler(profiler_config);

  // Create batch scheduler.
  BatchSchedulerConfig batch_scheduler_config;
  if (!context_->global->IsDistributedWorker()) {
    STATUS_CHECK_RETURN(env_->GetBatchSchedulerConfig(batch_scheduler_config));
    batch_scheduler_config.max_batch_size = runtime_config.max_batch_size;
    batch_scheduler_config.max_vocab_size = model_config.vocab_size;
    KLLM_LOG_DEBUG << "Batch Scheduler Config Max Batch Size= " << batch_scheduler_config.max_batch_size
                   << ", Max Vocab Size= " << batch_scheduler_config.max_vocab_size
                   << ", max_pp_batch_num=" << batch_scheduler_config.max_pp_batch_num;
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

  // Create cache manager.
  if (!context_->global->IsDistributedWorker()) {
    std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = std::make_shared<MemoryAllocator>();
    // Pass ipc_func_wrapper_ to BlockAllocatorManager for engine-side BatchSwap IPC support
    BlockAllocatorManager block_allocator_manager(true, block_allocator_manager_config, memory_allocator_, context_,
                                                  nullptr, nullptr, ipc_func_wrapper_);

    CacheManagerConfig cache_manager_config;
    STATUS_CHECK_RETURN(env_->GetCacheManagerConfig(cache_manager_config));

    KLLM_LOG_INFO << "InferenceEngine create cache manager.";
    for (size_t dp_id = 0; dp_id < context_->global->GetAttentionDataParallelSize(); ++dp_id) {
      std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group =
          block_allocator_manager.GetBlockAllocatorGroup(dp_id);
      cache_managers_.emplace_back(
          CacheManagerFactory::CreateCacheManager(cache_manager_config, block_allocator_group));
    }
  }

  // Create block manager on executors.
  STATUS_CHECK_RETURN(model_driver_->CreateBlockManager(block_allocator_manager_config));

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

  // Create TransferEngine instance first (before batch scheduler)
  ConnectorConfig connector_config;
  Status status = env_->GetConnectorConfigs(connector_config);
  if (status.OK() && connector_config.group_role != GroupRole::NONE) {
    transfer_engine_ = std::make_shared<TransferEngine>();
    // Set IpcFuncWrapper to TransferEngine (which will set it to Connector)
    transfer_engine_->SetIpcFuncWrapper(ipc_func_wrapper_);
    transfer_engine_->Initialize(connector_config.group_role);
    auto layer_progress_queue = std::make_shared<BlockingQueue<LayerProgressInfo>>();
    ipc_func_wrapper_->SetLayerProgressRecvQueue(layer_progress_queue);
    transfer_engine_->SetLayerProgressRecvQueue(layer_progress_queue);
    KLLM_LOG_INFO << "TransferEngine created and initialized.";
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

      event_driven_schedule_processor->SetScheduleBackend(
          std::make_shared<ContinuousBatchingBackend>(batch_scheduler_config));
      event_driven_schedule_processor->SetCacheManagers(cache_managers_);

      auto hook_manager = std::make_shared<ScheduleHookManager>();

      // 创建 GlobalCacheFetcher 并注入到 GlobalCacheQueryHook
      {
        int device_count = context_->global->GetAttentionTensorParallelSize();
        std::shared_ptr<IpcFuncWrapper> fetcher_ipc = global_cache_connector_ ? ipc_func_wrapper_ : nullptr;
        auto global_cache_fetcher =
            std::make_shared<GlobalCacheFetcher>(device_count, fetcher_ipc, global_cache_config.role);

        if (global_cache_connector_) {
          ipc_func_wrapper_->SetGlobalCacheQueryDoneCallableFetcher(global_cache_fetcher);
        }
        global_cache_fetcher->Start();

        // 创建全局 cache 查询 Hook（hook_idx=0，第一个添加的 hook）
        auto global_cache_hook =
            std::make_shared<GlobalCacheQueryHook>(global_cache_fetcher, batch_scheduler_config, /*hook_idx=*/0);
        hook_manager->AddHook(global_cache_hook);

        KLLM_LOG_INFO << "GlobalCacheQueryHook created and added to event-driven scheduler, device_count="
                      << device_count << ", global_cache_enabled=" << (global_cache_connector_ != nullptr);
      }

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

      event_driven_schedule_processor->SetWorkloadBalancer(std::make_shared<BasicWorkloadBalancer>(
          attn_dp_size, static_cast<size_t>(batch_scheduler_config.max_batch_size),
          static_cast<size_t>(batch_scheduler_config.max_decode_tokens_per_req),
          static_cast<size_t>(batch_scheduler_config.max_step_token_num)));

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

  // Synchronize EP communicator config after executor runtime is created
  if (!context_->global->IsExpertParallelStandalone()) {
    STATUS_CHECK_RETURN(model_driver_->SynchronizeEPCommunicationConfig(distributed_control_coordinator_));
  }

  return Status();
}

Status InferenceEngine::DoWarmupRun() {
  if (std::getenv("DISABLE_WARMUP") != nullptr) {
    KLLM_LOG_DEBUG << "warmup is disabled";
    return Status();
  }

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
  KLLM_LOG_INFO << "Start to do warmup run";

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
      req->kv_comm_request_id = -1;
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
  // Start warmup run (auto-detects embedding vs LLM mode).
  if (!context_->global->IsDistributedWorker()) {
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
