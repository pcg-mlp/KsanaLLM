/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/service/inference_engine.h"

#include <chrono>
#include <memory>
#include <thread>

#include "ksana_llm/cache_manager/cache_manager_factory.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/periphery/version_reporter.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tokenizer.h"
#include "ksana_llm/utils/waiter.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/runtime/cuda_graph_runner.h"
#endif

namespace ksana_llm {

InferenceEngine::InferenceEngine(Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue)
    : request_queue_(request_queue) {
  Initialize();
}

InferenceEngine::~InferenceEngine() { KLLM_LOG_DEBUG << "InferenceEngine destroyed."; }

Status InferenceEngine::Initialize() {
  std::shared_ptr<Environment> env = Singleton<Environment>::GetInstance();
  if (!env) {
    return Status(RET_INVALID_ARGUMENT, "The Environment is nullptr.");
  }

  // Environment is must be initialized befroe context.
  KLLM_LOG_DEBUG << "Get tensor parallel: " << env->GetTensorParallelSize();
  context_.reset(new Context(env->GetTensorParallelSize(), env->GetPipeLineParallelSize()));

  // Load model configs.
  std::unordered_map<std::string, ModelConfig> model_configs;
  Status status = env->GetModelConfigs(model_configs);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get model configs error:" + status.ToString());
  }
  if (model_configs.empty()) {
    return Status(RET_INVALID_ARGUMENT, "No model config found.");
  }
  KLLM_LOG_DEBUG << "Get model instance size: " << model_configs.size();

  // Initialize schedule output and hidden unit buffer pool.
  // Must be called after block manager is set.
  InitializeScheduleOutputPool();
  InitializeHiddenUnitBufferPool();

  // Only for distributed mode.
  if (!context_->IsStandalone()) {
    distributed_coordinator_ = std::make_shared<DistributedCoordinator>(
        context_, GetPacketObject, GetScheduleOutputPool(), GetHiddenUnitBufferPool(), env);

    KLLM_LOG_INFO << "Initialize distributed coordinator.";
    distributed_coordinator_->InitializeCluster();
  }

  // Set model layers for standalone mode, assume only one model now.
  KLLM_LOG_INFO << "InferenceEngine IsStandalone:" << context_->IsStandalone();
  if (context_->IsStandalone()) {
    ModelConfig model_config = model_configs.begin()->second;

    PipelineConfig pipeline_config;
    Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
    pipeline_config.lower_layer_idx = 0;
    pipeline_config.upper_layer_idx = model_config.num_layer - 1;
    Singleton<Environment>::GetInstance()->SetPipelineConfig(pipeline_config);
    KLLM_LOG_INFO << "InferenceEngine Set layer range:[" << pipeline_config.lower_layer_idx << ", "
                  << pipeline_config.upper_layer_idx << "].";
  } else {
    KLLM_LOG_INFO << "Start to synchronize node layers.";
    distributed_coordinator_->SynchronizeNodeLayers();

    PipelineConfig pipeline_config;
    Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
    KLLM_LOG_INFO << "InferenceEngine Synchronize layer range:[" << pipeline_config.lower_layer_idx << ", "
                  << pipeline_config.upper_layer_idx << "].";
  }

  // Get block manager config of specific layers.
  status = env->InitializeBlockManagerConfig();
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Initialize block manager config error:" + status.ToString());
  }

  // Initialize global block manager.
  BlockManagerConfig block_manager_config;
  status = env->GetBlockManagerConfig(block_manager_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get block manager config error:" + status.ToString());
  }
  block_manager_ = new BlockManager(block_manager_config, context_);
  SetBlockManager(block_manager_);

  ProfilerConfig profiler_config;
  status = env->GetProfilerConfig(profiler_config);
  Singleton<Profiler>::GetInstance()->Init(profiler_config);

  size_t max_batch_size = 0;
  size_t max_vocab_size = 0;
  for (auto &[model_name, model_config] : model_configs) {
    max_batch_size = std::max(max_batch_size, (size_t)model_config.max_batch_size);
    max_vocab_size = std::max(max_vocab_size, (size_t)model_config.vocab_size);
  }

  // Create batch manager.
  BatchSchedulerConfig batch_scheduler_config;
  status = env->GetBatchSchedulerConfig(batch_scheduler_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get batch manager config error:" + status.ToString());
  }

  batch_scheduler_config.max_batch_size = max_batch_size;
  batch_scheduler_config.max_vocab_size = max_vocab_size;
  KLLM_LOG_DEBUG << "Batch Scheduler Config Max Batch Size = " << max_batch_size
                 << " Max Vocab Size = " << max_vocab_size;
  batch_manager_ = std::make_shared<BatchManager>(context_);

  // Register model instance.
  for (auto &[model_name, model_config] : model_configs) {
    // Update pipeline_config first, and then load model.
    std::shared_ptr<ModelInstance> model_instance = std::make_shared<ModelInstance>(model_config, context_);
    model_instance->Load();

    // Register model instance.
    model_instances_.push_back(model_instance);
    batch_manager_->RegisterModelInstance(model_instance);

    // Register to data hub.
    SetModelInstance(model_name, model_instance);
  }

  // Update block manager after model loading successfully
  BlockManagerConfig update_block_manager_config;
  env->GetBlockManagerConfig(update_block_manager_config);
  block_manager_->UpdateConfig(update_block_manager_config);

  // Create cache manager.
  CacheManagerConfig cache_manager_config;
  status = env->GetCacheManagerConfig(cache_manager_config);
  cache_manager_ = CacheManagerFactory::CreateCacheManager(cache_manager_config);

  // Initialize tokenzier
  tokenizer_ = std::make_shared<Tokenizer>();
  tokenizer_->InitTokenizer(model_instances_[0]->GetModelConfig().path);

  // Create batch scheduler.
  batch_scheduler_ = std::make_shared<BatchScheduler>(batch_scheduler_config, context_->GetTensorParallelSize());
  batch_scheduler_->SetCacheManager(cache_manager_);
  batch_scheduler_->SetTokenizer(tokenizer_);

  // Create llm runtime
  llm_runtime_ = std::make_shared<LlmRuntime>(batch_scheduler_config, context_);
  llm_runtime_->SetCacheManager(cache_manager_);

  batch_manager_->SetBatchScheduler(batch_scheduler_);
  batch_manager_->SetLlmRuntime(llm_runtime_);

  if (Singleton<Environment>::GetInstance()->IsReportVersion()) {
    VersionReporter::GetInstance().Init();
  }
  return Status();
}

Status InferenceEngine::HandleRequest(std::shared_ptr<Request> &req) {
  opentelemetry::common::KeyValueIterableView<std::unordered_map<std::string, std::string>> attributes(*req->req_ctx);
  REPORT_COUNTER(forward_req_total_num, static_cast<size_t>(1), attributes);
  REPORT_METRIC(metric_input_tokens_num, req->input_tokens.size(), attributes);

  Status handle_req_status = batch_manager_->Enqueue(req);
  if (!handle_req_status.OK()) {
    REPORT_COUNTER(forward_req_error_num, static_cast<size_t>(1), attributes);
    return handle_req_status;
  }
  return Status();
}

Status InferenceEngine::HandleLoop() {
  KLLM_LOG_DEBUG << "Start handler";

  while (!terminated_) {
    std::pair<Status, std::shared_ptr<Request>> req_pair;
    request_queue_.Read(&req_pair);
    if (terminated_) {
      break;
    }

    Status status = req_pair.first;
    if (status.GetCode() == RET_TERMINATED) {
      break;
    }

    std::shared_ptr<Request> req = req_pair.second;
    if (req) {
      HandleRequest(req);
    }
  }

  return Status();
}

Status InferenceEngine::StartHandler() {
  handle_thread_ = std::thread(&InferenceEngine::HandleLoop, this);
  return Status();
}

Status InferenceEngine::DoWarmupRun() {
  if (std::getenv("DISABLE_WARMUP") != nullptr) {
    KLLM_LOG_DEBUG << "warmup is disabled";
    return Status();
  }
  pybind11::gil_scoped_release release;
  KLLM_LOG_INFO << "Start to do warmup run";
  auto warmup_run_input = std::make_shared<KsanaPythonInput>();
  // Prepare the warm up input.
  warmup_run_input->input_tokens = std::vector<int>{1};
  // Warm up with one context and one decoding.
  warmup_run_input->sampling_config.max_new_tokens = 2;
  warmup_run_input->sampling_config.ignore_eos = true;

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto req = std::make_shared<Request>(warmup_run_input, req_ctx);
  req->waiter = std::make_shared<Waiter>(1);
  HandleRequest(req);

  // Wait the warm up.
  req->waiter->Wait();

  for (const auto &[output, req_logprobs, total_score] : req->output_group) {
    KLLM_CHECK_WITH_INFO(req->input_tokens.size() + req->padded_size < output.size(),
                         "Ksana warmup run generate empty output tokens. Warmup inference run failed");
  }
  KLLM_LOG_INFO << "End to do warmup run";
  pybind11::gil_scoped_acquire acquire;
  return Status();
}

#ifdef ENABLE_CUDA
Status InferenceEngine::CudaGraphCapture() {
  if (!Singleton<Environment>::GetInstance()->IsCudagraphEnabled()) {
    KLLM_LOG_INFO << "cuda graph capture is disabled";
    return Status();
  }
  pybind11::gil_scoped_release release;
  auto cuda_graph_builder = std::make_shared<CudaGraphBuilder>();
  // currently support cudagraph bs=1,2,3
  // for VRAM usage consideration (each cudagraph for specific bs takes 15~25mb VRAM)
  const int capture_batch_sizes = 3;
  size_t max_batch_size =
      cuda_graph_builder->GetMaxGraphBatchSize(model_instances_[0]->GetModelConfig().max_batch_size);
  std::vector<int> batch_size_capture_list;
  batch_size_capture_list.reserve(cuda_graph_builder->GetBatchSizeCaptureList().size());
  std::copy_if(cuda_graph_builder->GetBatchSizeCaptureList().begin(),
               cuda_graph_builder->GetBatchSizeCaptureList().end(), std::back_inserter(batch_size_capture_list),
               [&](size_t bs) { return bs <= max_batch_size; });
  std::vector<int> input_tokens(batch_size_capture_list.back(), 0);
  for (int batchsize = 1; batchsize <= capture_batch_sizes; ++batchsize) {
    KLLM_LOG_INFO << "start to capture graph: batchsize: " << batchsize;
    auto warmup_run_input = std::make_shared<KsanaPythonInput>();
    warmup_run_input->input_tokens = std::vector<int>(input_tokens.begin(), input_tokens.begin() + capture_batch_sizes);
    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto req = std::make_shared<Request>(warmup_run_input, req_ctx);
    for (int i = 0; i <= batchsize; ++i) {
      std::vector<int> output_tuple_;
      output_tuple_.emplace_back(std::get<0>(req->output_group[0])[0]);
      std::vector<std::vector<std::pair<int, float>>> req_logprobs;
      auto req_tuple = std::make_tuple(output_tuple_, req_logprobs, std::get<2>(req->output_group[0]));
      req->output_group.emplace_back(req_tuple);
    }
    req->is_cudagraph_capture_request = true;
    // we only need one context decode + one decode process
    req->sampling_config.max_new_tokens = 2;
    req->waiter = std::make_shared<Waiter>(1);
    HandleRequest(req);
    req->waiter->Wait();
    KLLM_LOG_INFO << "end to capture graph batchsize: " << batchsize;
  }
  pybind11::gil_scoped_acquire acquire;
  return Status();
}
#endif

Status InferenceEngine::Start() {
  if (!context_->IsStandalone()) {
    KLLM_LOG_INFO << "Start to synchronize cache block num.";
    distributed_coordinator_->SynchronizeCacheBlockNum();

    PipelineConfig pipeline_config;
    Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
    KLLM_LOG_INFO << "InferenceEngine Synchronize device block num " << pipeline_config.device_block_num
                  << ", host block_num " << pipeline_config.host_block_num << ".";
  }

  // Reset block num via device memory usage.
  block_manager_->ResetPreAllocatedBlocks();

  // Check block number, the block number is determined after all models loaded.
  BatchSchedulerConfig batch_scheduler_config;
  Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
  KLLM_CHECK_WITH_INFO((block_manager_->GetDeviceFreeBlockNumber() * block_manager_->GetBlockTokenNum()) >=
                           (batch_scheduler_config.max_token_len),
                       FormatStr("Total device block_num(%d) * block_token_size(%d) must large than max_token_len(%d).",
                                 block_manager_->GetDeviceFreeBlockNumber(), block_manager_->GetBlockTokenNum(),
                                 batch_scheduler_config.max_token_len));

  // Initialize cached block tree only for chief node.
  if (context_->IsChief()) {
    cache_manager_->InitializeCachedBlocks();
  }

  // Start batch manager.
  batch_manager_->Start();

  // Start service handler.
  if (context_->IsChief()) {
    StartHandler();
  }

#ifndef ENABLE_ACL
  // Start warmup run
  if (context_->IsChief()) {
    DoWarmupRun();
  }
#endif

#ifdef ENABLE_CUDA
  CudaGraphCapture();
#endif

  return Status();
}

Status InferenceEngine::Stop() {
  if (terminated_) {
    return Status();
  }

  terminated_ = true;

  request_queue_.Write({Status(RET_TERMINATED), nullptr});

  if (handle_thread_.joinable()) {
    handle_thread_.join();
  }

  request_queue_.Close();

  // Wait all request done.
  KLLM_LOG_INFO << "Waiting all running request.";
  Status status = batch_manager_->WaitAllDone();
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Wait all requests done error:" << status.ToString();
  }

  if (!context_->IsStandalone()) {
    KLLM_LOG_INFO << "Destroy distributed coordinator.";
    distributed_coordinator_->DestroyCluster();
    distributed_coordinator_ = nullptr;
  }

  // Stop the batch manger.
  KLLM_LOG_INFO << "Stop batch manager.";
  batch_manager_->Stop();
  batch_manager_ = nullptr;
  llm_runtime_ = nullptr;

  // Destroy all model instances.
  KLLM_LOG_INFO << "Destroy model instances.";
  DestroyModelInstance();
  model_instances_.clear();

  // Clear batch scheduler
  KLLM_LOG_INFO << "Destroy batch scheduler.";
  batch_scheduler_.reset();

  // Destroy schedule output and hidden unit buffer pool.
  DestroyScheduleOutputPool();
  DestroyHiddenUnitBufferPool();

  // Clear model instance.
  ModelInstance::Reset();

  if (Singleton<Environment>::GetInstance()->IsReportVersion()) {
    KLLM_LOG_INFO << "Stop version reporter.";
    VersionReporter::GetInstance().StopReporting();
    VersionReporter::GetInstance().Destroy();
  }

  KLLM_LOG_INFO << "Destroy block manager.";
  if (block_manager_) {
    delete block_manager_;
    block_manager_ = nullptr;
  }

  KLLM_LOG_INFO << "The Inference Engine has stopped.";
  return Status();
}

}  // namespace ksana_llm
