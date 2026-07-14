/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "engine/model_driver.h"

#include <torch/csrc/autograd/python_variable.h>

#include "distributed/control_message.h"

namespace ksana_llm {

ModelDriver::ModelDriver(Channel<std::pair<Status, std::shared_ptr<Request>>>& request_queue,
                         std::shared_ptr<ScheduleOutputPool> schedule_output_pool,
                         std::shared_ptr<SamplingOutputPool> sampling_output_pool, std::shared_ptr<Context> context)
    : request_queue_(request_queue), context_(context) {
  schedule_output_pool_ = schedule_output_pool;
  schedule_output_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<ScheduleOutput>>>();

  sampling_output_pool_ = sampling_output_pool;
  if (sampling_output_pool_) {
    sampling_output_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<SamplingOutput>>>();
  }
}

void ModelDriver::SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper) {
  ipc_func_wrapper_ = ipc_func_wrapper;
}

Status ModelDriver::InitDeviceCommunication() { return ipc_func_wrapper_->InitDeviceCommunication(); }

Status ModelDriver::InitCustomCommunication() { return ipc_func_wrapper_->InitCustomCommunication(); }

Status ModelDriver::InitMulticastCommunication() { return ipc_func_wrapper_->InitMulticastCommunication(); }

void ModelDriver::SetTransferEngine(std::shared_ptr<TransferEngine> transfer_engine) {
  transfer_engine_ = transfer_engine;
}

Status ModelDriver::GetDeviceMemory(size_t* device_free, size_t* device_total) {
  return ipc_func_wrapper_->GetDeviceMemory(device_free, device_total);
}

Status ModelDriver::LoadWeight() { return ipc_func_wrapper_->LoadWeight(); }

Status ModelDriver::LoadModel() { return ipc_func_wrapper_->LoadModel(); }

Status ModelDriver::ResetPipelineConfig(const PipelineConfig& pipeline_config) {
  return ipc_func_wrapper_->ResetPipelineConfig(pipeline_config);
}

Status ModelDriver::ConnectDataChannels(
    std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator) {
  const char* use_tcp_data_channel = std::getenv("USE_TCP_DATA_CHANNEL");
  if (use_tcp_data_channel && strcmp(use_tcp_data_channel, "1") == 0) {
    std::vector<HostPort> executor_host_ports;

    if (context_->global->IsDistributedWorker()) {
      KLLM_LOG_INFO << "Start to listen tcp data channel.";
      ipc_func_wrapper_->ListenTcpDataChannel(executor_host_ports);
      for (size_t i = 0; i < executor_host_ports.size(); ++i) {
        KLLM_LOG_INFO << "Executor id:" << i << ", host_port:" << executor_host_ports[i].host << ":"
                      << executor_host_ports[i].port;
      }
    }

    KLLM_LOG_INFO << "Start to sync tcp downstream data channel.";
    std::vector<HostPort> executor_downstream_host_ports;
    distributed_control_coordinator->SynchronizeTcpDataPort(executor_host_ports, executor_downstream_host_ports);

    if (!context_->global->IsDistributedLastWorker()) {
      for (size_t i = 0; i < executor_downstream_host_ports.size(); ++i) {
        KLLM_LOG_INFO << "Executor id:" << i << ", downstream:" << executor_downstream_host_ports[i].host << ":"
                      << executor_downstream_host_ports[i].port;
      }

      KLLM_LOG_INFO << "Start to connect tcp downstream data channel.";
      ipc_func_wrapper_->ConnectTcpDataChannel(executor_downstream_host_ports);
    }

    distributed_control_coordinator->Barrier();
    KLLM_LOG_INFO << "ModelDriver::ConnectDataChannels finished.";
#ifdef ENABLE_CUDA
  } else {
    ncclUniqueId nccl_unique_id;
    distributed_control_coordinator->SynchronizeNcclUniqueId(nccl_unique_id);
    ipc_func_wrapper_->ConnectNcclDataChannel(nccl_unique_id);
    distributed_control_coordinator->Barrier();
#endif
  }

  return Status();
}

Status ModelDriver::SynchronizeEPCommunicationConfig(
    std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator) {
  KLLM_LOG_DEBUG << "Start to synchronize EP communicator config.";

  // Step 1: Get communicator config from all executors in current node
  std::vector<EPCommunicationConfig> executor_comm_configs;
  STATUS_CHECK_RETURN(ipc_func_wrapper_->GetEPCommunicationConfig(executor_comm_configs));
  KLLM_LOG_DEBUG << "Got EP communicator config from " << executor_comm_configs.size() << " executors.";

  // Step 2: Synchronize config across all nodes (master aggregates and broadcasts)
  std::vector<EPCommunicationConfig> aggregated_comm_configs;
  STATUS_CHECK_RETURN(distributed_control_coordinator->SynchronizeEPCommunicationConfig(executor_comm_configs,
                                                                                        aggregated_comm_configs));
  KLLM_LOG_DEBUG << "Synchronized EP communicator config across nodes, got " << aggregated_comm_configs.size()
                 << " entries.";

  // Step 3: Set aggregated config back to all executors in current node
  STATUS_CHECK_RETURN(ipc_func_wrapper_->SetEPCommunicationConfig(aggregated_comm_configs));
  distributed_control_coordinator->Barrier();
  KLLM_LOG_DEBUG << "EP communicator config synchronization finished.";
  return Status();
}

Status ModelDriver::InitializeResultReporter(
    std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator) {
  Status status;
  if (context_->global->IsDistributedLastWorker()) {
    KLLM_LOG_INFO << "Start to initialize result reporter channel.";
    status = ipc_func_wrapper_->ConnectResultReporterChannel();
  }

  distributed_control_coordinator->Barrier();
  KLLM_LOG_INFO << "ModelDriver::InitializeResultReporter initialize result reporter channel finished.";
  return status;
}

Status ModelDriver::CreateBlockManager(const BlockAllocatorManagerConfig& block_allocator_manager_config) {
  return ipc_func_wrapper_->CreateBlockAllocatorManager(block_allocator_manager_config);
}

Status ModelDriver::CreateExecutorRuntime(const VocabInfo& vocab_info) {
  return ipc_func_wrapper_->CreateExecutorRuntime(vocab_info);
}

void ModelDriver::SetScheduleProcessor(std::shared_ptr<ScheduleProcessorInterface> schedule_processor) {
  if (!schedule_processor) {
    KLLM_THROW("SetBatchScheduler error, nullptr found.");
  }
  schedule_processor_ = schedule_processor;
}

Status ModelDriver::Start() {
  if (!context_->global->IsDistributedWorker()) {
    request_recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&ModelDriver::HandleRequestRecv, this));
    schedule_processor_->Start();
    schedule_thread_ = std::unique_ptr<std::thread>(new std::thread(&ModelDriver::HandleSchedule, this));
  }

  if (!context_->global->IsDistributedWorker()) {
    schedule_update_thread_ = std::unique_ptr<std::thread>(new std::thread(&ModelDriver::HandleScheduleUpdate, this));
  }

  if (context_->global->IsDistributedMaster()) {
    distribted_master_thread_ =
        std::unique_ptr<std::thread>(new std::thread(&ModelDriver::HandleDistributedSamplingOutput, this));
  }

  if (context_->global->IsDistributedWorker()) {
    distribted_worker_thread_ =
        std::unique_ptr<std::thread>(new std::thread(&ModelDriver::HandleDistributedScheduleOutput, this));
  }

  schedule_output_send_thread_ =
      std::unique_ptr<std::thread>(new std::thread(&ModelDriver::HandleScheduleOutputSend, this));

  return Status();
}

Status ModelDriver::Stop() {
  terminated_.store(true);

  schedule_processor_->Stop();

  schedule_output_queue_->Stop();

  if (sampling_output_queue_) {
    sampling_output_queue_->Stop();
  }

  if (request_recv_thread_) {
    request_recv_thread_->join();
  }
  if (schedule_thread_) {
    schedule_thread_->join();
  }
  if (schedule_output_send_thread_) {
    schedule_output_send_thread_->join();
  }
  if (schedule_update_thread_) {
    schedule_update_thread_->join();
  }
  if (distribted_worker_thread_) {
    distribted_worker_thread_->join();
  }

  return Status();
}

void ModelDriver::ConvertMultimodalEmbedding(std::shared_ptr<Request>& req) {
  const std::vector<std::vector<float>>& embeddings = req->input_refit_embedding.embeddings;
  const std::vector<std::vector<uint8_t>>& embedding_bytes = req->input_refit_embedding.embedding_bytes;
  const std::vector<py::object>& embedding_pyobjects = req->input_refit_embedding.embedding_pyobjects;
  const std::vector<py::object>& additional_pyobjects = req->input_refit_embedding.additional_pyobjects;
  if (!embeddings.empty()) {
    KLLM_LOG_DEBUG << "Converting multimodal embedding data from vector<float> to Tensor";
    std::vector<Tensor>& embedding_tensors = req->input_refit_embedding.embedding_tensors;
    embedding_tensors.resize(embeddings.size());
    for (size_t i = 0; i < embeddings.size(); i++) {
      const std::vector<float>& embedding = embeddings[i];
      embedding_tensors[i] =
          Tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, std::vector<size_t>{embedding.size()});
      memcpy(embedding_tensors[i].GetPtr<void>(), embedding.data(), embedding_tensors[i].GetTotalBytes());
    }
  } else if (!embedding_bytes.empty()) {
    KLLM_LOG_DEBUG << "Converting multimodal embedding data from vector<uint8_t> to Tensor";
    DataType embedding_dtype;
    if (const std::string& dtype_str = req->input_refit_embedding.embedding_dtype; dtype_str == "fp32") {
      embedding_dtype = DataType::TYPE_FP32;
    } else if (dtype_str == "fp16") {
      embedding_dtype = DataType::TYPE_FP16;
    } else if (dtype_str == "bf16") {
      embedding_dtype = DataType::TYPE_BF16;
    } else {
      embedding_dtype = DataType::TYPE_FP32;
      KLLM_LOG_WARNING << fmt::format("Invalid input refit embedding dtype: {}, use fp32 instead.", dtype_str);
    }
    std::vector<Tensor>& embedding_tensors = req->input_refit_embedding.embedding_tensors;
    embedding_tensors.resize(embedding_bytes.size());
    for (size_t i = 0; i < embedding_bytes.size(); i++) {
      const std::vector<uint8_t>& embedding = embedding_bytes[i];
      embedding_tensors[i] = Tensor(MemoryLocation::LOCATION_HOST, embedding_dtype,
                                    std::vector<size_t>{embedding.size() / GetTypeSize(embedding_dtype)});
      memcpy(embedding_tensors[i].GetPtr<void>(), embedding.data(), embedding_tensors[i].GetTotalBytes());
    }
  } else if (!embedding_pyobjects.empty() || !additional_pyobjects.empty()) {
    KLLM_LOG_DEBUG << "Converting multimodal embedding data from py::object to Tensor";
    std::vector<Tensor>& embedding_tensors = req->input_refit_embedding.embedding_tensors;
    std::vector<Tensor>& additional_tensors = req->input_refit_embedding.additional_tensors;
    embedding_tensors.resize(embedding_pyobjects.size());
    additional_tensors.resize(additional_pyobjects.size());
    {
      py::gil_scoped_acquire acquire;
      // embedding_tensors
      for (size_t i = 0; i < embedding_pyobjects.size(); i++) {
        torch::Tensor py_tensor = THPVariable_Unpack(embedding_pyobjects[i].ptr());
        embedding_tensors[i] = Tensor(MemoryLocation::LOCATION_HOST, GetDataTypeFromTorchType(py_tensor.scalar_type()),
                                      std::vector<size_t>(py_tensor.sizes().begin(), py_tensor.sizes().end()));
        memcpy(embedding_tensors[i].GetPtr<void>(), py_tensor.data_ptr(), embedding_tensors[i].GetTotalBytes());
      }
      // additional_tensors
      for (size_t i = 0; i < additional_pyobjects.size(); i++) {
        torch::Tensor py_tensor = THPVariable_Unpack(additional_pyobjects[i].ptr());
        additional_tensors[i] = Tensor(MemoryLocation::LOCATION_HOST, GetDataTypeFromTorchType(py_tensor.scalar_type()),
                                       std::vector<size_t>(py_tensor.sizes().begin(), py_tensor.sizes().end()));
        memcpy(additional_tensors[i].GetPtr<void>(), py_tensor.data_ptr(), additional_tensors[i].GetTotalBytes());
      }
    }
  }
}

Status ModelDriver::Enqueue(std::shared_ptr<Request>& req) {
  KLLM_LOG_DEBUG << "BatchScheduler::Enqueue enqueue req id " << req->req_id;
  KLLM_LOG_DEBUG << "req id:" << req->req_id << ", input_tokens:" << Vector2Str(req->input_tokens);

  // Update `stop_token_ids` based on the config of the requested model.
  std::vector<int>& stop_token_ids = req->sampling_config.stop_token_ids;
  if (req->sampling_config.ignore_eos) {  // Ignore any end ids.
    stop_token_ids.clear();
  } else {  // Supplement the end ids in model config or generation config.
    for (int end_id : model_config_.end_ids) {
      if (std::find(stop_token_ids.begin(), stop_token_ids.end(), end_id) == stop_token_ids.end()) {
        stop_token_ids.push_back(end_id);
      }
    }
  }

  // 多模态embedding数据转换
  ConvertMultimodalEmbedding(req);

  std::vector<std::shared_ptr<InferRequest>> infer_request_group;
  for (size_t i = 0; i < req->output_group.size(); i++) {
    std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(req, i);
    infer_request_group.push_back(infer_req);
    infer_req->kv_cache_blocks.clear();
    infer_req->block_token_num = runtime_config_.attn_backend_config.block_token_num;
    infer_req->infer_stage = InferStage::kContext;
    infer_req->step = 0;
    infer_req->kv_cached_token_num = 0;
  }

  for (auto& infer_req : infer_request_group) {
    infer_req->SetReqGroup(infer_request_group);
  }

  Status enqueue_status = schedule_processor_->AddInferRequest(infer_request_group);
  if (enqueue_status.OK()) {
    KLLM_LOG_DEBUG << "batch scheduler: added req id " << req->req_id << " and "
                   << infer_request_group[0]->input_tokens.size() << " input tokens";
  } else {
    KLLM_LOG_ERROR << "batch scheduler: add req id " << req->req_id << " and "
                   << infer_request_group[0]->input_tokens.size()
                   << " input tokens failed, message: " << enqueue_status.ToString();
    if (req->sampling_config.num_beams > 1) {
      for (auto& infer_req : infer_request_group) {
        infer_req->ClearReqGroup();
      }
    }
  }

  return enqueue_status;
}

Status ModelDriver::HandleRequestRecv() {
  while (!terminated_.load()) {
    std::pair<Status, std::shared_ptr<Request>> req_pair;
    request_queue_.Read(&req_pair);
    if (terminated_) {
      KLLM_LOG_INFO << "ModelDriver is terminated, HandleRequestRecv stopped.";
      break;
    }

    Status status = req_pair.first;
    if (status.GetCode() == RET_REQUEST_TERMINATED) {
      KLLM_LOG_INFO << "InferenceEngine is terminated, HandleRequestRecv stopped.";
      break;
    }

    std::shared_ptr<Request> req = req_pair.second;
    if (req) {
      Enqueue(req);
    }
  }
  return Status();
}

void ModelDriver::SetInferRequestTransferMode(std::shared_ptr<ScheduleOutput> schedule_output) {
  static std::unordered_set<int64_t> req_ids_cache;

  // clear local cache, transfer in full mode.
  for (const auto& v : schedule_output->recompute_req_ids) {
    for (const int64_t& req_id : v) {
      req_ids_cache.erase(req_id);
      KLLM_LOG_DEBUG << "Erase req " << req_id << " from local cache(recomputed).";
    }
  }

  for (const auto& v : schedule_output->finish_req_ids) {
    for (const int64_t& req_id : v) {
      req_ids_cache.erase(req_id);
      KLLM_LOG_DEBUG << "Erase req " << req_id << " from local cache(finished).";
    }
  }

  for (auto& req : schedule_output->running_reqs) {
    if (req_ids_cache.count(req->req_id) == 0) {
      req_ids_cache.insert(req->req_id);
      KLLM_LOG_DEBUG << "Cache req " << req->req_id << " to local cache.";
      req->patch_transfer = false;
    } else {
      KLLM_LOG_DEBUG << "req " << req->req_id << " already in local cache.";
      req->patch_transfer = true;
    }
  }
}

Status ModelDriver::HandleSchedule() {
  while (!terminated_.load()) {
    // The scheduler should be blocked if no requests.
    std::shared_ptr<ScheduleResult> schedule_result = schedule_processor_->GetNextScheduleResult();
    std::shared_ptr<ScheduleOutput> schedule_output = schedule_result->schedule_output;
    if (!schedule_output) {
      KLLM_LOG_INFO << "HandleSchedule stopped.";
      break;
    }

    SetInferRequestTransferMode(schedule_output);

    // Prepare structured generator configs for Executor-side generation
    if (!schedule_output->running_reqs.empty()) {
      for (const auto& req : schedule_output->running_reqs) {
        KLLM_LOG_DEBUG << "req_id " << req->req_id
                       << ": HasConstraint=" << req->structured_generator_config.HasConstraint()
                       << ", constraint_type=" << static_cast<int>(req->structured_generator_config.constraint_type)
                       << ", constraint_spec=" << req->structured_generator_config.constraint_spec
                       << ", output_tokens_size=" << req->output_tokens.size();

        if (req->structured_generator_config.HasConstraint()) {
          // Create generator only once: when no tokens have been generated yet
          // output_tokens initially contains input_tokens, so we check if they're equal
          bool need_create = (req->output_tokens.size() == req->input_tokens.size());
          if (need_create) {
            // Set need_recreate_generator flag on the request itself
            // This will be serialized and sent to Executor
            req->need_recreate_generator = true;

            // Also add to ScheduleOutput level for backward compatibility
            schedule_output->structured_generator_configs[req->req_id] = req->structured_generator_config;
            schedule_output->need_create_generator_req_ids.insert(req->req_id);
            KLLM_LOG_INFO << "Marked req_id " << req->req_id << " for generator creation";
          } else {
            req->need_recreate_generator = false;
            KLLM_LOG_DEBUG << "req_id " << req->req_id << " already has output tokens, skipping generator creation";
          }
        }
      }
    }

    {
      std::lock_guard<std::mutex> guard(schedule_result_map_mutex_);
      schedule_result_map_[schedule_output->schedule_id] = schedule_result;
    }
    schedule_output_queue_->Put(schedule_output);
    if (context_->global->IsDistributedMaster()) {
      schedule_output_pool_->PutSendScheduleOutput(schedule_output.get());
    }
  }
  return Status();
}

Status ModelDriver::HandleScheduleOutputSend() {
  while (!terminated_.load()) {
    std::shared_ptr<ScheduleOutput> schedule_output = schedule_output_queue_->Get();
    if (!schedule_output) {
      KLLM_LOG_INFO << "HandleScheduleOutputSend stopped.";
      break;
    }

    ipc_func_wrapper_->DispatchScheduleOutput(schedule_output);
  }
  return Status();
}

void ModelDriver::UpdateSamplingOutput(std::shared_ptr<SamplingOutput> sampling_output) {
  std::lock_guard<std::mutex> guard(schedule_result_map_mutex_);
  size_t schedule_id = sampling_output->schedule_id;
  auto it = schedule_result_map_.find(schedule_id);
  if (it == schedule_result_map_.end()) {
    KLLM_LOG_FATAL << "UpdateSamplingOutput error, schedule_id " << schedule_id << " not found.";
    return;
  }

  std::shared_ptr<ScheduleResult> schedule_result = it->second;

  // update response of request target
  for (auto& result_it : sampling_output->target_responses) {
    auto req_it = schedule_result->running_reqs.find(result_it.first);
    if (req_it != schedule_result->running_reqs.end()) {
      auto req = req_it->second;
      req->response = std::move(result_it.second);
    }
  }
  sampling_output->target_responses.clear();

  // update logprobs of request target
  for (const auto& result_it : sampling_output->target_logprobs) {
    auto req_it = schedule_result->running_reqs.find(result_it.first);
    if (req_it != schedule_result->running_reqs.end()) {
      auto req = req_it->second;
      req->logprobs.insert(req->logprobs.end(), result_it.second.begin(), result_it.second.end());
    }
  }
  sampling_output->target_logprobs.clear();

  // 收集需要通过transfer_engine发送token的请求
  std::vector<std::tuple<std::string, int, std::vector<int>>> reqs_transfer_tokens;

  // update schedule state of every dp group.
  for (auto& result_it : sampling_output->sampling_result) {
    auto req_it = schedule_result->running_reqs.find(result_it.first);
    if (req_it != schedule_result->running_reqs.end()) {
      GenerationResult gen_result;
      gen_result.DeserializeFromVector(result_it.second);
      auto req = req_it->second;
      req->generated_tokens = gen_result.generated_tokens;
      req->accepted_tokens = gen_result.accepted_tokens;
      req->draft_tokens.mtp = gen_result.draft_tokens;

      // 如果有transfer_engine，准备发送token数据
      if (transfer_engine_ && transfer_engine_->GetGroupRole() == GroupRole::PREFILL) {
        std::vector<int> draft_tokens = req->draft_tokens.GetDraftTokens();
        // {generated_tokens.size(), generated_tokens, draft_tokens.size(), draft_tokens}
        std::vector<int> send_tokens(MAX_TRANSFER_TOKENS, -1);
        send_tokens[0] = req->generated_tokens.size();
        std::copy(req->generated_tokens.begin(), req->generated_tokens.end(), send_tokens.begin() + 1);
        send_tokens[req->generated_tokens.size() + 1] = draft_tokens.size();
        if (draft_tokens.size() > 0) {
          std::copy(draft_tokens.begin(), draft_tokens.end(), send_tokens.begin() + 2 + req->generated_tokens.size());
        }
        KLLM_LOG_DEBUG << "UpdateSamplingOutput: prepare to send tokens for req " << req->req_id
                       << ", send tokens: " << Vector2Str(send_tokens);
        reqs_transfer_tokens.emplace_back(req->kv_comm_group_key, req->kv_comm_request_id, send_tokens);
      }
    }
  }

  // 使用transfer_engine发送所有收集到的token
  if (transfer_engine_ && !reqs_transfer_tokens.empty()) {
    transfer_engine_->Send(reqs_transfer_tokens);
    KLLM_LOG_DEBUG << "UpdateSamplingOutput: sent tokens for " << reqs_transfer_tokens.size() << " requests";
  }

  schedule_processor_->UpdateWithGenerationResult(schedule_result->generation_output_group);
  schedule_result_map_.erase(it);
}

Status ModelDriver::HandleScheduleUpdate() {
  // set attributes of recv function.
  ipc_func_wrapper_->SetSamplingOutputRecvQueue(sampling_output_queue_);

  if (sampling_output_queue_ == nullptr) {
    throw std::runtime_error("HandleScheduleUpdate error, sampling_output_queue is nullptr.");
  }

  while (!terminated_.load()) {
    std::shared_ptr<SamplingOutput> sampling_output = sampling_output_queue_->Get();
    if (!sampling_output) {
      KLLM_LOG_INFO << "HandleScheduleUpdate stopped.";
      break;
    }

    if (!context_->global->IsDistributedWorker()) {
      UpdateSamplingOutput(sampling_output);
    }
  }
  return Status();
}

Status ModelDriver::HandleDistributedScheduleOutput() {
  if (!context_->global->IsDistributedWorker()) {
    return Status(RET_INVALID_ARGUMENT, "HandleDistributedScheduleOutput only for distributed worker.");
  }

  while (!terminated_.load()) {
    Packet* serialized_packet = schedule_output_pool_->GetRecvSerializedPacket();
    if (!serialized_packet) {
      KLLM_LOG_INFO << "HandleDistributedScheduleOutput stopped.";
      break;
    }

    ipc_func_wrapper_->DispatchScheduleOutput(serialized_packet);

    // DO NOT forget this.
    free(serialized_packet);
  }

  return Status();
}

Status ModelDriver::HandleDistributedSamplingOutput() {
  if (!context_->global->IsDistributedMaster()) {
    return Status(RET_INVALID_ARGUMENT, "HandleDistributedSamplingOutput only for distributed master.");
  }

  while (!terminated_.load()) {
    SamplingOutput* sampling_output = sampling_output_pool_->GetRecvSamplingOutput();
    if (!sampling_output) {
      KLLM_LOG_INFO << "HandleDistributedSamplingOutput stopped.";
      break;
    }

    auto nop_deleter = [](SamplingOutput* p) {};
    std::shared_ptr<SamplingOutput> shared_sampling_output(sampling_output, nop_deleter);
    sampling_output_queue_->Put(shared_sampling_output);
  }

  return Status();
}

}  // namespace ksana_llm
