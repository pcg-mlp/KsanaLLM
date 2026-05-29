/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/model_runner.h"

#include <unistd.h>

#include <algorithm>

#include "executor/executor_block_manager.h"
#include "executor/global_cache_put_worker.h"
#include "executor/layer_progress_tracker.h"
#include "profiler/profile_event.h"
#include "runtime/structured_generation/structured_generator_interface.h"
#include "utils/waiter.h"

namespace ksana_llm {

ModelRunner::ModelRunner(std::shared_ptr<Context> context) : mtp_step_num_(context->global->GetMtpStepNum()) {
  context_ = context;

  schedule_output_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<ScheduleOutput>>>();
  model_input_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<NewModelInput>>>();
  model_output_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<NewModelOutput>>>();
}

void ModelRunner::SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper) {
  ipc_func_wrapper_ = ipc_func_wrapper;
}

void ModelRunner::SetExecutorRuntime(std::shared_ptr<ExecutorRuntime> executor_runtime) {
  executor_runtime_ = executor_runtime;
}

void ModelRunner::SetModelInstance(std::shared_ptr<ModelInstance> model_instance) {
  model_instance_ = model_instance;
  executor_runtime_->SetModelInstance(model_instance_);
}

void ModelRunner::SetModelWeight(std::shared_ptr<WeightInstanceInterface> model_weight) {
  model_weight_ = model_weight;
  executor_runtime_->SetModelWeight(model_weight_);
}

void ModelRunner::SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager) {
  block_allocator_manager_ = block_allocator_manager;
  executor_runtime_->SetBlockAllocatorManager(block_allocator_manager_);
}

void ModelRunner::SetSwaBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager) {
  executor_runtime_->SetSwaBlockAllocatorManager(swa_block_allocator_manager);
}

void ModelRunner::SetExecutorBlockManager(std::shared_ptr<ExecutorBlockManager> executor_block_manager) {
  executor_block_manager_ = executor_block_manager;
}

void ModelRunner::SetHiddenUnitBufferPool(std::shared_ptr<HiddenUnitBufferPool> hidden_unit_buffer_pool) {
  hidden_unit_buffer_pool_ = hidden_unit_buffer_pool;
  executor_runtime_->SetHiddenUnitBufferPool(hidden_unit_buffer_pool);
}

void ModelRunner::SetSamplingOutputPool(std::shared_ptr<SamplingOutputPool> sampling_output_pool) {
  sampling_output_pool_ = sampling_output_pool;
}

void ModelRunner::SetDraftGeneratorController(std::shared_ptr<DraftGeneratorController> draft_generator_controller) {
  executor_runtime_->SetDraftGeneratorController(draft_generator_controller);
}

void ModelRunner::SetGlobalCachePutWorker(std::shared_ptr<GlobalCachePutWorker> global_cache_put_worker) {
  global_cache_put_worker_ = global_cache_put_worker;
}

Status ModelRunner::Start() {
  preprocess_thread_ = std::thread(&ModelRunner::HandlePreprocess, this);
  execute_thread_ = std::thread(&ModelRunner::HandleExecute, this);

  if (context_->global->IsSamplingNode()) {
    postprocess_thread_ = std::thread(&ModelRunner::HandlePostprocess, this);
  }

  return Status();
}

Status ModelRunner::Stop() {
  terminated_.store(true, std::memory_order_relaxed);

  schedule_output_queue_->Stop();
  model_input_queue_->Stop();
  model_output_queue_->Stop();

  if (preprocess_thread_.joinable()) {
    preprocess_thread_.join();
  }
  if (execute_thread_.joinable()) {
    execute_thread_.join();
  }
  if (postprocess_thread_.joinable()) {
    postprocess_thread_.join();
  }

  return Status();
}

void ModelRunner::GetHiddenShape(const std::shared_ptr<ScheduleOutput>& schedule_output, std::vector<size_t>& shape) {
  std::vector<size_t> input_ids_cpu;
  for (const auto& req : schedule_output->executor_running_reqs) {
    const size_t skip_token_num = std::max(req->kv_cached_token_num, static_cast<size_t>(req->prefix_cache_len));
    input_ids_cpu.insert(input_ids_cpu.end(), req->forwarding_tokens.begin() + skip_token_num,
                         req->forwarding_tokens.end());
  }
  shape = {input_ids_cpu.size(),
           model_instance_->GetModelConfig().hidden_units * model_instance_->GetModelConfig().mhc_config.hc_mult};
}

DataType ModelRunner::GetHiddenDataType() { return model_instance_->GetModelConfig().weight_data_type; }

Status ModelRunner::HandlePreprocess() {
  SetDevice(context_->device->GetDeviceId());

  ipc_func_wrapper_->SetScheduleOutputRecvQueue(schedule_output_queue_);

  while (!terminated_.load(std::memory_order_relaxed)) {
    std::shared_ptr<ScheduleOutput> schedule_output;
    {
      PROFILE_EVENT_SCOPE(preprocess_queue_get, "Exec.HandlePreprocess.QueueGet");
      schedule_output = schedule_output_queue_->Get();
    }
    if (!schedule_output) {
      KLLM_LOG_INFO << "HandlePreprocess stopped.";
      break;
    }

    PROFILE_EVENT_SCOPE(preprocess_iter, "Exec.HandlePreprocess.iter");

    if (executor_block_manager_ != nullptr) {
      PROFILE_EVENT_SCOPE(preprocess_swap, "Exec.HandlePreprocess.Swap");
      Status status = executor_block_manager_->SwapoutRequestMemoryBlockAsync(schedule_output->swapout_req_block_ids);
      if (!status.OK()) {
        KLLM_LOG_ERROR << "HandlePreprocess swapout request failed: " << status.ToString()
                       << ", schedule_id=" << schedule_output->schedule_id;
        _exit(EXIT_FAILURE);
      }
      executor_block_manager_->WaitSwapoutRequestMemoryBlock(schedule_output->merged_swapout_req_ids);
      status = executor_block_manager_->SwapinRequestMemoryBlockAsync(schedule_output->swapin_req_block_ids);
      if (!status.OK()) {
        KLLM_LOG_ERROR << "HandlePreprocess swapin request failed: " << status.ToString()
                       << ", schedule_id=" << schedule_output->schedule_id;
        _exit(EXIT_FAILURE);
      }
      executor_block_manager_->WaitSwapinRequestMemoryBlock(schedule_output->merged_swapin_req_ids);
    }

    {
      PROFILE_EVENT_SCOPE(preprocess_update_cached, "Exec.HandlePreprocess.UpdateCachedRequests");
      executor_runtime_->UpdateCachedRequests(schedule_output);
    }
    {
      PROFILE_EVENT_SCOPE(preprocess_rebuild, "Exec.HandlePreprocess.RebuildScheduleOutput");
      executor_runtime_->RebuildScheduleOutput(schedule_output);
    }

    auto generation_controller = executor_runtime_->GetGenerationController();
    if (generation_controller) {
      PROFILE_EVENT_SCOPE(preprocess_generator, "Exec.HandlePreprocess.RestoreGenerators");
      if (!schedule_output->need_create_generator_req_ids.empty()) {
        generation_controller->CreateGenerators(schedule_output);
      }
      generation_controller->RestoreGenerators(schedule_output);
    }

    if (!context_->global->IsStandalone()) {
      PROFILE_EVENT_SCOPE(preprocess_hidden_meta, "Exec.HandlePreprocess.HiddenMeta");
      std::vector<size_t> shape;
      GetHiddenShape(schedule_output, shape);
      DataType data_type = GetHiddenDataType();
      if (context_->global->IsDistributedWorker()) {
        hidden_unit_buffer_pool_->AddHiddenUnitRecvMeta(shape, data_type);
      }

      if (!context_->global->IsDistributedLastWorker()) {
        hidden_unit_buffer_pool_->AddHiddenUnitSendMeta(shape, data_type);
      }
    }

    std::shared_ptr<NewModelInput> model_input = std::make_shared<NewModelInput>();
    {
      PROFILE_EVENT_SCOPE(preprocess_build_model_input, "Exec.HandlePreprocess.BuildModelInput");
      executor_runtime_->BuildModelInput(schedule_output, model_input, RunMode::kMain);
    }

    model_input_queue_->Put(model_input);
  }
  return Status();
}

Status ModelRunner::HandleExecute() {
  SetDevice(context_->device->GetDeviceId());

  while (!terminated_.load(std::memory_order_relaxed)) {
    std::shared_ptr<NewModelInput> model_input;
    {
      PROFILE_EVENT_SCOPE(execute_input_get, "Exec.HandleExecute.InputQueueGet");
      model_input = model_input_queue_->Get();
    }
    if (!model_input) {
      KLLM_LOG_INFO << "HandleExecute stopped.";
      break;
    }

    PROFILE_EVENT_SCOPE(execute_iter, "Exec.HandleExecute.iter");
    std::shared_ptr<NewModelOutput> model_output = std::make_shared<NewModelOutput>();

    HiddenUnitDeviceBuffer* hidden_unit = nullptr;
    {
      PROFILE_EVENT_SCOPE(execute_get_buffer, "Exec.HandleExecute.GetHiddenBuffer");
      hidden_unit = hidden_unit_buffer_pool_->GetBuffer();
    }
    GetHiddenShape(model_input->schedule_output, hidden_unit->tensor.shape);
    hidden_unit->tensor.dtype = GetHiddenDataType();

    {
      PROFILE_EVENT_SCOPE(execute_forward_launch, "Exec.HandleExecute.ForwardLaunch");
      executor_runtime_->Forward(model_input, hidden_unit, model_output);
    }

    if (global_cache_put_worker_) {
      global_cache_put_worker_->SubmitBatchPut(model_input->schedule_output->executor_running_reqs);
    }

    if (context_->global->IsSamplingNode()) {
      model_output->hidden_unit = hidden_unit;
      model_output_queue_->Put(model_output);
    } else {
      {
        PROFILE_EVENT_SCOPE(execute_stream_sync, "Exec.HandleExecute.NonSamplingStreamSync");
        StreamSynchronize(context_->device->GetPrimaryComputeStream());
      }
      hidden_unit_buffer_pool_->FreeBuffer(hidden_unit);
    }
  }
  return Status();
}

// TODO(robertyuan): Change serializing struct from SamplingOutput to GenerationResult.
void SerializeGenerationResultToSamplingResult(const std::unordered_map<int64_t, GenerationResult>& generation_results,
                                               std::shared_ptr<SamplingOutput> sampling_output) {
  for (auto& [req_id, result] : generation_results) {
    result.SerializeToVector(sampling_output->sampling_result[req_id]);
  }
}

Status ModelRunner::HandlePostprocess() {
  SetDevice(context_->device->GetDeviceId());
  while (!terminated_.load(std::memory_order_relaxed)) {
    std::shared_ptr<NewModelOutput> model_output;
    {
      PROFILE_EVENT_SCOPE(postprocess_output_get, "Exec.HandlePostprocess.OutputQueueGet");
      model_output = model_output_queue_->Get();
    }
    if (!model_output) {
      KLLM_LOG_INFO << "HandlePostprocess stopped.";
      break;
    }

    PROFILE_EVENT_SCOPE(postprocess_iter, "Exec.HandlePostprocess.iter");
    std::shared_ptr<SamplingOutput> sampling_output = std::make_shared<SamplingOutput>();
    {
      PROFILE_EVENT_SCOPE(postprocess_sampling, "Exec.HandlePostprocess.Sampling");
      executor_runtime_->Sampling(model_output, sampling_output);
    }

    auto generation_controller = executor_runtime_->GetGenerationController();
    {
      PROFILE_EVENT_SCOPE(postprocess_update_gen_state, "Exec.HandlePostprocess.UpdateGenerationState");
      generation_controller->UpdateGenerationState(model_output->schedule_output->executor_running_reqs);
    }

    std::unordered_map<int64_t, GenerationResult> generation_results;
    std::vector<int64_t> finished_req_ids;
    for (auto& req : model_output->schedule_output->executor_running_reqs) {
      auto& result = generation_results[req->req_id];
      result.generated_tokens = req->generated_tokens;
      result.accepted_tokens = req->accepted_tokens;
      sampling_output->target_logprobs[req->req_id] = req->logprobs;
      if (req->structured_generator && req->structured_generator->IsTerminated()) {
        finished_req_ids.push_back(req->req_id);
      }
    }

    if (!finished_req_ids.empty() && generation_controller) {
      for (int64_t req_id : finished_req_ids) {
        generation_controller->RemoveGenerator(req_id, model_output->schedule_output->executor_running_reqs);
        KLLM_LOG_DEBUG << "Cleaned up generator for finished req_id: " << req_id;
      }
    }

    HiddenUnitDeviceBuffer* const hidden_unit = model_output->hidden_unit;

    {
      PROFILE_EVENT_SCOPE(postprocess_gen_draft, "Exec.HandlePostprocess.GenerateDraft");
      for (auto& req : model_output->schedule_output->executor_running_reqs) {
        generation_controller->GenerateDraft(req);
      }
    }

    if (context_->global->IsSamplingNode() && mtp_step_num_ > 0) {
      PROFILE_EVENT_SCOPE(postprocess_mtp_forward, "Exec.HandlePostprocess.MtpForward");
      GetHiddenShape(model_output->schedule_output, hidden_unit->tensor.shape);
      hidden_unit->tensor.dtype = GetHiddenDataType();
      executor_runtime_->MtpForward(mtp_step_num_, model_output->schedule_output->executor_running_reqs, hidden_unit,
                                    model_output->schedule_output);
    }
    {
      PROFILE_EVENT_SCOPE(postprocess_free_buffer, "Exec.HandlePostprocess.FreeHiddenBuffer");
      hidden_unit_buffer_pool_->FreeBuffer(hidden_unit);
    }
    Singleton<LayerProgressTracker>::GetInstance()->ResetState();

    for (auto& req : model_output->schedule_output->executor_running_reqs) {
      generation_results[req->req_id].draft_tokens = req->draft_tokens;
    }

    if (global_cache_put_worker_) {
      KLLM_LOG_DEBUG << "ModelRunner::HandlePostprocess: calling SubmitFirstTokensPut, req_count="
                     << model_output->schedule_output->executor_running_reqs.size();
      global_cache_put_worker_->SubmitFirstTokensPut(model_output->schedule_output->executor_running_reqs);
    }

    {
      PROFILE_EVENT_SCOPE(postprocess_serialize_gen, "Exec.HandlePostprocess.SerializeGenerationResult");
      SerializeGenerationResultToSamplingResult(generation_results, sampling_output);
    }

    if (context_->device->GetDeviceId() == 0) {
      PROFILE_EVENT_SCOPE(postprocess_report, "Exec.HandlePostprocess.Report");
      if (context_->global->IsStandalone()) {
        ipc_func_wrapper_->ReportSamplingOutput(sampling_output);
      } else if (context_->global->IsDistributedResultReporter()) {
        sampling_output_pool_->PutSendSamplingOutput(sampling_output.get());
      }
    }
  }
  return Status();
}

}  // namespace ksana_llm
