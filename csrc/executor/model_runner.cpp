/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/model_runner.h"

#include <algorithm>

#include "fmt/format.h"

#include "device/device_utils.h"
#include "executor/global_cache_put_worker.h"
#include "executor/layer_progress_tracker.h"
#include "executor/model_sampler.h"
#include "profiler/profile_event.h"
#include "runtime/fast_path_controller.h"
#include "runtime/structured_generation/structured_generator_interface.h"
#include "samplers/sampler.h"
#include "utils/singleton.h"
#include "utils/waiter.h"

namespace ksana_llm {

ModelRunner::ModelRunner(std::shared_ptr<Context> context) : mtp_step_num_(context->global->GetMtpStepNum()) {
  context_ = context;

  schedule_output_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<ScheduleOutput>>>();
  model_input_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<NewModelInput>>>();
  model_output_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<NewModelOutput>>>();
  sampling_output_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<SamplingOutput>>>();
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
    if (FastPathController::GetInstance().IsEnabledAtStartup()) {
      // build-ahead: Forward/Sampling/WriteRing 在 Execute 线程串行, IPC 上报走独立线程.
      result_report_thread_ = std::thread(&ModelRunner::HandleResultReportProcess, this);
    } else {
      // 慢路径: Preprocess / Execute / Postprocess 三线程（build-ahead 关闭时的默认模型）。
      postprocess_thread_ = std::thread(&ModelRunner::HandlePostprocess, this);
    }
  }

  return Status();
}

Status ModelRunner::Stop() {
  terminated_.store(true, std::memory_order_relaxed);

  schedule_output_queue_->Stop();
  model_input_queue_->Stop();
  model_output_queue_->Stop();
  sampling_output_queue_->Stop();

  // Wake the execute thread if it is currently parked inside ExecutorRuntime::Forward()'s wait for
  // the previous MtpForward Notify(); without this the join below could hang on a Notify that
  // HandlePostprocess will never deliver after the queues have been stopped.
  if (executor_runtime_) {
    executor_runtime_->Shutdown();
  }

  if (preprocess_thread_.joinable()) {
    preprocess_thread_.join();
  }
  if (execute_thread_.joinable()) {
    execute_thread_.join();
  }
  if (postprocess_thread_.joinable()) {
    postprocess_thread_.join();
  }
  if (result_report_thread_.joinable()) {
    result_report_thread_.join();
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

    // ModelInput / ring 双 slot 乒乓翻转, 使 Preprocess 与 Execute 可重叠且互不覆写.
    // 慢路径恒用 slot 0。
    if (FastPathController::GetInstance().IsEnabledAtStartup()) {
      cur_ring_slot_ ^= 1u;
      model_input->slot_index = cur_ring_slot_;
    } else {
      model_input->slot_index = 0;
    }

    // 双 H2D stream 开启时, 在 Preprocess 线程完成 ParseFromRequests, 与上一步 Forward 重叠.
    model_input->gpu_input_ready = false;
    if (context_->device->IsDualH2DStreamsEnabled()) {
      Status prep_status = executor_runtime_->PrepareModelInputGpu(model_input);
      if (!prep_status.OK()) {
        KLLM_LOG_ERROR << "PrepareModelInputGpu failed: " << prep_status.ToString()
                       << " schedule_id=" << schedule_output->schedule_id;
        return prep_status;
      }
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

    // 挂上一步 ring slot 指针, CommonModel::Forward 内 ReplaceLastTokenKernel 按 req_id 回填 placeholder.
    // kernel 内自检 input_ids[i]==-1 且 ring count>0, 无占位/首步时自动 no-op.
    if (FastPathController::GetInstance().IsEnabledAtStartup()) {
      const size_t prev_slot = model_input->slot_index ^ 1u;
      auto model_sampler = executor_runtime_->GetModelSampler();
      KLLM_CHECK_WITH_INFO(model_sampler != nullptr,
                           "build_ahead_enabled but ModelSampler is null in HandleExecute (config inconsistency).");
      auto sampler = model_sampler->GetSampler();
      KLLM_CHECK_WITH_INFO(sampler != nullptr,
                           "build_ahead_enabled but Sampler is null in HandleExecute (config inconsistency).");
      int64_t* prev_ring = sampler->GetRingSlot(prev_slot);
      KLLM_CHECK_WITH_INFO(
          prev_ring != nullptr,
          fmt::format("build_ahead_enabled but Sampler ring slot {} not allocated (config inconsistency).", prev_slot));
      model_input->prev_ring_dev = prev_ring;
    }

    std::shared_ptr<NewModelOutput> model_output = std::make_shared<NewModelOutput>();

    HiddenUnitDeviceBuffer* hidden_unit = nullptr;
    {
      PROFILE_EVENT_SCOPE(execute_get_buffer, "Exec.HandleExecute.GetHiddenBuffer");
      hidden_unit = hidden_unit_buffer_pool_->GetBuffer();
    }
    GetHiddenShape(model_input->schedule_output, hidden_unit->tensor.shape);
    hidden_unit->tensor.dtype = GetHiddenDataType();

    {
      // Compressor slot eviction is intentionally done BEFORE Forward (i.e. before any Prepare()
      // call inside the model) so that when a single schedule_output carries BOTH new running_reqs
      // AND accumulated finish_req_ids / recompute_req_ids from prior steps, the freed slots are
      // available for the upcoming Prepare() to consume. Doing this in the reverse order can
      // exhaust the per-rank CompressorMeta slot pool whenever the master coalesces "finish a
      // batch + start a new batch" into a single dispatch (the dominant failure mode after mass
      // client aborts: running_reqs goes empty for one or more schedule rounds, finish_req_ids
      // stays buffered in batch_state, and the next non-empty schedule round dispatches them
      // together).
      // Applied uniformly on sampling and non-sampling ranks; MTP on the sampling rank operates
      // on executor_running_reqs of the same schedule_output, which never overlap with
      // finish_req_ids of that schedule_output (a req cannot be simultaneously "forwarded this
      // step" and "already finished in a prior step"), so MTP does not need a separate evict point.
      PROFILE_EVENT_SCOPE(execute_evict_compressor, "Exec.HandleExecute.EvictCompressorSlots");
      executor_runtime_->EvictCompletedCompressorSlots(model_input->schedule_output);
    }

    {
      PROFILE_EVENT_SCOPE(execute_forward_launch, "Exec.HandleExecute.ForwardLaunch");
      executor_runtime_->Forward(model_input, hidden_unit, model_output);
    }

    if (global_cache_put_worker_) {
      global_cache_put_worker_->SubmitBatchPut(model_input->schedule_output->executor_running_reqs);
    }

    if (context_->global->IsSamplingNode()) {
      model_output->hidden_unit = hidden_unit;
      if (FastPathController::GetInstance().IsEnabledAtStartup()) {
        // fast path: Forward 与 Sampling/WriteRing 同线程串行, 结果上报走独立线程.
        HandlePostprocessInline(model_output);
      } else {
        // slow path: 三线程模型, Sampling 在 HandlePostprocess.
        model_output_queue_->Put(model_output);
      }
    } else {
      {
        PROFILE_EVENT_SCOPE(execute_stream_sync, "Exec.HandleExecute.NonSamplingStreamSync");
        StreamSynchronize(context_->device->GetPrimaryComputeStream());
      }
      hidden_unit_buffer_pool_->FreeBuffer(hidden_unit);
    }

    // TP: 采样卡 rank 0 WriteRing 后，所有 rank（含非采样 PP stage 上的 TP worker）必须参与 collective。
    if (FastPathController::GetInstance().IsEnabledAtStartup()) {
      BroadcastBuildAheadRingSlot(model_output->slot_index);
    }
  }
  return Status();
}

void ModelRunner::BroadcastBuildAheadRingSlot(size_t slot) {
  if (context_->global->GetTensorParallelSize() <= 1) {
    return;
  }
  auto model_sampler = executor_runtime_->GetModelSampler();
  KLLM_CHECK_WITH_INFO(model_sampler != nullptr,
                       "build_ahead_enabled but ModelSampler is null in BroadcastBuildAheadRingSlot.");
  auto sampler = model_sampler->GetSampler();
  KLLM_CHECK_WITH_INFO(sampler != nullptr, "build_ahead_enabled but Sampler is null in BroadcastBuildAheadRingSlot.");
  sampler->BroadcastRingSlot(slot, context_->device->GetPrimaryComputeStream());
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
    // build-ahead：Execute 线程经 HandlePostprocessInline 注入本步输出，不经 model_output_queue_。
    // 慢路径：独立 postprocess 线程消费 model_output_queue_，postprocess_one_shot_ 恒为 false。
    bool one_shot_iteration = false;
    if (postprocess_one_shot_) {
      model_output = std::move(postprocess_one_shot_output_);
      postprocess_one_shot_ = false;
      one_shot_iteration = true;
    } else {
      {
        PROFILE_EVENT_SCOPE(postprocess_output_get, "Exec.HandlePostprocess.OutputQueueGet");
        model_output = model_output_queue_->Get();
      }
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
    // build-ahead 专有：Sampling 后写 ring slot；慢路径 postprocess_enqueue_report_=false 时函数内直接返回。
    WriteBuildAheadRingAfterSampling(model_output);

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
        // build-ahead：Execute 与下一步 Forward 重叠，需把 draft 计入 sampling_token_num 供调度/workload 估算。
        // 慢路径：Sampling 在独立线程，req 状态在入队前已定型，此处无需再改 sampling_token_num。
        if (postprocess_enqueue_report_) {
          size_t draft_token_num = req->draft_tokens.size();
          req->sampling_token_num = kStepGenerateTokenNum + draft_token_num;
        }
      }
    }

    if (context_->global->IsSamplingNode() && mtp_step_num_ > 0) {
      PROFILE_EVENT_SCOPE(postprocess_mtp_forward, "Exec.HandlePostprocess.MtpForward");
      GetHiddenShape(model_output->schedule_output, hidden_unit->tensor.shape);
      hidden_unit->tensor.dtype = GetHiddenDataType();
      executor_runtime_->MtpForward(mtp_step_num_, model_output->schedule_output->executor_running_reqs, hidden_unit,
                                    model_output->schedule_output);
    }

    // Compressor slot eviction has been moved to the start of HandleExecute (before the main
    // Forward), uniformly on both sampling and non-sampling ranks. See the comment there for the
    // rationale: doing evict before Prepare lets a single schedule_output carry finish_req_ids and
    // new running_reqs together without exhausting the slot pool. MTP on this rank only Prepare()s
    // for executor_running_reqs of the same schedule_output, which never overlap with that SO's
    // finish_req_ids, so MTP does not require a dedicated evict point here.

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

    // 结果出口：fast 入队给 HandleResultReportProcess，避免 Execute 线程阻塞 IPC；
    // slow 在 postprocess 线程、device 0 上直接上报（三线程模型下无独立 report 线程）。
    if (postprocess_enqueue_report_) {
      sampling_output_queue_->Put(sampling_output);
      postprocess_enqueue_report_ = false;
    } else if (context_->device->GetDeviceId() == 0) {
      PROFILE_EVENT_SCOPE(postprocess_report, "Exec.HandlePostprocess.Report");
      if (context_->global->IsStandalone()) {
        ipc_func_wrapper_->ReportSamplingOutput(sampling_output);
      } else if (context_->global->IsDistributedResultReporter()) {
        sampling_output_pool_->PutSendSamplingOutput(sampling_output.get());
      }
    }

    // build-ahead：Execute 线程单次调用，处理完即返回；不进入下一轮 while（慢路径无此分支）。
    if (one_shot_iteration) {
      break;
    }
  }
  return Status();
}

void ModelRunner::WriteBuildAheadRingAfterSampling(const std::shared_ptr<NewModelOutput>& model_output) {
  if (!postprocess_enqueue_report_) {
    return;
  }
  // 将本步采样结果写入 ring slot: 布局 [count, (req_id, token) * B_cur],
  // 供下一步 ReplaceLastTokenKernel 按 req_id 回填 placeholder.
  // 与 Sampling 同处 PrimaryComputeStream, FIFO 保证先采样后写 ring; B_cur=0 时 kernel 内 no-op.
  if (FastPathController::GetInstance().IsEnabledAtStartup()) {
    KLLM_CHECK_WITH_INFO(model_output->cur_pairs_dev != nullptr,
                         "build-ahead enabled but model_output->cur_pairs_dev is null in HandlePostprocess.");
    auto model_sampler = executor_runtime_->GetModelSampler();
    KLLM_CHECK_WITH_INFO(model_sampler != nullptr,
                         "build_ahead_enabled but ModelSampler is null in HandlePostprocess.");
    auto sampler = model_sampler->GetSampler();
    KLLM_CHECK_WITH_INFO(sampler != nullptr, "build_ahead_enabled but Sampler is null in HandlePostprocess.");
    // TP 下仅 rank 0 持有有效 sampled_tokens；非 0 rank 不写 garbage，靠 BroadcastRingSlot 同步。
    if (context_->device->GetDeviceId() == 0) {
      sampler->WriteRing(model_output->slot_index, model_output->cur_pairs_dev, model_output->B_cur,
                         context_->device->GetPrimaryComputeStream());
    }
  }
}

Status ModelRunner::HandlePostprocessInline(std::shared_ptr<NewModelOutput> model_output) {
  // build-ahead：在 Execute 线程同步跑一轮 HandlePostprocess，与 Forward/WriteRing 同线程串行。
  // 与慢路径 postprocess 线程互斥（Start 时二选一），共享同一套 postprocess 逻辑，无需加锁。
  postprocess_one_shot_output_ = std::move(model_output);
  postprocess_one_shot_ = true;
  postprocess_enqueue_report_ = true;
  return HandlePostprocess();
}

Status ModelRunner::HandleResultReportProcess() {
  SetDevice(context_->device->GetDeviceId());

  while (!terminated_.load(std::memory_order_relaxed)) {
    std::shared_ptr<SamplingOutput> sampling_output = sampling_output_queue_->Get();
    if (!sampling_output) {
      KLLM_LOG_INFO << "HandleResultReportProcess stopped.";
      break;
    }

    // Only device 0 need report sampling output.
    if (context_->device->GetDeviceId() != 0) {
      continue;
    }

    if (context_->global->IsStandalone()) {
      ipc_func_wrapper_->ReportSamplingOutput(sampling_output);
    } else if (context_->global->IsDistributedResultReporter()) {
      sampling_output_pool_->PutSendSamplingOutput(sampling_output.get());
    }
  }
  return Status();
}

}  // namespace ksana_llm
