/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/executor_runtime.h"
#include "runtime/structured_generation/xgrammar/xgrammar_structured_generator_creator.h"

namespace ksana_llm {

ExecutorRuntime::ExecutorRuntime(std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : serialize_main_mtp_(context->global->IsSamplingNode() && context->global->GetMtpStepNum() > 0 &&
                          context->global->GetMaxPpBatchNum() > 1) {
  env_ = env;
  context_ = context;
  executor_request_manager_ = std::make_shared<ExecutorRequestManager>();
  generation_controller_ = std::make_shared<GenerationController>(nullptr);

  // Waiter starts in the "ready" state (num_wait == 0) so that the very first main Forward()
  // returns from Wait() immediately without needing a paired Notify(). Subsequent iterations are
  // armed by Reset(1) inside Forward() and released by Notify() at the end of MtpForward().
  prev_mtp_done_ = std::make_shared<Waiter>(0);

  KLLM_LOG_INFO << "ExecutorRuntime serialize_main_mtp=" << serialize_main_mtp_
                << " (IsSamplingNode=" << context_->global->IsSamplingNode()
                << ", mtp_step_num=" << context_->global->GetMtpStepNum()
                << ", max_pp_batch_num=" << context_->global->GetMaxPpBatchNum() << ")";
}

void ExecutorRuntime::SetModelInstance(std::shared_ptr<ModelInstance> model_instance) {
  model_instance_ = model_instance;
}

void ExecutorRuntime::SetModelWeight(std::shared_ptr<WeightInstanceInterface> model_weight) {
  model_weight_ = model_weight;
}

void ExecutorRuntime::SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager) {
  block_allocator_manager_ = block_allocator_manager;
}

void ExecutorRuntime::SetSwaBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager) {
  swa_block_allocator_manager_ = swa_block_allocator_manager;
}

void ExecutorRuntime::SetHiddenUnitBufferPool(std::shared_ptr<HiddenUnitBufferPool> hidden_unit_buffer_pool) {
  hidden_unit_buffer_pool_ = hidden_unit_buffer_pool;
}

void ExecutorRuntime::SetDraftGeneratorController(
    std::shared_ptr<DraftGeneratorController> draft_generator_controller) {
  generation_controller_->SetDraftGeneratorController(draft_generator_controller);
}

Status ExecutorRuntime::InitializeStructuredGeneratorFactory() {
  BatchSchedulerConfig batch_scheduler_config;
  env_->GetBatchSchedulerConfig(batch_scheduler_config);
  if (batch_scheduler_config.enable_xgrammar) {
    const VocabInfo& vocab_info = env_->GetVocabInfoRef();
    if (!vocab_info.initialized) {
      KLLM_LOG_ERROR << "Vocab info not initialized in Environment";
      return Status(RET_RUNTIME_FAILED, "Vocab info not initialized");
    }
    try {
      // Create StructuredGeneratorFactory with vocab_info (including think_end_token_id for reasoning mode)
      structured_generator_factory_ = std::make_shared<StructuredGeneratorFactory>(vocab_info);
      KLLM_LOG_INFO << "Created structured generator factory with vocab_size: " << vocab_info.vocab_size
                    << ", think_end_token_id: " << vocab_info.think_end_token_id;

      // Create GrammarGeneratorCreator with VocabInfo reference
      auto creator = std::make_unique<GrammarGeneratorCreator>(vocab_info);
      structured_generator_factory_->RegisterCreator(StructuredConstraintType::JSON, std::move(creator));
      KLLM_LOG_INFO << "Structured generator factory initialized successfully";
      generation_controller_->SetStructuredGeneratorFactory(structured_generator_factory_);
    } catch (const std::exception& e) {
      KLLM_LOG_ERROR << "Failed to init structured generator factory: " << e.what();
      structured_generator_factory_ = nullptr;
    }
  }
  return Status();
}

Status ExecutorRuntime::Initialize() {
  // Allocate necessary device memory after block manager initialized
  hidden_unit_buffer_pool_->InitializeHiddenUnitBuffer();

  // Initialize structured generator factory if xgrammar is enabled
  Status status = InitializeStructuredGeneratorFactory();
  if (!status.OK()) {
    return status;
  }
  model_builder_ = std::make_shared<ModelBuilder>(context_);
  model_builder_->SetModelInstance(model_instance_);
  model_builder_->SetBlockAllocatorManager(block_allocator_manager_);
  if (swa_block_allocator_manager_) {
    model_builder_->SetSwaBlockAllocatorManager(swa_block_allocator_manager_);
  }

  model_executor_ = std::make_shared<ModelExecutor>();
  model_executor_->SetModelInstance(model_instance_);
  model_executor_->SetModelWeight(model_weight_);
  model_executor_->SetBlockAllocatorManager(block_allocator_manager_);

  model_sampler_ = std::make_shared<ModelSampler>(env_, context_);
  return Status();
}

void ExecutorRuntime::UpdateCachedRequests(std::shared_ptr<ScheduleOutput> schedule_output) {
  for (auto& vec : schedule_output->finish_req_ids) {
    for (auto req_id : vec) {
      executor_request_manager_->EraseRequest(req_id);
    }
  }

  for (auto& req : schedule_output->executor_running_reqs) {
    executor_request_manager_->AddInferRequest(req->req_id, req);
  }

  // Guard infer_req against MTPForward
  std::lock_guard<std::mutex> infer_req_lock(infer_req_mutex_);
  for (auto& req_patch : schedule_output->running_req_patches) {
    executor_request_manager_->UpdateInferRequest(req_patch->req_id, req_patch);
  }
}

void ExecutorRuntime::EvictCompletedCompressorSlots(const std::shared_ptr<ScheduleOutput>& schedule_output) {
  // IMPORTANT: This must be called from the execute thread (in lock-step with Forward), NOT from
  // the preprocess thread. Reason: HandlePreprocess and HandleExecute are pipelined threads; the
  // preprocess thread can race ahead and queue several model_inputs while the execute thread is
  // still forwarding earlier ones. If we evicted here in the preprocess path, a later SO's
  // finish_req_ids could release a request's compressor slot before the execute thread reaches an
  // earlier SO's Forward (which still needs that slot for Prepare()) — producing a spurious
  // "req appears without a slot but kv > 0" assertion.
  if (!model_instance_) {
    return;
  }
  std::vector<int64_t> evicted_req_ids;
  for (const auto& vec : schedule_output->finish_req_ids) {
    evicted_req_ids.insert(evicted_req_ids.end(), vec.begin(), vec.end());
  }
  for (const auto& vec : schedule_output->recompute_req_ids) {
    evicted_req_ids.insert(evicted_req_ids.end(), vec.begin(), vec.end());
  }
  if (evicted_req_ids.empty()) {
    return;
  }
  model_instance_->EvictFinishedRequests(evicted_req_ids);
}

void ExecutorRuntime::RebuildScheduleOutput(std::shared_ptr<ScheduleOutput> schedule_output) {
  for (auto& req_patch : schedule_output->running_req_patches) {
    std::shared_ptr<ExecutorInferRequest> req = executor_request_manager_->GetInferRequest(req_patch->req_id);
    if (req == nullptr) {
      KLLM_LOG_ERROR << "The req " << req_patch->req_id << " is not exists in cache. ";
    }
    schedule_output->executor_running_reqs.push_back(req);
  }
}

Status ExecutorRuntime::BuildModelInput(std::shared_ptr<ScheduleOutput> schedule_output,
                                        std::shared_ptr<NewModelInput> model_input, const RunMode run_mode) {
  return model_builder_->BuildModelInput(schedule_output, model_input, run_mode);
}

Status ExecutorRuntime::PrepareModelInputGpu(std::shared_ptr<NewModelInput> model_input) {
  if (model_instance_ == nullptr || model_input == nullptr) {
    return Status(RET_INVALID_ARGUMENT, "PrepareModelInputGpu invalid argument.");
  }
  if (!context_->device->IsDualH2DStreamsEnabled()) {
    return Status();
  }
  Status status =
      model_instance_->PrepareModelInputGpu(model_input->reqs, model_input->run_mode, model_input->slot_index);
  if (status.OK()) {
    model_input->gpu_input_ready = true;
  }
  return status;
}

Status ExecutorRuntime::Forward(std::shared_ptr<NewModelInput> model_input, HiddenUnitDeviceBuffer* hidden_unit,
                                std::shared_ptr<NewModelOutput> model_output) {
  // Serialize the main forward (RunMode::kMain) against the previous iteration's MtpForward
  // (RunMode::kNextN). Without this gate, on a sampling rank with TP > 1 and max_pp_batch_num >= 2
  // the per-rank CommonModel::forward_mutex_ can be acquired in different order on different TP
  // ranks (kMain on one rank vs. kNextN on its peer), causing both ranks to enter different
  // TrtAllReduce collectives and deadlock. See common_model.cpp:328 / trt_all_reduce_sum_layer.cpp.
  // Wait order: first wait for prior Notify() (no-op on the first iteration since the Waiter starts
  // at zero), then arm the Waiter for the upcoming MtpForward by Reset(1).
  if (serialize_main_mtp_) {
    prev_mtp_done_->Wait();
    prev_mtp_done_->Reset(1);
  }
  return model_executor_->Forward(model_input, hidden_unit, model_output);
}

Status ExecutorRuntime::Sampling(std::shared_ptr<NewModelOutput> model_output,
                                 std::shared_ptr<SamplingOutput> sampling_output) {
  return model_sampler_->Sampling(model_output, sampling_output);
}

Status ExecutorRuntime::MtpForward(const size_t mtp_step_num, std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs,
                                   HiddenUnitDeviceBuffer* hidden_unit,
                                   std::shared_ptr<ScheduleOutput> schedule_output) {
  if (mtp_step_num == 0) {
    return Status();
  }

  // update forward request
  std::unordered_map<decltype(ExecutorInferRequest::req_id), size_t> original_pos;
  std::unordered_map<decltype(ExecutorInferRequest::req_id), size_t> original_kv;
  std::unordered_map<decltype(ExecutorInferRequest::req_id), std::vector<int>> original_tokens;
  original_pos.reserve(reqs.size());
  for (size_t i = 0; i < reqs.size(); ++i) {
    auto& infer_req = reqs[i];
    infer_req->draft_tokens.clear();
    original_pos[infer_req->req_id] = i;
    original_kv[infer_req->req_id] = infer_req->kv_cached_token_num;
    original_tokens[infer_req->req_id] = infer_req->forwarding_tokens;
    infer_req->forwarding_tokens.resize(infer_req->forwarding_tokens.size() - infer_req->forwarding_tokens_draft_num +
                                        infer_req->accepted_tokens.size());
    infer_req->forwarding_tokens.insert(infer_req->forwarding_tokens.end(), infer_req->generated_tokens.begin(),
                                        infer_req->generated_tokens.end());
    infer_req->forwarding_tokens.erase(infer_req->forwarding_tokens.begin());
    infer_req->sampling_token_num = kStepGenerateTokenNum;
  }

  // Guard infer_req against UpdateInferRequest
  std::lock_guard<std::mutex> infer_req_lock(infer_req_mutex_);
  for (size_t mtp_step = 0; mtp_step < mtp_step_num; ++mtp_step) {
    KLLM_LOG_DEBUG << "MTP forward step: " << mtp_step;
    std::shared_ptr<NewModelInput> model_input = std::make_shared<NewModelInput>();
    BuildModelInput(schedule_output, model_input, RunMode::kNextN);
    std::shared_ptr<NewModelOutput> model_output = std::make_shared<NewModelOutput>();
    model_executor_->Forward(model_input, hidden_unit, model_output);
    std::shared_ptr<SamplingOutput> sampling_output = std::make_shared<SamplingOutput>();
    model_sampler_->Sampling(model_output, sampling_output);

    for (auto& infer_req : reqs) {
      infer_req->draft_tokens.mtp.emplace_back(infer_req->sampling_result_tokens.back());
      if (mtp_step != mtp_step_num - 1) {
        infer_req->kv_cached_token_num = infer_req->forwarding_tokens.size();
        infer_req->prefix_cache_len = infer_req->kv_cached_token_num;
        infer_req->forwarding_tokens.emplace_back(infer_req->draft_tokens.mtp.back());
        infer_req->sampling_result_tokens.clear();
      }
    }
  }

  for (auto& req : reqs) {
    req->forwarding_tokens = original_tokens[req->req_id];
    req->kv_cached_token_num = original_kv[req->req_id];
    req->prefix_cache_len = req->kv_cached_token_num;
  }

  // Release the next main Forward() that has been waiting on prev_mtp_done_. Paired with the
  // Wait()/Reset(1) at the top of Forward(); see the comment there for the deadlock context.
  if (serialize_main_mtp_) {
    prev_mtp_done_->Notify();
  }

  return Status();
}

void ExecutorRuntime::Shutdown() {
  if (prev_mtp_done_) {
    prev_mtp_done_->Stop();
  }
}

}  // namespace ksana_llm
