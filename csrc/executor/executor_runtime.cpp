/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/executor_runtime.h"
#include "models/base/model_delegate.h"
#include "models/common/common_model.h"
#include "runtime/speculative/orchestration/composite_speculative_decoder.h"
#include "runtime/structured_generation/xgrammar/xgrammar_structured_generator_creator.h"
#include "utils/logger.h"

namespace ksana_llm {

ExecutorRuntime::ExecutorRuntime(std::shared_ptr<Environment> env, std::shared_ptr<Context> context) {
  env_ = env;
  context_ = context;
  executor_request_manager_ = std::make_shared<ExecutorRequestManager>();
  generation_controller_ = std::make_shared<GenerationController>(nullptr);

  // The inflight resource manager does not depend on any external objects; create it early so
  // that it is always available regardless of whether Initialize() is called.
  inflight_resource_mgr_ = std::make_shared<InflightResourceManager>();

  KLLM_LOG_INFO << "ExecutorRuntime initialized";
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

void ExecutorRuntime::SetCompositeSpeculativeDecoder(std::shared_ptr<CompositeSpeculativeDecoder> composite_decoder) {
  composite_decoder_ = std::move(composite_decoder);
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

  // Execute post-init callbacks (e.g. speculative decoder lazy creation).
  for (auto& cb : post_init_cbs_) {
    cb();
  }

  // Verify speculative decoder compatibility after all decoders are registered.
  // Fail-fast on incompatible combinations (e.g. MTP+DSpark, Trie+ASR) that
  // would corrupt draft_tokens at runtime.
  if (composite_decoder_ && !composite_decoder_->AreAllCompatible()) {
    KLLM_LOG_ERROR << "Incompatible speculative decoders detected at startup";
    return Status(RET_RUNTIME_FAILED, "Incompatible speculative decoders registered");
  }

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
                                        std::shared_ptr<NewModelInput> model_input) {
  return model_builder_->BuildModelInput(schedule_output, model_input);
}

Status ExecutorRuntime::PrepareModelInputGpu(std::shared_ptr<NewModelInput> model_input) {
  if (model_instance_ == nullptr || model_input == nullptr) {
    return Status(RET_INVALID_ARGUMENT, "PrepareModelInputGpu invalid argument.");
  }
  if (!context_->device->IsDualH2DStreamsEnabled()) {
    return Status();
  }
  Status status =
      model_instance_->PrepareModelInputGpu(model_input->reqs, ksana_llm::RunMode::kMain, model_input->slot_index);
  if (status.OK()) {
    model_input->gpu_input_ready = true;
  }
  return status;
}

Status ExecutorRuntime::Forward(std::shared_ptr<NewModelInput> model_input, HiddenUnitDeviceBuffer* hidden_unit,
                                std::shared_ptr<NewModelOutput> model_output) {
  // Decouple MTP serialization and hidden buffer injection via PreMainForward hook.
  if (composite_decoder_) {
    SpecContext ctx;
    if (model_instance_) {
      auto* delegate = dynamic_cast<ModelDelegate*>(model_instance_->GetBaseModel().get());
      if (delegate) {
        ctx.forwarding_context = delegate->GetForwardingContext();
      }
    }
    composite_decoder_->PreMainForward(ctx);
  }

  return model_executor_->Forward(model_input, hidden_unit, model_output);
}

Status ExecutorRuntime::Sampling(std::shared_ptr<NewModelOutput> model_output,
                                 std::shared_ptr<SamplingOutput> sampling_output) {
  return model_sampler_->Sampling(model_output, sampling_output);
}

void ExecutorRuntime::Shutdown() {
  // Wake the execute thread if it is currently blocked on prev_mtp_done_->Wait()
  // inside PreMainForward. Without this, ModelRunner::Stop()'s subsequent join
  // call will hang because HandlePostprocess (which would deliver the Notify) has
  // already been stopped.
  if (composite_decoder_) {
    composite_decoder_->Shutdown();
  }
}

}  // namespace ksana_llm
