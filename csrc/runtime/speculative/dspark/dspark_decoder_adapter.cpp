/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/dspark/dspark_decoder_adapter.h"

#include "configure/schedule_config_parser.h"
#include "executor/executor_runtime.h"
#include "executor/model_builder.h"
#include "executor/model_executor.h"
#include "executor/model_sampler.h"
#include "runtime/model_instance.h"
#include "runtime/speculative/common/draft_generator_interface.h"
#include "runtime/speculative/common/speculative_resource_profile.h"
#include "runtime/speculative/dspark/dspark_decoder.h"
#include "runtime/speculative/orchestration/composite_speculative_decoder.h"
#include "utils/logger.h"

namespace ksana_llm {

DSparkDecoderAdapter::DSparkDecoderAdapter(std::shared_ptr<DSparkDecoder> dspark_decoder, ModelExecutor* model_executor,
                                           ModelSampler* model_sampler, ModelBuilder* model_builder,
                                           ModelInstance* model_instance)
    : dspark_decoder_(std::move(dspark_decoder)),
      model_executor_(model_executor),
      model_sampler_(model_sampler),
      model_builder_(model_builder),
      model_instance_(model_instance) {}

SpecSchedulingInfo DSparkDecoderAdapter::GetSchedulingInfo() const {
  if (dspark_decoder_) {
    return dspark_decoder_->GetSchedulingInfo();
  }
  return {};
}

void DSparkDecoderAdapter::ExecuteForward(SpecContext& ctx) {
  if (!dspark_decoder_ || !ctx.reqs || !ctx.hidden_unit || !ctx.schedule_output) return;
  dspark_decoder_->ExecuteForward(*ctx.reqs, ctx.hidden_unit, ctx.schedule_output, model_executor_, model_sampler_,
                                  model_builder_, model_instance_);
}

void DSparkDecoderAdapter::PreMainForward(SpecContext& ctx) {
  if (!dspark_decoder_) return;

  // Serialize main forward against DSpark forward:
  // 1) Protect the shared dspark_hidden_tensor_ buffer — main forward writes
  //    cross-layer hidden states, DSpark forward reads them for markov-head correction.
  //    Without serialization, the next decode step's main forward (Execute thread) can
  //    overwrite the buffer while DSpark forward (Postprocess thread) is still reading it.
  // 2) Guard against TrtAllReduce-ordering deadlock on PP > 1 sampling ranks.
  // Wait/Reset/Notify run on the same main inference thread, so no extra lock is needed.
  if (dspark_decoder_->IsSerializationRequired()) {
    auto waiter = dspark_decoder_->GetPrevDsparkDoneWaiter();
    if (waiter) {
      waiter->Wait();
      waiter->Reset(1);
    }
  }
}

void DSparkDecoderAdapter::Shutdown() {
  if (dspark_decoder_) {
    dspark_decoder_->Shutdown();
  }
}

void DSparkDecoderAdapter::Verify(SpecContext& ctx) {
  if (!ctx.reqs) return;
  for (auto& req : *ctx.reqs) {
    DefaultFilterDraftTokens(req);
  }
}

bool DSparkDecoderAdapter::IsCombinableWith(const SpeculativeDecoderInterface* other) const {
  if (!other) return true;
  return strcmp(other->GetName(), "MTP") != 0;
}

// Reverse static registration: DSpark resource estimation.
// Parser-phase estimator: mtp_step_num proxies dspark_block_size
// (overridden in schedule_config_parser.cpp:937-949).
// False-positive in MTP models is harmless — the DSpark adapter
// rejects registration at post-init time via GetDSparkBlockSize().
namespace {

SpeculativeResourceProfile DSparkEstimateProfile(const BatchSchedulerConfig& cfg) {
  SpeculativeResourceProfile profile;
  if (cfg.mtp_step_num > 0) {
    // cfg.mtp_step_num overrides dspark_block_size at config parse time
    // (schedule_config_parser.cpp:944-954), so it is the canonical block_size here.
    auto info = DSparkDecoder::ComputeSchedulingInfo(cfg.mtp_step_num);
    profile.extra_decode_tokens_per_req = cfg.mtp_step_num;
    profile.workload_multiplier = info.workload_multiplier;
    profile.extra_token_capacity = info.extra_token_capacity;
    profile.extra_kv_tokens = info.extra_kv_tokens;
  }
  return profile;
}

const bool kDSparkResourceRegistered = []() {
  SpeculativeRegistry::Get().Register(DSparkEstimateProfile);
  return true;
}();

// Post-init setup: lazy-creation callback registered with ExecutorRuntime.
// DSparkDecoder is created on first ExecutorRuntime::Initialize() callback execution,
// after ModelInstance::Load() and component construction are complete.
const bool kDSparkPostInitRegistered = []() {
  SpeculativeRegistry::Get().RegisterPostInitSetup([](const SpeculativeSetupContext& ctx) {
    if (!ctx.context || !ctx.executor_runtime || !ctx.composite_decoder || !ctx.model_instance) return;
    size_t block_size = ctx.context->global->GetDSparkBlockSize();
    if (block_size == 0) return;
    ctx.executor_runtime->RegisterPostInitCallback([composite_decoder = ctx.composite_decoder,
                                                    model_instance = ctx.model_instance, context = ctx.context,
                                                    executor_runtime = ctx.executor_runtime, block_size]() {
      int noise_id = context->global->GetDSparkNoiseTokenId();
      // Main forward and DSpark's own forward share a single model-owned cross-layer
      // hidden-state capture buffer (DeepSeekV4Model::dspark_hidden_tensor_), written by
      // the main forward and read back by DSpark's markov-head correction. Without
      // serialization, ExecutorRuntime::Forward() for the next decode step (Execute thread)
      // can start overwriting that buffer while DSparkDecoder::ExecuteForward (Postprocess
      // thread) is still reading it, corrupting the draft prediction for any request whose
      // GPU work hasn't completed yet — this is unrelated to the TrtAllReduce-ordering
      // required whenever DSpark is active, regardless of pipeline-parallel batch depth.
      const bool serialize_dspark = context->global->IsSamplingNode();
      auto dspark_decoder = std::make_shared<DSparkDecoder>(block_size, noise_id, &executor_runtime->GetInferReqMutex(),
                                                            serialize_dspark);
      auto adapter = std::make_unique<DSparkDecoderAdapter>(
          dspark_decoder, executor_runtime->GetModelExecutor().get(), executor_runtime->GetModelSampler().get(),
          executor_runtime->GetModelBuilder().get(), model_instance.get());
      composite_decoder->Register(std::move(adapter));
      KLLM_LOG_INFO << "DSpark decoder registered: block_size=" << block_size << ", noise_id=" << noise_id
                    << ", serialize=" << serialize_dspark;
    });
  });
  return true;
}();

}  // namespace

}  // namespace ksana_llm
