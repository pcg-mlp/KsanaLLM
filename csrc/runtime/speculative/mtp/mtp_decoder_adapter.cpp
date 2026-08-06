/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/mtp/mtp_decoder_adapter.h"

#include "configure/schedule_config_parser.h"
#include "executor/executor_runtime.h"
#include "executor/model_builder.h"
#include "executor/model_executor.h"
#include "executor/model_sampler.h"
#include "models/base/model_delegate.h"
#include "runtime/model_instance.h"
#include "runtime/speculative/common/draft_generator_interface.h"
#include "runtime/speculative/common/speculative_resource_profile.h"
#include "runtime/speculative/mtp/mtp_decoder.h"
#include "runtime/speculative/orchestration/composite_speculative_decoder.h"

namespace ksana_llm {

MtpDecoderAdapter::MtpDecoderAdapter(std::shared_ptr<MtpDecoder> mtp_decoder, ModelExecutor* model_executor,
                                     ModelSampler* model_sampler, ModelBuilder* model_builder)
    : mtp_decoder_(mtp_decoder),
      model_executor_(model_executor),
      model_sampler_(model_sampler),
      model_builder_(model_builder) {}

SpecSchedulingInfo MtpDecoderAdapter::GetSchedulingInfo() const {
  if (mtp_decoder_) {
    return mtp_decoder_->GetSchedulingInfo();
  }
  return {};
}

void MtpDecoderAdapter::PreMainForward(SpecContext& ctx) {
  if (!mtp_decoder_) return;

  // 1. MTP serialization: Wait for previous MTP forward, then arm for upcoming one.
  // Wait/Reset/Notify run on the same main inference thread, so no extra lock is needed —
  // the sequence Wait()→Reset(1)→…→Notify() is serialized across iterations by design.
  if (mtp_decoder_->IsSerializationRequired()) {
    auto waiter = mtp_decoder_->GetPrevMtpDoneWaiter();
    if (waiter) {
      waiter->Wait();
      waiter->Reset(1);
    }
  }

  // 2. Setup gather index builder and hidden buffer before main forward runs.
  if (ctx.forwarding_context) {
    mtp_decoder_->PrepareForMainForward(*ctx.forwarding_context);
  }
}

void MtpDecoderAdapter::BeforeForward(SpecContext& ctx) {
  // TODO(yancyliu): move MTP state-save here when execution is unified.
}

void MtpDecoderAdapter::ExecuteForward(SpecContext& ctx) {
  if (!mtp_decoder_ || !ctx.reqs || !ctx.hidden_unit || !ctx.schedule_output) return;
  mtp_decoder_->ExecuteForward(*ctx.reqs, ctx.hidden_unit, ctx.schedule_output, model_executor_, model_sampler_,
                               model_builder_);
}

void MtpDecoderAdapter::AfterForward(SpecContext& ctx) {
  // Notify handled internally by MtpDecoder::ExecuteForward.
}

void MtpDecoderAdapter::Shutdown() {
  if (mtp_decoder_) {
    mtp_decoder_->Shutdown();
  }
}

void MtpDecoderAdapter::Verify(SpecContext& ctx) {
  if (!ctx.reqs || !mtp_decoder_) return;
  for (auto& req : *ctx.reqs) {
    DefaultFilterDraftTokens(req);
  }
}

bool MtpDecoderAdapter::IsCombinableWith(const SpeculativeDecoderInterface* other) const { return true; }

namespace {

SpeculativeResourceProfile MtpEstimateProfile(const BatchSchedulerConfig& cfg) {
  SpeculativeResourceProfile profile;
  if (cfg.mtp_step_num > 0) {
    auto info = MtpDecoder::ComputeSchedulingInfo(cfg.mtp_step_num);
    profile.extra_decode_tokens_per_req = cfg.mtp_step_num;
    profile.workload_multiplier = info.workload_multiplier;
    profile.extra_token_capacity = info.extra_token_capacity;
    profile.extra_kv_tokens = info.extra_kv_tokens;
    // next-step hidden buffer is now owned and managed by each model (DeepSeekV4/V3/GLM4MoE).
  }
  return profile;
}

const bool kMtpResourceRegistered = []() {
  SpeculativeRegistry::Get().Register(MtpEstimateProfile);
  return true;
}();

// Post-init setup: registers a lazy-creation callback with ExecutorRuntime.
// MtpDecoder is created on first ExecutorRuntime::Initialize() callback execution,
// after ModelInstance::Load() and component construction are complete.
const bool kMtpPostInitRegistered = []() {
  SpeculativeRegistry::Get().RegisterPostInitSetup([](const SpeculativeSetupContext& ctx) {
    if (!ctx.context || !ctx.executor_runtime || !ctx.composite_decoder || !ctx.model_instance) return;
    // DSpark and MTP are mutually exclusive at the model layer
    // (deepseek_v4_model.cpp:432-449). Guard against MTP decoder creation
    // when the model is configured for DSpark (dspark_block_size > 0).
    if (ctx.context->global->GetDSparkBlockSize() > 0) return;
    size_t mtp_step_num = ctx.context->global->GetMtpStepNum();
    if (mtp_step_num == 0) return;
    ctx.executor_runtime->RegisterPostInitCallback([composite_decoder = ctx.composite_decoder,
                                                    model_instance = ctx.model_instance, context = ctx.context,
                                                    executor_runtime = ctx.executor_runtime, mtp_step_num]() {
      auto base_model = model_instance->GetBaseModel();
      if (!base_model) return;
      auto* draft_model = base_model->GetDraftModel();
      auto* delegate = dynamic_cast<ModelDelegate*>(base_model.get());
      if (!draft_model || !delegate) return;
      const bool mtp_serialize = context->global->IsSamplingNode() && context->global->GetMaxPpBatchNum() > 1;
      auto mtp_decoder = std::make_shared<MtpDecoder>(draft_model, mtp_step_num, mtp_serialize, delegate,
                                                      &executor_runtime->GetInferReqMutex());
      auto adapter = std::make_unique<MtpDecoderAdapter>(mtp_decoder, executor_runtime->GetModelExecutor().get(),
                                                         executor_runtime->GetModelSampler().get(),
                                                         executor_runtime->GetModelBuilder().get());
      composite_decoder->Register(std::move(adapter));
      KLLM_LOG_INFO << "MTP decoder created via post-init setup: mtp_step_num=" << mtp_step_num
                    << ", serialize=" << mtp_serialize;
    });
  });
  return true;
}();

}  // namespace

}  // namespace ksana_llm
