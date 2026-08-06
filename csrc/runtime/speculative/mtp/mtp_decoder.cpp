/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/mtp/mtp_decoder.h"

#include "executor/model_builder.h"
#include "executor/model_executor.h"
#include "executor/model_sampler.h"
#include "models/base/draft_model_interface.h"
#include "models/base/forwarding_context.h"
#include "models/base/model_delegate.h"
#include "runtime/executor_infer_request.h"
#include "runtime/model_input.h"
#include "runtime/model_output.h"
#include "runtime/sampling_output.h"
#include "runtime/schedule_output.h"
#include "runtime/speculative/mtp/mtp_gather_index_builder.h"
#include "runtime/speculative/mtp/mtp_state_guard.h"
#include "utils/logger.h"
#include "utils/waiter.h"

namespace ksana_llm {

MtpDecoder::MtpDecoder(DraftModelInterface* draft_model, size_t mtp_step_num, bool serialize_main_mtp,
                       ModelDelegate* model_delegate, std::mutex* infer_req_mutex)
    : draft_model_(draft_model),
      model_delegate_(model_delegate),
      mtp_step_num_(mtp_step_num),
      serialize_main_mtp_(serialize_main_mtp),
      prev_mtp_done_(std::make_shared<Waiter>(0)),
      infer_req_mutex_(infer_req_mutex) {
  KLLM_LOG_INFO << "MtpDecoder created: mtp_step_num=" << mtp_step_num_
                << ", serialize_main_mtp=" << serialize_main_mtp_;
}

MtpDecoder::~MtpDecoder() { Shutdown(); }

SpecSchedulingInfo MtpDecoder::ComputeSchedulingInfo(size_t step_num) {
  SpecSchedulingInfo info;
  info.extra_kv_tokens = step_num;
  info.extra_token_capacity = step_num - std::min(step_num, static_cast<size_t>(1));
  info.workload_multiplier = step_num;
  return info;
}

SpecSchedulingInfo MtpDecoder::GetSchedulingInfo() const { return ComputeSchedulingInfo(mtp_step_num_); }

Status MtpDecoder::ExecuteForward(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs,
                                  HiddenUnitDeviceBuffer* hidden_unit, std::shared_ptr<ScheduleOutput> schedule_output,
                                  ModelExecutor* model_executor, ModelSampler* model_sampler,
                                  ModelBuilder* model_builder) {
  if (mtp_step_num_ == 0 || reqs.empty()) {
    return Status();
  }

  KLLM_LOG_DEBUG << "MTP forward start: " << reqs.size() << " requests, mtp_step_num=" << mtp_step_num_;

  // Lock the mutex shared with ExecutorRuntime::UpdateCachedRequests (Preprocess thread) so that
  // thread cannot mutate forwarding_tokens/kv_cached_token_num/prefix_cache_len/
  // forwarding_tokens_draft_num on these same requests while this function runs. Acquired before
  // MtpStateGuard's construction so that, per C++ reverse-destruction-order, the guard's
  // destructor (which restores those fields) still runs while the lock is held — the entire
  // save/mutate/loop/restore sequence below is one atomic critical section, mirroring upstream's
  // MtpForward which holds the same-purpose mutex across its equivalent sequence.
  std::lock_guard<std::mutex> lock(*infer_req_mutex_);

  // Save state for all requests via RAII guard.
  // The guard restores forwarding_tokens, kv_cached_token_num, prefix_cache_len,
  // and forwarding_tokens_draft_num on scope exit, even if an exception is thrown.
  MtpStateGuard state_guard(reqs);

  // Prepare forwarding_tokens for the MTP loop. MTP sampling mode is now derived per-call from
  // model_output->run_mode (see ModelSampler::BuildSamplingRequest) instead of being persisted
  // on the request, so it no longer needs to be set/reset here.
  for (auto& infer_req : reqs) {
    infer_req->spec_state.draft_tokens.clear();
    infer_req->forwarding_tokens.resize(infer_req->forwarding_tokens.size() -
                                        infer_req->spec_state.forwarding_tokens_draft_num +
                                        infer_req->spec_state.accepted_tokens.size());
    infer_req->forwarding_tokens.insert(infer_req->forwarding_tokens.end(), infer_req->generated_tokens.begin(),
                                        infer_req->generated_tokens.end());
    infer_req->forwarding_tokens.erase(infer_req->forwarding_tokens.begin());
    // Keep kv_cached_token_num/prefix_cache_len consistent with the forwarding_tokens mutation above
    // (mirrors DSparkDecoder's equivalent preparation in dspark_decoder.cpp). Only the freshly generated
    // token(s) are uncached at MTP step 0; everything else (including the just-accepted draft prefix) is
    // treated as already cached. Without this, GetInputIdsLength() at step 0 would equal
    // `spec_state.accepted_tokens.size() + 1`, a value that varies per-request with the (probabilistic)
    // draft-acceptance count, breaking the invariant PrepareCompressor's IsPaged-ordering assertion
    // relies on (see deepseek_v4_compressor_builder.cpp PrepareCompressor).
    infer_req->kv_cached_token_num = infer_req->forwarding_tokens.size() - infer_req->generated_tokens.size();
    infer_req->prefix_cache_len = infer_req->kv_cached_token_num;
    infer_req->sampling_token_num = kStepGenerateTokenNum;
  }

  // MTP forward loop: one step per draft token.
  {
    // Get resources for MTP steps:
    // - CommonModel operations via ModelDelegate (injected at construction)
    // - model weights via ModelExecutor (for embedding/lm_head weights)
    auto base_weight = model_executor->GetModelWeight()->GetWeight();
    auto* fwd_ctx = model_delegate_->GetForwardingContext();
    model_delegate_->SetHiddenUnitBufferTensors(hidden_unit);

    for (size_t mtp_step = 0; mtp_step < mtp_step_num_; ++mtp_step) {
      KLLM_LOG_DEBUG << "MTP forward step: " << mtp_step;

      // Build model input for this MTP step. run_mode is set to kNextN so that
      // ModelSampler::Sampling (via model_output->run_mode) applies MTP-specific sampling
      // behavior (skip logprobs/no-repeat-ngram) only for this forward, not for subsequent
      // main-forward sampling calls on the same request.
      std::shared_ptr<NewModelInput> model_input = std::make_shared<NewModelInput>();
      model_builder->BuildModelInput(schedule_output, model_input);
      model_input->run_mode = RunMode::kNextN;

      // Prepare buffers, build attention meta, and parse model input.
      //
      // UpdateBeforeForward internally calls ParseFromRequests -> PrepareMainForwardGather ->
      // MtpGatherIndexBuilder::UpdateMainForward, which is meant to run ONLY for the actual main
      // forward (recording each request's starting position in the main-forward batch). But
      // ParseFromRequests is shared with the MTP forward path, so without guarding this, every
      // MTP step here would call UpdateMainForward again with this step's (tiny decode) batch,
      // clobbering the end-position that the previous step's BuildGatherIndices just computed —
      // corrupting the gather index for all steps after the first. Detach the external gather
      // index builder for the duration of this call so UpdateMainForward becomes a no-op here,
      // then re-attach it before PrepareMtpDraftGatherIndices, which is the call that actually
      // needs it (to consume the position recorded by the real main forward / previous step).
      auto* gather_idx_builder = fwd_ctx->GetModelInput()->GetExternalGatherIdxBuilder();
      fwd_ctx->GetModelInput()->SetExternalGatherIdxBuilder(nullptr);
      fwd_ctx->UpdateBeforeForward(model_input->reqs, ksana_llm::RunMode::kMain, /*model_input_slot=*/0,
                                   /*gpu_input_already_prepared=*/false);
      fwd_ctx->GetModelInput()->SetExternalGatherIdxBuilder(gather_idx_builder);
      // Build MTP-specific gather indices (replaces the removed is_mtp_step=true path).
      fwd_ctx->GetModelInput()->PrepareMtpDraftGatherIndices(model_input->reqs);

      // Run forward pipeline: Embed → MTP Layer → SaveNextStepHidden → MHC/Norm →
      // SaveNextStepNormedHidden → LmHeadPost.
      // Step 1: Embed the draft token (skip input_refit and initial MHC).
      Tensor embedding_weight = base_weight->GetModelWeights("model.embed_tokens.weight");
      if (embedding_weight.location == MemoryLocation::LOCATION_DEVICE) {
        model_delegate_->EmbedTokensUseGpu(embedding_weight, *fwd_ctx);
      }

      // Step 2: MTP decoder layer via DraftModelInterface.
      auto& residual_buffer = model_delegate_->GetHiddenUnitBufferTensors(*fwd_ctx);
      fwd_ctx->SetIsForwardingLayers(true);
      draft_model_->ForwardLayer(residual_buffer, *fwd_ctx);
      fwd_ctx->SetIsForwardingLayers(false);

      // Step 3: Save hidden states for speculative decoding (next MTP step input).
      // Must be called AFTER ForwardLayer (which produces the hidden state) and
      // BEFORE ForwardMhcNormLmHead (which squeezes the MHC dimension).
      model_delegate_->SaveNextStepHiddenBuffer(*fwd_ctx);

      // Step 4: MTP-specific MHC squeeze + final norm via DraftModelInterface.
      draft_model_->ForwardMhcNormLmHead(residual_buffer, base_weight, *fwd_ctx);

      // Step 4b: Save normed hidden after MHC squeeze+norm for models that
      // save after normalization (return_hidden_states_before_norm=false).
      model_delegate_->SaveNextStepNormedHiddenBuffer(*fwd_ctx);

      // Step 5: Assemble → LmHead → Cast → logits via CommonModel.
      CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, fwd_ctx->GetForwardingBuffers()->hidden_buffer_0);
      model_delegate_->LmHeadPost(*fwd_ctx, base_weight, model_input->reqs,
                                  /*assemble_already_done=*/false, hidden_buffer_tensors_0);

      // Build model_output for sampling (same fields that ModelExecutor::Forward sets).
      std::shared_ptr<NewModelOutput> model_output = std::make_shared<NewModelOutput>();
      model_output->schedule_output = schedule_output;
      // Mirror ModelExecutor::Forward's run_mode propagation, since this model_output is built
      // manually here rather than going through ModelExecutor::Forward.
      model_output->run_mode = model_input->run_mode;
      model_output->frozen_logits_offsets.reserve(model_input->reqs.size());
      for (const auto& fr : model_input->reqs) {
        model_output->frozen_logits_offsets.push_back(fr.logits_offset);
      }
      model_output->slot_index = model_input->slot_index;

      // Sample the next token.
      std::shared_ptr<SamplingOutput> sampling_output = std::make_shared<SamplingOutput>();
      model_sampler->Sampling(model_output, sampling_output);

      // Collect the generated draft token and update state for the next step.
      for (auto& infer_req : reqs) {
        infer_req->spec_state.draft_tokens.mtp.emplace_back(infer_req->sampling_result_tokens.back());
        if (mtp_step != mtp_step_num_ - 1) {
          infer_req->kv_cached_token_num = infer_req->forwarding_tokens.size();
          infer_req->prefix_cache_len = infer_req->kv_cached_token_num;
          infer_req->forwarding_tokens.emplace_back(infer_req->spec_state.draft_tokens.mtp.back());
          infer_req->sampling_result_tokens.clear();
        }
      }
    }
  }

  // MtpStateGuard destructor restores forwarding_tokens, kv_cached_token_num,
  // prefix_cache_len, and forwarding_tokens_draft_num here.

  // Log MTP results.
  for (const auto& infer_req : reqs) {
    KLLM_LOG_DEBUG << "MTP forward done: req_id=" << infer_req->req_id
                   << ", draft_tokens=" << infer_req->spec_state.draft_tokens.mtp.size()
                   << ", accepted_tokens=" << infer_req->spec_state.accepted_tokens.size();
  }

  // Release the next main Forward() waiting on prev_mtp_done_.
  if (serialize_main_mtp_) {
    prev_mtp_done_->Notify();
  }

  // Cleanup external buffer override.
  CleanupAfterMtpForward(*model_delegate_->GetForwardingContext());

  return Status();
}

void MtpDecoder::Shutdown() {
  if (prev_mtp_done_) {
    prev_mtp_done_->Stop();
  }
}

void MtpDecoder::PrepareForMainForward(ForwardingContext& fwd_ctx) {
  auto* fwd_buffers = fwd_ctx.GetForwardingBuffers();
  if (!fwd_buffers) return;

  // Lazy-init the gather index builder.
  if (!gather_idx_builder_) {
    const size_t max_token_num = fwd_buffers->runtime_config.max_step_token_num;
    gather_idx_builder_ =
        std::make_unique<MtpGatherIndexBuilder>(max_token_num, fwd_ctx.GetContext(), fwd_ctx.GetCurrentRank());
  }

  // Inject the gather index builder into ModelInput.
  if (gather_idx_builder_) {
    fwd_ctx.GetModelInput()->SetExternalGatherIdxBuilder(gather_idx_builder_.get());
  }
}

void MtpDecoder::CleanupAfterMtpForward(ForwardingContext& fwd_ctx) {
  fwd_ctx.GetModelInput()->SetExternalGatherIdxBuilder(nullptr);
}

}  // namespace ksana_llm