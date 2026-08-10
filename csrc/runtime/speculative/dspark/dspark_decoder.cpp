/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/dspark/dspark_decoder.h"

#include "executor/model_builder.h"
#include "executor/model_executor.h"
#include "executor/model_sampler.h"
#include "models/base/base_model.h"
#include "runtime/executor_infer_request.h"
#include "runtime/model_input.h"
#include "runtime/model_instance.h"
#include "runtime/model_output.h"
#include "runtime/sampling_output.h"
#include "runtime/schedule_output.h"
#include "runtime/speculative/mtp/mtp_state_guard.h"
#include "utils/logger.h"
#include "utils/waiter.h"

namespace ksana_llm {

DSparkDecoder::DSparkDecoder(size_t dspark_block_size, int dspark_noise_token_id, std::mutex* infer_req_mutex,
                             bool serialize_dspark)
    : dspark_block_size_(dspark_block_size),
      dspark_noise_token_id_(dspark_noise_token_id),
      serialize_dspark_(serialize_dspark),
      prev_dspark_done_(std::make_shared<Waiter>(0)),
      infer_req_mutex_(infer_req_mutex) {
  KLLM_LOG_INFO << "DSparkDecoder created: block_size=" << dspark_block_size_ << ", noise_id=" << dspark_noise_token_id_
                << ", serialize=" << serialize_dspark_;
}

SpecSchedulingInfo DSparkDecoder::ComputeSchedulingInfo(size_t block_size) {
  SpecSchedulingInfo info;
  info.extra_kv_tokens = block_size;
  info.extra_token_capacity = block_size;
  info.workload_multiplier = block_size;
  return info;
}

SpecSchedulingInfo DSparkDecoder::GetSchedulingInfo() const { return ComputeSchedulingInfo(dspark_block_size_); }

Status DSparkDecoder::ExecuteForward(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs,
                                     HiddenUnitDeviceBuffer* hidden_unit,
                                     std::shared_ptr<ScheduleOutput> schedule_output, ModelExecutor* model_executor,
                                     ModelSampler* model_sampler, ModelBuilder* model_builder,
                                     ModelInstance* model_instance) {
  if (dspark_block_size_ == 0 || reqs.empty()) {
    if (serialize_dspark_) {
      prev_dspark_done_->Notify();
    }
    return Status();
  }

  KLLM_LOG_DEBUG << "DSpark forward start: " << reqs.size() << " requests, block_size=" << dspark_block_size_;

  // Lock the mutex shared with ExecutorRuntime::UpdateCachedRequests (Preprocess thread) so that
  // thread cannot mutate forwarding_tokens/kv_cached_token_num/prefix_cache_len/
  // forwarding_tokens_draft_num on these same requests while this function runs. Acquired before
  // MtpStateGuard's construction so that, per C++ reverse-destruction-order, the guard's
  // destructor (which restores those fields) still runs while the lock is held.
  std::lock_guard<std::mutex> lock(*infer_req_mutex_);

  // Save state for all requests via RAII guard (same 4 fields as MTP).
  MtpStateGuard state_guard(reqs);

  // Prepare forwarding_tokens: pad with noise tokens to block_size.
  for (auto& infer_req : reqs) {
    infer_req->spec_state.draft_tokens.clear();
    infer_req->forwarding_tokens.resize(infer_req->forwarding_tokens.size() -
                                        infer_req->spec_state.forwarding_tokens_draft_num +
                                        infer_req->spec_state.accepted_tokens.size());
    infer_req->prefix_cache_len = infer_req->kv_cached_token_num = infer_req->forwarding_tokens.size();
    infer_req->forwarding_tokens.insert(infer_req->forwarding_tokens.end(), infer_req->generated_tokens.begin(),
                                        infer_req->generated_tokens.end());
    // DSpark: no erase(begin); pad with noise tokens to block_size after generated tokens.
    infer_req->forwarding_tokens.insert(infer_req->forwarding_tokens.end(),
                                        dspark_block_size_ - infer_req->generated_tokens.size(),
                                        dspark_noise_token_id_);
    infer_req->sampling_token_num = dspark_block_size_;
  }

  // Single DSpark forward with non-causal attention.
  std::shared_ptr<NewModelInput> model_input = std::make_shared<NewModelInput>();
  model_builder->BuildModelInput(schedule_output, model_input);
  model_input->run_mode = RunMode::kDSpark;

  std::shared_ptr<NewModelOutput> model_output = std::make_shared<NewModelOutput>();
  model_executor->Forward(model_input, hidden_unit, model_output);
  // model_output->run_mode is propagated from model_input->run_mode inside ModelExecutor::Forward.
  // Logits tensor: [dspark_block_size, batch_size, vocab_size] (step-major),
  // accessible via model's internal forwarding_context_->GetModelOutput()->logits_tensor.

  // Per-step head-only sampling: ForwardHead corrects logits via markov bias, then sample.
  const size_t batch_size = reqs.size();
  for (size_t step_index = 0; step_index < dspark_block_size_; ++step_index) {
    KLLM_LOG_DEBUG << "DSpark step: " << step_index;

    // Apply markov bias correction to logits for this step.
    // ForwardHead reads from the model's internal forwarding_context_->GetModelOutput()->logits_tensor.
    model_instance->ForwardHead(model_input->reqs, step_index);

    for (size_t i = 0; i < batch_size; ++i) {
      reqs[i]->sampling_token_num = kStepGenerateTokenNum;
      model_output->frozen_logits_offsets[i] = i;
    }

    // Sample one token per request from this step's [batch_size, V] logits row.
    std::shared_ptr<SamplingOutput> sampling_output = std::make_shared<SamplingOutput>();
    model_sampler->Sampling(model_output, sampling_output);

    // Collect the generated draft token and update state for the next step.
    for (auto& infer_req : reqs) {
      infer_req->spec_state.draft_tokens.mtp.emplace_back(infer_req->sampling_result_tokens.back());
      if (step_index < dspark_block_size_ - 1) {
        // Replace the noise token at the corresponding position in forwarding_tokens.
        infer_req->forwarding_tokens[infer_req->forwarding_tokens.size() - dspark_block_size_ + step_index + 1] =
            infer_req->sampling_result_tokens.back();
        infer_req->sampling_result_tokens.clear();
      }
    }
  }

  // MtpStateGuard destructor restores forwarding_tokens, kv_cached_token_num,
  // prefix_cache_len, and forwarding_tokens_draft_num here.

  for (const auto& infer_req : reqs) {
    KLLM_LOG_DEBUG << "DSpark forward done: req_id=" << infer_req->req_id
                   << ", draft_tokens=" << infer_req->spec_state.draft_tokens.mtp.size();
  }

  // Release the next main Forward() waiting on prev_dspark_done_.
  // Paired with the Wait()/Reset(1) in DSparkDecoderAdapter::PreMainForward.
  if (serialize_dspark_) {
    prev_dspark_done_->Notify();
  }

  return Status();
}

void DSparkDecoder::Shutdown() {
  if (prev_dspark_done_) {
    prev_dspark_done_->Stop();
  }
}

}  // namespace ksana_llm
