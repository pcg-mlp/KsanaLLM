/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/model_sampler.h"

#include <climits>

namespace ksana_llm {

constexpr int kBitsPerInt = sizeof(int) * CHAR_BIT;

ModelSampler::ModelSampler(std::shared_ptr<Environment> env, std::shared_ptr<Context> context) {
  env_ = env;
  context_ = context;

  BatchSchedulerConfig batch_scheduler_config;
  env_->GetBatchSchedulerConfig(batch_scheduler_config);
  sampler_ = std::make_shared<Sampler>(batch_scheduler_config, context_->device->GetDeviceId(), context_);

  vocab_size_ = env_->GetVocabSize();
  bitmask_stride_ = (vocab_size_ + kBitsPerInt - 1) / kBitsPerInt;
}

void ModelSampler::BuildSamplingRequest(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs,
                                        std::vector<SamplingRequest>& sampling_reqs,
                                        const bool enable_main_layers_sampler) {
  PROFILE_EVENT_SCOPE(BuildSamplingRequest_, "BuildSamplingRequest");
  for (std::shared_ptr<ExecutorInferRequest> req_ptr : reqs) {
    SamplingRequest sampling_req;
    sampling_req.req_id = req_ptr->req_id;
    sampling_req.logits_custom_length = req_ptr->logits_custom_length;
    sampling_req.input_tokens = std::shared_ptr<std::vector<int>>(req_ptr, &req_ptr->input_tokens);
    sampling_req.forwarding_tokens = &(req_ptr->forwarding_tokens);
    sampling_req.sampling_token_num = req_ptr->sampling_token_num;
    sampling_req.last_step_token_num = req_ptr->last_step_token_num;
    sampling_req.sampling_result_tokens = &(req_ptr->sampling_result_tokens);
    sampling_req.sampling_result_tokens->clear();
    sampling_req.response = &(req_ptr->response);
    sampling_req.request_target = std::make_shared<const decltype(req_ptr->request_target)>(req_ptr->request_target);
    sampling_req.logprobs = std::shared_ptr<decltype(req_ptr->logprobs)>(req_ptr, &req_ptr->logprobs);
    sampling_req.logits_offset = req_ptr->logits_offset;
    sampling_req.logits_buf = req_ptr->model_instance->GetLogitsPtr();
    sampling_req.sampling_config = &(req_ptr->sampling_config);
    if (sampling_req.sampling_config->num_beams > 1) {
      sampling_req.sampling_config->logprobs_num =
          std::max(sampling_req.sampling_config->logprobs_num, sampling_req.sampling_config->num_beams);
      sampling_req.sampling_config->topk =
          std::max(sampling_req.sampling_config->topk, sampling_req.sampling_config->num_beams);
    }
    sampling_req.ngram_dict = &(req_ptr->ngram_dict);
    sampling_req.grammar_matcher = req_ptr->grammar_matcher;
    sampling_req.apply_grammar_constraint = enable_main_layers_sampler;
    sampling_req.enable_mtp_sampler = !enable_main_layers_sampler;
    sampling_reqs.push_back(sampling_req);
  }
}

Status ModelSampler::GenerateAndApplyBitmask(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs,
                                             std::vector<SamplingRequest>& sampling_reqs) {
  if (sampling_reqs.empty()) {
    return Status();
  }

  // Collect requests that have structured generators
  std::vector<size_t> logits_offsets;
  std::vector<int32_t> combined_bitmask;

  for (size_t i = 0; i < reqs.size(); ++i) {
    auto& req = reqs[i];
    if (req->structured_generator) {
      // Skip if generator is already terminated
      if (req->structured_generator->IsTerminated()) {
        KLLM_LOG_DEBUG << "Structured generator already terminated for req_id " << req->req_id
                       << ", skipping bitmask generation";
        continue;
      }
      // Generate bitmask for this request using the stride expected by ApplyTokenBitmaskSelective
      std::vector<int32_t> bitmask(bitmask_stride_, 0);
      bool need_constraint = req->structured_generator->FillNextTokenBitmask(bitmask.data());
      if (need_constraint) {
        logits_offsets.push_back(sampling_reqs[i].logits_offset);
        combined_bitmask.insert(combined_bitmask.end(), bitmask.begin(), bitmask.end());
      }
    }
  }

  // Apply bitmask if we have any
  if (!logits_offsets.empty() && !combined_bitmask.empty()) {
    float* device_logits = sampling_reqs[0].logits_buf;
    // Allocate device memory and copy bitmask
    size_t bitmask_size = combined_bitmask.size() * sizeof(int32_t);
    void* device_bitmask = nullptr;
    Malloc(&device_bitmask, bitmask_size);
    MemcpyAsync(device_bitmask, combined_bitmask.data(), bitmask_size, MEMCPY_HOST_TO_DEVICE,
                context_->device->GetH2DStream());
    StreamSynchronize(context_->device->GetH2DStream());

    // Apply bitmask to logits
    sampler_->ApplyTokenBitmaskSelective(device_logits, device_bitmask, vocab_size_, logits_offsets,
                                         context_->device->GetPrimaryComputeStream());
    // Synchronize and free device memory
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    Free(device_bitmask);
  }
  return Status();
}

Status ModelSampler::Sampling(std::shared_ptr<NewModelOutput> model_output,
                              std::shared_ptr<SamplingOutput> sampling_output) {
  sampling_output->schedule_id = model_output->schedule_output->schedule_id;
  sampling_output->schedule_output = model_output->schedule_output;

  BuildSamplingRequest(model_output->schedule_output->executor_running_reqs, sampling_output->sampling_reqs,
                       model_output->run_mode == RunMode::kMain);

  // Generate and apply bitmask from Executor-side generators
  GenerateAndApplyBitmask(model_output->schedule_output->executor_running_reqs, sampling_output->sampling_reqs);

  sampler_->Sampling(sampling_output->sampling_reqs, context_->device->GetPrimaryComputeStream());
  return Status();
}

}  // namespace ksana_llm
