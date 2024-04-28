/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/llm_runtime.h"
#include <memory>
#include <unordered_map>
#include <vector>

#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/runtime/model_instance.h"
#include "ksana_llm/runtime/sampling_request.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

template <typename Key, typename Value, typename T>
inline Value& GetMapValue(std::unordered_map<Key, Value>& m, const Key& key, T&& default_value) {
  return m.emplace(key, std::forward<T>(default_value)).first->second;
}

LlmRuntime::LlmRuntime(const BatchSchedulerConfig& batch_scheduler_config, std::shared_ptr<Context> context)
    : batch_schedule_config_(batch_scheduler_config), context_(context) {
  worker_group_ =
      std::make_shared<WorkerGroup>(context_->GetTensorParallelSize(), context_->GetTensorParallelSize(), context_);

  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    samplers_.push_back(std::make_shared<Sampler>(batch_schedule_config_, worker_id, context_));
  }
}

void LlmRuntime::BuildForwardRequests(
    std::vector<std::shared_ptr<InferRequest>>& reqs,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs) {
  for (size_t i = 0; i < reqs.size(); ++i) {
    std::shared_ptr<InferRequest>& req_ptr = reqs[i];

    req_ptr->step += 1;
    req_ptr->logits_offset = i;
    ModelInstance* key = req_ptr->model_instance.get();
    InferStage stage = req_ptr->infer_stage;
    if (grouped_reqs.find(key) == grouped_reqs.end()) {
      grouped_reqs[key] = {};
    }

    if (grouped_reqs[key].find(stage) == grouped_reqs[key].end()) {
      grouped_reqs[key][stage] = {};
    }

    ForwardRequest forward_req;
    forward_req.infer_stage = req_ptr->infer_stage;
    forward_req.step = req_ptr->step;
    forward_req.block_size = req_ptr->block_size;
    forward_req.kv_cache_ptrs = req_ptr->GetBlockPtrs();
    forward_req.logits_buf = req_ptr->GetLogitsPtr();
    forward_req.logits_offset = req_ptr->logits_offset;
    forward_req.output_tokens = &(req_ptr->output_tokens);
    forward_req.is_use_prefix_cache = req_ptr->is_use_prefix_cache;
    forward_req.prefix_cache_len = req_ptr->prefix_cache_len;
    forward_req.prefix_cache_blocks_number = req_ptr->prefix_cache_blocks_number;
    grouped_reqs[key][stage].push_back(forward_req);
  }
}

Status LlmRuntime::RunSerially(
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs) {
  Status result_status = Status();
  for (auto& [model_inst, stage_vec_reqs] : grouped_reqs) {
    for (auto& [stage, vec_req] : stage_vec_reqs) {
      std::vector<std::future<Status>> inst_results = model_inst->ForwardAsync(worker_group_, stage, vec_req);
      for (auto& worker_result : inst_results) {
        Status status = worker_result.get();
        if (!status.OK()) {
          result_status = status;
        }
      }
    }
  }
  return result_status;
}

Status LlmRuntime::Forward(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>> grouped_reqs;
  BuildForwardRequests(reqs, grouped_reqs);

  // context decode and decode run serially in single thread
  if (context_->IsRunContextDecodeAndDecodeSerially()) {
    // Wait all instances done and check status.
    return RunSerially(grouped_reqs);
  }

  std::vector<std::vector<std::future<Status>>> results;
  for (auto& [model_inst, stage_vec_reqs] : grouped_reqs) {
    for (auto& [stage, vec_req] : stage_vec_reqs) {
      results.push_back(model_inst->ForwardAsync(worker_group_, stage, vec_req));
    }
  }

  // Wait all instances done and check status.
  Status result_status = Status();
  for (auto& inst_results : results) {
    for (auto& worker_result : inst_results) {
      Status status = worker_result.get();
      if (!status.OK()) {
        result_status = status;
      }
    }
  }
  return result_status;
}

void LlmRuntime::BuildSamplingRequest(std::vector<std::shared_ptr<InferRequest>>& reqs,
                                      std::vector<SamplingRequest>& sampling_reqs) {
  for (std::shared_ptr<InferRequest> req_ptr : reqs) {
    SamplingRequest sampling_req;
    sampling_req.req_id = req_ptr->req_id;
    sampling_req.input_tokens = &(req_ptr->input_tokens);
    sampling_req.output_tokens = &(req_ptr->output_tokens);
    sampling_req.logprobs = &(req_ptr->logprobs);
    sampling_req.output_mutex = &(req_ptr->output_mutex);
    sampling_req.logits_offset = req_ptr->logits_offset;
    sampling_req.logits_buf = req_ptr->GetLogitsPtr();
    sampling_req.sampling_config = &(req_ptr->sampling_config);
    sampling_req.model_config = &(req_ptr->model_instance->GetModelConfig());
    sampling_reqs.push_back(sampling_req);
  }
}

Status LlmRuntime::Sampling(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  std::vector<SamplingRequest> sampling_reqs;
  BuildSamplingRequest(reqs, sampling_reqs);

  std::vector<std::future<Status>> results;
  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    results.push_back(worker_group_->GetWorker(worker_id)->SamplingAsync(samplers_[worker_id], sampling_reqs));
  }

  // Wait all instances done and check status.
  Status result_status = Status();
  for (auto& result : results) {
    Status status = result.get();
    if (!status.OK()) {
      result_status = status;
    }
  }
  return result_status;
}

Status LlmRuntime::Step(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  NLLM_LOG_DEBUG << "llm runtime step invoked.";
  Forward(reqs);
  return Sampling(reqs);
}

}  // namespace ksana_llm
