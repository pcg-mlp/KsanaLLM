/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/llm_runtime.h"
#include <memory>
#include <unordered_map>
#include <vector>

#include "numerous_llm/runtime/forward_request.h"
#include "numerous_llm/runtime/infer_stage.h"
#include "numerous_llm/runtime/model_instance.h"
#include "numerous_llm/runtime/sampling_request.h"
#include "numerous_llm/samplers/sampler.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

template <typename Key, typename Value, typename T>
inline Value& GetMapValue(std::unordered_map<Key, Value>& m, const Key& key, T&& default_value) {
  return m.emplace(key, std::forward<T>(default_value)).first->second;
}

LlmRuntime::LlmRuntime(std::shared_ptr<Context> context) : context_(context) {
  worker_group_ = std::make_shared<WorkerGroup>(context_->GetTensorParallelSize(), context_->GetTensorParallelSize());

  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    samplers_.push_back(std::make_shared<Sampler>(worker_id));
  }
}

void LlmRuntime::BuildForwardRequests(
    std::vector<std::shared_ptr<InferRequest>>& reqs,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs) {
  for (std::shared_ptr<InferRequest> req_ptr : reqs) {
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
    forward_req.block_size = 4096;
    forward_req.kv_cache_ptrs = req_ptr->GetBlockPtrs();
    forward_req.logits_buf = req_ptr->GetLogitsPtr();
    forward_req.logits_offset = req_ptr->logits_offset;
    forward_req.output_tokens = &(req_ptr->output_tokens);
    grouped_reqs[key][stage].push_back(forward_req);
  }
}

Status LlmRuntime::Forward(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>> grouped_reqs;
  BuildForwardRequests(reqs, grouped_reqs);

  std::vector<std::vector<std::future<Status>>> results;
  for (auto& [model_inst, stage_vec_reqs] : grouped_reqs) {
    for (auto& [stage, vec_req] : stage_vec_reqs) {
      results.push_back(model_inst->ForwardAsync(worker_group_, stage, vec_req));
    }
  }

  // Wait all instances donw and check status.
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
    sampling_req.output_tokens = &(req_ptr->output_tokens);
    sampling_req.logits_offset = req_ptr->logits_offset;
    sampling_req.logits_buf = req_ptr->GetLogitsPtr();
    sampling_req.sampling_config = &(req_ptr->sampling_config);
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

  // Wait all instances donw and check status.
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
  NLLM_LOG_INFO << "llm runtime step invoked.";
  Forward(reqs);
  return Sampling(reqs);
}

}  // namespace numerous_llm
