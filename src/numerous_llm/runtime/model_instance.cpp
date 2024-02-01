/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/model_instance.h"
#include <future>
#include <memory>
#include <vector>

#include "numerous_llm/runtime/worker.h"
#include "numerous_llm/utils/dtypes.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/nvidia/cuda_utils.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

std::vector<std::shared_ptr<BaseModel>> ModelInstance::models_;
std::vector<std::shared_ptr<BaseWeight>> ModelInstance::weights_;

void ModelInstance::Load() {
  std::string unified_model_name = model_config_.name;
  // unify it to lower case
  std::transform(unified_model_name.begin(), unified_model_name.end(), unified_model_name.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (unified_model_name.find("llama") != std::string::npos) {
    name = "llama";
    NLLM_LOG_DEBUG << "Start to init LLaMA model instance";

    // Load model and weights on every device.
    for (size_t worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
      models_.push_back(CreateModel<Llama>(worker_id));
      weights_.push_back(CreateModelWeight<LlamaWeight>(worker_id));
      // weights_.push_back(std::make_shared<LlamaWeight<float>>());
    }
  } else {
    throw std::runtime_error(
        "Unknown model type. Hint: if your model is llama, please let model name in config.ini contains 'llama' word "
        "(ignore upper case or lower case)");
  }
}

std::vector<float*> ModelInstance::GetLogitsPtr() {
  std::vector<float*> results;
  for (auto& model : models_) {
    results.push_back(model->GetLogitsPtr());
  }
  return results;
}

std::vector<Status> ModelInstance::Forward(std::shared_ptr<WorkerGroup> worker_group, InferStage stage,
                                           std::vector<ForwardRequest>& forward_reqs) {
  std::vector<Status> results;
  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    results.push_back(
        worker_group->GetWorker(worker_id)->Forward(models_[worker_id], weights_[worker_id], stage, forward_reqs));
  }
  return results;
}

std::vector<std::future<Status>> ModelInstance::ForwardAsync(std::shared_ptr<WorkerGroup> worker_group,
                                                             InferStage stage,
                                                             std::vector<ForwardRequest>& forward_reqs) {
  std::vector<std::future<Status>> results;
  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    results.push_back(
        worker_group->GetWorker(worker_id)->ForwardAsync(models_[worker_id], weights_[worker_id], stage, forward_reqs));
  }
  return results;
}

}  // namespace numerous_llm
