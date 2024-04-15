/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/model_instance.h"
#include <future>
#include <memory>
#include <vector>

#include "ksana_llm/runtime/worker.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"

#include "ksana_llm/models/baichuan/baichuan_weight.h"
#include "ksana_llm/models/llama/llama_weight.h"
#include "ksana_llm/models/qwen/qwen_weight.h"

#include "ksana_llm/models/baichuan/baichuan_model.h"
#include "ksana_llm/models/llama/llama_model.h"
#include "ksana_llm/models/qwen/qwen_model.h"

namespace ksana_llm {

std::vector<std::shared_ptr<BaseModel>> ModelInstance::models_;
std::vector<std::shared_ptr<BaseWeight>> ModelInstance::weights_;

void ModelInstance::Load() {
  std::string unified_model_type = model_config_.type;
  // unify it to lower case
  std::transform(unified_model_type.begin(), unified_model_type.end(), unified_model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (unified_model_type.find("llama") != std::string::npos) {
    type = "llama";
    CreateModelInstance<LlamaModel, LlamaWeight>(unified_model_type);
  } else if (unified_model_type.find("qwen") != std::string::npos) {
    type = "qwen";
    CreateModelInstance<QwenModel, QwenWeight>(unified_model_type);
  } else if (unified_model_type.find("baichuan") != std::string::npos) {
    type = "baichuan";
    CreateModelInstance<BaichuanModel, BaichuanWeight>(unified_model_type);
  } else {
    throw std::runtime_error(fmt::format("Unknown model type {}.", unified_model_type));
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

}  // namespace ksana_llm
