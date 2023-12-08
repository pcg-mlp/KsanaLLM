/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/model_instance.h"
#include "numerous_llm/models/llama/llama.h"
#include "numerous_llm/models/llama/llama_weight.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/nvidia/cuda_utils.h"

namespace numerous_llm {

ModelInstance::ModelInstance(const std::shared_ptr<Context>& context) : context_(context) {}

void ModelInstance::Load(const ModelConfig& model_config) {
  std::string unified_model_name = model_config.name;
  // unify it to lower case
  std::transform(unified_model_name.begin(), unified_model_name.end(), unified_model_name.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  tensor_parallel_size_ = context_->GetTensorParallelSize();
  pipeline_parallel_size_ = context_->GetPipeLineParallelSize();

  workers_.resize(tensor_parallel_size_ * pipeline_parallel_size_);
  if (unified_model_name.find("llama") != std::string::npos) {
    NLLM_LOG_INFO << "Start to init LLaMA model instance";
  } else {
    throw std::runtime_error(
        "Unknown model type. Hint: if your model is llama, please let model name in config.ini contains 'llama' word "
        "(ignore upper case or lower case)");
  }
}

void ModelInstance::Forward(const TensorMap& input_tensor_map, const SamplingConfig& sampling_config,
                            TensorMap& output_tensor_map) {

  for (int worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    // copy inputs to device
    
    // worker forward
    workers_[worker_id];
  }
}

}  // namespace numerous_llm
