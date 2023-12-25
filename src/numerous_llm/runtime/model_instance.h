/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <future>
#include <memory>
#include <string>

#include "numerous_llm/models/llama/llama.h"
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/runtime/forward_request.h"
#include "numerous_llm/runtime/infer_stage.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class Worker;
class WorkerGroup;

class ModelInstance {
 public:
  ModelInstance(const ModelConfig& model_config, std::shared_ptr<Context> context)
      : model_config_(model_config), context_(context) {}

  // Load model with specified model config.
  void Load();

  // The instance name.
  std::string name;

  std::vector<Status> Forward(std::shared_ptr<WorkerGroup> worker_group, InferStage stage,
                              std::vector<ForwardRequest>& forward_reqs);

  std::vector<std::future<Status>> ForwardAsync(std::shared_ptr<WorkerGroup> worker_group, InferStage stage,
                                                std::vector<ForwardRequest>& forward_reqs);

  // Get the kv cache size per token needed, its size is:
  //   (num_layer / pipeline_para) * (head_num / tensor_para) * size_per_head;
  int GetTokenCacheSize() {
    return (model_config_.num_layer / context_->GetPipeLineParallelSize()) *
           (model_config_.head_num / context_->GetTensorParallelSize()) * model_config_.size_per_head;
  }

  // Get  the data type.
  DataType GetDataType() { return model_config_.weight_data_type; }

  // Get the base ptr of model's logits buf.
  std::vector<float*> GetLogitsPtr();

  const ModelConfig& GetModelConfig() { return model_config_; }

 private:
  // Create the object and return a shared pointer.
  template <template <class> class ClassT, class BaseT>
  std::shared_ptr<BaseT> CreatetModelObject(int rank) {
    std::shared_ptr<BaseT> model_obj = nullptr;
    switch (model_config_.weight_data_type) {
      case DataType::TYPE_FP16:
        model_obj = std::make_shared<ClassT<half>>(model_config_, rank, context_);
        break;
      case DataType::TYPE_FP32:
        model_obj = std::make_shared<ClassT<float>>(model_config_, rank, context_);
        break;
      default:
        throw std::runtime_error("Unsupported Tensor type.");
    };
    return model_obj;
  }

  template <template <class> class ClassT>
  std::shared_ptr<BaseModel> CreateModel(int rank) {
    return CreatetModelObject<ClassT, BaseModel>(rank);
  }

  template <template <class> class ClassT>
  std::shared_ptr<BaseWeight> CreateModelWeight(int rank) {
    return CreatetModelObject<ClassT, BaseWeight>(rank);
  }

 private:
  // The model config.
  ModelConfig model_config_;

  // The global context.
  std::shared_ptr<Context> context_ = nullptr;

  // The base model and weight, shared by all model instances.
  static std::vector<std::shared_ptr<BaseModel>> models_;
  static std::vector<std::shared_ptr<BaseWeight>> weights_;
};

}  // namespace numerous_llm
