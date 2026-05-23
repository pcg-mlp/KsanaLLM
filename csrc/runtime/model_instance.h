/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <future>
#include <memory>
#include <string>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/context.h"
#include "configure/environment.h"
#include "kernels/cast.h"
#include "models/base/base_model.h"
#include "runtime/forward_request.h"
#include "runtime/hidden_unit_buffer.h"
#include "runtime/infer_stage.h"
#include "runtime/threadpool.h"
#include "runtime/weight_instance_inferface.h"
#include "utils/search_path.h"
#include "utils/singleton.h"
#include "utils/tensor.h"

namespace ksana_llm {

class ModelInstance {
 public:
  ModelInstance(const ModelConfig& model_config, const RuntimeConfig& runtime_config, std::shared_ptr<Context> context,
                std::shared_ptr<WeightInstanceInterface>& weight_instance)
      : model_config_(model_config),
        runtime_config_(runtime_config),
        context_(context),
        weight_instance_(weight_instance) {
    loader_models_threadpool_ = std::make_shared<ThreadPool>(context->global->GetTensorParallelSize());
    loader_models_threadpool_->Start();
  }

  ~ModelInstance() { loader_models_threadpool_->Stop(); }

  // Load model with specified model config.
  void Load();

  // Set the cache layout for specific model instance.
  void SetBlockCacheLayout(std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

  // The instance name.
  std::string name;
  std::string type;

  Status Forward(std::shared_ptr<BaseWeight> weight, InferStage stage, std::vector<ForwardRequest>& forward_reqs,
                 HiddenUnitDeviceBuffer* hidden_unit, RunMode run_mode);

  // Get  the data type.
  DataType GetWeightDataType() { return model_config_.weight_data_type; }

  // Get the base ptr of model's logits buf.
  float* GetLogitsPtr();

  const ModelConfig& GetModelConfig() { return model_config_; }

  size_t GetMaxTokenNum() { return runtime_config_.max_seq_len; }

  uint32_t GetLayerNum() {
    if (pipeline_config_.upper_layer_idx < 0 || pipeline_config_.lower_layer_idx < 0) {
      Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config_);
    }
    uint32_t layer_num = pipeline_config_.upper_layer_idx - pipeline_config_.lower_layer_idx + 1;
    if (pipeline_config_.lower_nextn_layer_idx >= static_cast<int>(model_config_.num_layer)) {
      layer_num += pipeline_config_.upper_nextn_layer_idx - pipeline_config_.lower_nextn_layer_idx + 1;
    }
    return layer_num;
  }

  void Reset() {
    KLLM_LOG_INFO << "ModelInstance::Reset clear models and weights.";
    weight_instance_ = nullptr;
  }

  // Get the base model instance
  std::shared_ptr<BaseModel> GetBaseModel() { return model_instance_; }

 private:
  // The model config.
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;

  PipelineConfig pipeline_config_;

  // The global context.
  std::shared_ptr<Context> context_ = nullptr;
  std::shared_ptr<WeightInstanceInterface> weight_instance_;

  // The base modes
  std::shared_ptr<BaseModel> model_instance_;
  std::shared_ptr<ThreadPool> loader_models_threadpool_ = nullptr;

  // The model-level cache layout (holds all block cache layouts keyed by type).
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;
};

}  // namespace ksana_llm
