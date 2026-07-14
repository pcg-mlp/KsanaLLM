/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <future>
#include <memory>
#include <string>

#include "configure/context.h"
#include "configure/environment.h"
#include "model_loader/config_parser/model_config_parser.h"
#include "model_loader/weight_loader/model_weight_loader.h"
#include "models/base/base_model.h"
#include "runtime/forward_request.h"
#include "runtime/infer_stage.h"
#include "runtime/threadpool.h"
#include "runtime/weight_instance_inferface.h"
#include "utils/search_path.h"
#include "utils/singleton.h"
#include "utils/tensor.h"

namespace ksana_llm {

class WeightInstance : public WeightInstanceInterface {
 public:
  WeightInstance(const ModelConfig& model_config, const RuntimeConfig& runtime_config, std::shared_ptr<Context> context)
      : model_config_(model_config), runtime_config_(runtime_config), context_(context) {
    loader_weight_threadpool_ = std::make_shared<ThreadPool>(context->global->GetTensorParallelSize());
    loader_weight_threadpool_->Start();
  }
  ~WeightInstance() { loader_weight_threadpool_->Stop(); }

  // Load model with specified model config.
  virtual void Load() override;

  virtual std::shared_ptr<BaseWeight> GetWeight() override { return weight_; }

 private:
  void SetEmbeddingsConfig();

  void CheckTieEmbeddings(int weight_file_size);

  void CheckTieEmbeddings(const std::vector<std::string>& custom_name_list);

  void CreateWeightInstances();

  void LoadWeightsAndModelsMap(bool& loaded_from_cache);

  void LoadWeights();

  void ProcessWeights();

  bool TryToLoadWeightsFromCache();

  bool SaveWeightsToCache();

  // tmp function to check whether current model and configs satisfy new model weight loader.
  // TODO(huicongyao): remove this function later.
  bool IsCompatibleWithNewLoader(std::shared_ptr<BaseModelConfig> model_config);

 private:
  // The model config.
  ModelConfig model_config_;

  RuntimeConfig runtime_config_;

  // The global context.
  std::shared_ptr<Context> context_ = nullptr;

  // The base model and weight, shared by all model instances.
  std::shared_ptr<BaseWeight> weight_;

  std::shared_ptr<ThreadPool> loader_weight_threadpool_ = nullptr;
};

}  // namespace ksana_llm
