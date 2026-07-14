/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "configure/context.h"
#include "model_loader/file_loader/model_file_loader.h"
#include "models/base/base_model_config.h"
#include "models/base/model_weight.h"
#include "runtime/threadpool.h"
#include "utils/status.h"

namespace ksana_llm {

// Used to load model weights.
class ModelWeightLoader {
 public:
  ModelWeightLoader(std::shared_ptr<Environment> env, std::shared_ptr<Context> context);
  ~ModelWeightLoader();

  // Load weights from file, the index of weights is device rank.
  Status LoadWeight(std::shared_ptr<BaseModelConfig>& model_config, std::shared_ptr<ModelWeight>& dev_weight);

 private:
  std::shared_ptr<Environment> env_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;

  // Used to load weight, one thread for every device.
  std::shared_ptr<ThreadPool> weight_loader_threadpool_ = nullptr;
};

}  // namespace ksana_llm
