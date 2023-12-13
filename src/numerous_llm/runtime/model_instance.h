/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

#include "numerous_llm/models/llama/llama.h"
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/runtime/infer_stage.h"
#include "numerous_llm/runtime/worker.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class ModelInstance {
 public:
  ModelInstance(const std::shared_ptr<Context>& context);

  // Load model with specified model config.
  void Load(const ModelConfig& model_config);

  // The instance name.
  std::string name;

  // forward
  void Forward(const InferStage stage, const std::vector<TensorMap*>& input_tensor_maps,
               std::vector<TensorMap*>& output_tensor_maps);

 private:
  std::shared_ptr<Context> context_{nullptr};

  std::vector<std::unique_ptr<Worker>> workers_;

  int tensor_parallel_size_{0};
  int pipeline_parallel_size_{0};
};

}  // namespace numerous_llm
