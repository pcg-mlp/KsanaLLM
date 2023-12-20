/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "numerous_llm/models/base/base_model.h"
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/runtime/infer_stage.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

// The worker executed on every device.
class Worker {
 public:
  Worker(std::shared_ptr<BaseModel> base_model, std::shared_ptr<BaseWeight> base_weight) {
    base_model_ptr_.reset();
    base_model_ptr_ = base_model;
    base_weight_ptr_.reset();
    base_weight_ptr_ = base_weight;
  }
  ~Worker() {}
  // Execute model inference.
  Status Execute(Context& ctx, const InferStage stage, const int worker_id,
                 const std::vector<TensorMap*>& input_tensor_maps, std::vector<TensorMap*>& output_tensor_maps);

 private:
  std::shared_ptr<BaseModel> base_model_ptr_{nullptr};
  std::shared_ptr<BaseWeight> base_weight_ptr_{nullptr};
};

}  // namespace numerous_llm
