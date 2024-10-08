/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <any>

#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
    context_ = context;
    rank_ = rank;
    return Status();
  }

  virtual size_t GetWorkSpaceSize() { return 0; }

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) = 0;

  virtual Status SetWorkSpaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) {
    workspace_buffer_ = workspace_buffer;
    return Status();
  }

  virtual Status Preprocess(const ModelConfig& model_config_) { return Status(); }

  virtual void Clear() {}

 protected:
  int rank_;
  std::shared_ptr<Context> context_;
  std::shared_ptr<Tensor> workspace_buffer_;
};

}  // namespace ksana_llm
