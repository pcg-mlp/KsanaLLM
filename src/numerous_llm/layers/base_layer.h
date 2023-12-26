/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <any>

#include "numerous_llm/runtime/context.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
    context_ = context;
    rank_ = rank;
    return Status();
  };

  virtual int GetWorkSpaceSize() { return 0; }

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) = 0;

 protected:
  int rank_;
  std::shared_ptr<Context> context_;
};

}  // namespace numerous_llm
