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

  virtual int GetWorkSpaceSize() { return 0; }

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) = 0;

  virtual void SetWorkSpaceBuffer(Tensor buffer) { workspace_buffer_ = buffer; }

 protected:
  int rank_;
  std::shared_ptr<Context> context_;
  Tensor workspace_buffer_;
};

}  // namespace ksana_llm
