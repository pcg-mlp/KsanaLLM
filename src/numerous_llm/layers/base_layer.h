/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <any>
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, cudaStream_t stream) {
    stream_ = stream;
    return Status();
  };

  virtual int GetWorkSpaceSize() { return 0; }

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) = 0;

 protected:
  cudaStream_t stream_;
};

}  // namespace numerous_llm
