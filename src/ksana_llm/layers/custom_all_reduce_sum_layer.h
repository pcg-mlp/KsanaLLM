/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

template <typename T>
class CustomAllReduceSumLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  void** metas_;
  void* buffer_;
  size_t buffer_size_;
  void* rank_data_;
  size_t rank_data_sz_;
  void** data_handles_;
  void** input_handles_;
  void* reduce_op_;
  bool is_init_ = false;
};

}  // namespace ksana_llm
