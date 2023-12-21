/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/nccl_all_reduce_sum_layer.h"

namespace numerous_llm {
// kernel host代码代补充
void nccl_all_reduce_sum(const Tensor& input, Tensor output, cudaStream_t stream) {}

Status NcclAllReduceSumLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  nccl_all_reduce_sum(input_tensors[0], output_tensors[0], stream_);
  return Status();
}
}  // namespace numerous_llm
