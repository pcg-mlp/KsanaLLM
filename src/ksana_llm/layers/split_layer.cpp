/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/split_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status SplitLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // TODO(karlluo): implement as same as feature
  // https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py#L152

  return Status();
}
}  // namespace ksana_llm