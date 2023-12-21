/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/layernorm_layer.h"

namespace numerous_llm {
// kernel host代码代补充
void layernorm(const Tensor& input, const Tensor& weight, Tensor output, cudaStream_t stream) {}

Status LayernormLayer::Init(const std::vector<std::any>& parameters, cudaStream_t stream) {
  stream_ = stream;
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("rms_norm_eps {}", rms_norm_eps_);
  return Status();
}

Status LayernormLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  layernorm(input_tensors[0], input_tensors[1], output_tensors[0], stream_);
  return Status();
}
}  // namespace numerous_llm
