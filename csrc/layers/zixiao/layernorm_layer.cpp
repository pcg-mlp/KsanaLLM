/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/layernorm_layer.h"

#include <cstdint>

namespace ksana_llm {

Status LayernormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer not supported.");
}

template <typename T>
Status LayernormLayer::ForwardNoWeightT(const Tensor& input, Tensor& output) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer not supported.");
}

template <typename T>
Status LayernormLayer::ForwardWeightT(const Tensor& input, const Tensor& weight, Tensor& output, const Tensor* bias) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer not supported.");
}

template <typename T>
Status LayernormLayer::ForwardParallelT(const Tensor& in1, const Tensor& w1, const Tensor& in2, const Tensor& w2,
                                        Tensor& out1, Tensor& out2, int stride1, int stride2) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer not supported.");
}

Status LayernormLayer::Forward(const Tensor* input, Tensor* output, const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer not supported.");
}

Status LayernormLayer::Forward(const Tensor* input, const Tensor* weight, Tensor* output, const Tensor* bias,
                               const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer not supported.");
}

Status LayernormLayer::Forward(const Tensor* input1, const Tensor* weight1, const Tensor* input2, const Tensor* weight2,
                               Tensor* output1, Tensor* output2, int stride1, int stride2,
                               const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer not supported.");
}

}  // namespace ksana_llm
