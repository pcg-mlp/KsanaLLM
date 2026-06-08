/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/add_mul_layer.h"

namespace ksana_llm {

Status AddMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                         std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "AddMul operation is not supported on Ascend backend");
}

template <typename T>
Status AddMulLayer::ForwardT(const Tensor& input1, const Tensor& input2, Tensor& output) {
  KLLM_THROW("AddMulLayer not implement in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "AddMulLayer not supported.");
}

Status AddMulLayer::Forward(const Tensor* input1, const Tensor* input2, Tensor* output,
                            const LayerExecContext& exec_ctx) {
  KLLM_THROW("AddMulLayer not implement in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "AddMulLayer not supported.");
}

}  // namespace ksana_llm
