/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/matmul_layer.h"

namespace ksana_llm {

template <typename T>
Status MatMulLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "MatMulLayer not supported.");
}

template <typename T>
size_t MatMulLayer<T>::GetWorkspaceSize() {
  return 0;
}

template class MatMulLayer<float>;
template class MatMulLayer<float16>;
template class MatMulLayer<bfloat16>;

Status MatMulLayer::Forward(const Tensor* input, const Tensor* weight, Tensor* output,
                            const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "MatMulLayer not supported.");
}

}  // namespace ksana_llm
