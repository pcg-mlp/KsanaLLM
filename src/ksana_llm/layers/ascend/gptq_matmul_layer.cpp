/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/gptq_matmul_layer.h"

namespace ksana_llm {

template <typename T, DataType WT>
Status GPTQMatMulLayer<T, WT>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context,
                                    int rank) {
  return Status(RET_RUNTIME, fmt::format("GPTQMatMulLayer is not supported in ascend\n"));
}

template <typename T, DataType WT>
int GPTQMatMulLayer<T, WT>::GetWorkSpaceSize() {
  return 0;
}

template <typename T, DataType WT>
Status GPTQMatMulLayer<T, WT>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_RUNTIME, fmt::format("GPTQMatMulLayer is not supported in ascend\n"));
}

template class GPTQMatMulLayer<float, TYPE_I4_G128>;
template class GPTQMatMulLayer<float16, TYPE_I4_G128>;
}  // namespace ksana_llm
