/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/group_matmul_layer.h"

namespace ksana_llm {

template <typename T, DataType WT>
Status GroupMatMulLayer<T, WT>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context,
                                     int rank) {
  return Status(RET_RUNTIME, "GroupMatMulLayer is not supported in ascend\n");
}

template <typename T, DataType WT>
size_t GroupMatMulLayer<T, WT>::GetWorkSpaceSize() {
  return 0;
}

template <typename T, DataType WT>
Status GroupMatMulLayer<T, WT>::Preprocess(const ModelConfig& model_config_) {
  return Status(RET_RUNTIME, "GroupMatMulLayer is not supported in ascend\n");
}

template <typename T, DataType WT>
Status GroupMatMulLayer<T, WT>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_RUNTIME, "GroupMatMulLayer is not supported in ascend\n");
}

template class GroupMatMulLayer<float, TYPE_I4_GROUP>;
template class GroupMatMulLayer<float16, TYPE_I4_GROUP>;
}  // namespace ksana_llm
