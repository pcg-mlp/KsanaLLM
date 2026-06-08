/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/eccl_all_gather_layer.h"

namespace ksana_llm {

template <typename T>
Status EcclAllGatherLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                   std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "EcclAllGatherLayer not supported.");
}

template <typename T>
Status EcclAllGatherLayer<T>::Forward(const Tensor* local, Tensor* output, const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "EcclAllGatherLayer not supported.");
}

template class EcclAllGatherLayer<float>;
template class EcclAllGatherLayer<float16>;
template class EcclAllGatherLayer<bfloat16>;

}  // namespace ksana_llm
