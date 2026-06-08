/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/eccl_all_reduce_sum_layer.h"

namespace ksana_llm {

template <typename T>
Status EcclAllReduceSumLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                      std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "EcclAllReduceSumLayer not supported.");
}

template <typename T>
Status EcclAllReduceSumLayer<T>::Forward(const Tensor* input, Tensor* output, const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "EcclAllReduceSumLayer not supported.");
}

template class EcclAllReduceSumLayer<float>;
template class EcclAllReduceSumLayer<float16>;
template class EcclAllReduceSumLayer<bfloat16>;

}  // namespace ksana_llm
