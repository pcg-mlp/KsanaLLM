/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/concat_layer.h"

#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status ConcatLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                         std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  concat_dim = std::any_cast<const size_t>(parameters[0]);
  return Status();
}

template <typename T>
Status ConcatLayer::ForwardT(const Tensor* a, const Tensor* b, Tensor* output) {
  std::vector<size_t> a_shape = std::vector<size_t>(a->shape);
  std::vector<size_t> b_shape = std::vector<size_t>(b->shape);
  KLLM_CHECK_WITH_INFO(a_shape.size() == b_shape.size(), "shape size must be identical in cacat");
  for (size_t i = 0; i < a_shape.size(); ++i) {
    if (i == concat_dim) continue;
    KLLM_CHECK_WITH_INFO(
        a_shape[i] == b_shape[i],
        fmt::format("The shapes of all dimensions except the concatenation dimension must be identical. {} vs {}",
                    a->ToString(), b->ToString()));
  }
  output->dtype = a->dtype;
  output->shape = a->shape;
  output->shape[concat_dim] = a_shape[concat_dim] + b_shape[concat_dim];
  std::vector<size_t> output_shape = std::vector<size_t>(output->shape);
  const size_t inner_dim_size =
      std::accumulate(output_shape.begin() + concat_dim + 1, output_shape.end(), 1, std::multiplies<>());
  const size_t outer_dim_size =
      std::accumulate(output_shape.begin(), output_shape.begin() + concat_dim, 1, std::multiplies<>());
  Concat<T>(reinterpret_cast<const void*>(a->GetPtr<void>()), reinterpret_cast<const void*>(b->GetPtr<void>()),
            a_shape[concat_dim], b_shape[concat_dim], outer_dim_size, inner_dim_size,
            reinterpret_cast<void*>(output->GetPtr<void>()), context_->device->GetPrimaryComputeStream().Get());
  return Status();
}

Status ConcatLayer::Forward(const Tensor* a, const Tensor* b, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, a, b, output);
}

}  // namespace ksana_llm
