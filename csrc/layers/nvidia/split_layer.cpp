/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/split_layer.h"

#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status SplitLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                        std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

template <typename T>
Status SplitLayer::ForwardT(const Tensor* input, const std::vector<size_t>& split_sizes, std::vector<Tensor*> outputs) {
  std::vector<T*> output_ptrs(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    output_ptrs[i] = outputs[i]->GetPtr<T>();
  }
  std::vector<int> col_offsets(outputs.size() + 1, 0);
  for (size_t i = 0; i < outputs.size(); ++i) {
    col_offsets[i + 1] = col_offsets[i] + static_cast<int>(split_sizes[i]);
    outputs[i]->shape[1] = split_sizes[i];
  }
  InvokeSplit<T>(input->GetPtr<T>(), output_ptrs, col_offsets, input->shape[0], input->shape[1], outputs.size(),
                 context_->device->GetPrimaryComputeStream().Get());
  return Status();
}

Status SplitLayer::Forward(const Tensor* input, const std::vector<size_t>& split_sizes, std::vector<Tensor*> outputs,
                           const LayerExecContext& exec_ctx) {
  KLLM_CHECK_WITH_INFO(split_sizes.size() == outputs.size(), "split_sizes and outputs must have same size");
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input, split_sizes, outputs);
}

}  // namespace ksana_llm
