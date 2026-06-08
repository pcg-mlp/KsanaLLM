/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/split.h"

#include "layers/split_layer.h"

namespace ksana_llm {

Split::Split(const LayerCreationContext& creation_context) {
  split_layer_ = std::make_shared<SplitLayer>();
  split_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

Split::~Split() {}

Status Split::Forward(const Tensor& input, std::vector<Tensor>& output_tensors) {
  auto* typed = static_cast<SplitLayer*>(split_layer_.get());
  std::vector<size_t> split_sizes(output_tensors.size());
  std::vector<Tensor*> out_ptrs(output_tensors.size());
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    split_sizes[i] = output_tensors[i].shape[1];
    out_ptrs[i] = &output_tensors[i];
  }
  STATUS_CHECK_RETURN(typed->Forward(&input, split_sizes, out_ptrs));
  return Status();
}

}  // namespace ksana_llm
