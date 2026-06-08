/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/torch.h>
#include "layers/base_layer.h"

namespace ksana_llm {

class InputRefitLayer : public BaseLayer {
 public:
  // emb_tensors: N embedding tensors; pos_tensor: position indices tensor.
  Status Forward(const std::vector<const Tensor*>& emb_tensors, const Tensor* pos_tensor, Tensor* output,
                 const LayerExecContext& exec_ctx = {});
  void Clear() { cast_cache_.clear(); }

 private:
  template <typename T>
  Status ForwardT(const std::vector<const Tensor*>& emb_tensors, const Tensor* pos_tensor, Tensor* output);

 public:
  std::vector<torch::Tensor> cast_cache_;
};

}  // namespace ksana_llm
