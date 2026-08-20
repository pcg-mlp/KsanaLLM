/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

class BkclAllGatherLayer : public BaseLayer {
 public:
  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);
};

}  // namespace ksana_llm
