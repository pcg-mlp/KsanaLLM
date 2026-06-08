/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

class NcclAllGatherLayer : public BaseLayer {
 public:
  // Gather along dim0: permute_buf==nullptr.
  // Gather along dim1: permute_buf is used as intermediate storage.
  Status Forward(const Tensor* local, Tensor* output, const Tensor* permute_buf = nullptr,
                 const LayerExecContext& exec_ctx = {});

 private:
  template <typename T>
  Status ForwardT(const Tensor& local, Tensor& output, const Tensor* permute_buf, const LayerExecContext& exec_ctx);
};

}  // namespace ksana_llm
