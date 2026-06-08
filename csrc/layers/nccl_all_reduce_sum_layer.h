/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

class NcclAllReduceSumLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;
  Status Forward(const Tensor* input, Tensor* output, const LayerExecContext& exec_ctx = {});

 private:
  bool need_sync_;
};

}  // namespace ksana_llm
