/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

template <typename T>
class EcclAllGatherLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  Status Forward(const Tensor* local, Tensor* output, const LayerExecContext& exec_ctx = {});

 protected:
};

}  // namespace ksana_llm
