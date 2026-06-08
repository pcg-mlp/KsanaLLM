/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

class TrtAllReduceSumLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  Status Forward(const Tensor* input, Tensor* output, const LayerExecContext& exec_ctx = {});

  void Clear() override;

 private:
  template <typename T>
  Status ForwardT(const Tensor& input, Tensor& output);

 private:
  bool is_init_ = false;
  bool need_sync_{false};
};

}  // namespace ksana_llm
