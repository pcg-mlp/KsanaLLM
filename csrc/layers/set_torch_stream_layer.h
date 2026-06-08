/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#  include <torch/torch.h>
#  include "layers/base_layer.h"

namespace ksana_llm {

class SetTorchStreamLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;
  // Typed interface: bind torch stream (no tensor parameters needed).
  Status Forward(const LayerExecContext& exec_ctx = {});

  void Clear() override;

 private:
  cudaStream_t torch_stream_;
};

}  // namespace ksana_llm
#endif
