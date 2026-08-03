/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "device/common_device.h"
#include "layers/base_layer.h"

namespace ksana_llm {

struct SituMulLayerParameters : public BaseLayerParameters {
  float situ_beta{0.0f};
  float situ_linear_beta{0.0f};

  std::string ToString() const { return fmt::format("situ_beta:{}, situ_linear_beta:{}", situ_beta, situ_linear_beta); }
};

class SituMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  Status Forward(const Tensor* fused_gate_up, Tensor* output, const LayerExecContext& exec_ctx = {});

 private:
  template <typename T>
  Status ForwardT(const Tensor& fused_gate_up, Tensor& output, const LayerExecContext& exec_ctx);

  struct SituMulLayerParameters params_;
};

}  // namespace ksana_llm
