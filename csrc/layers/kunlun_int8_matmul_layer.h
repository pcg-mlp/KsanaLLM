/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/matmul_base_layer.h"

namespace ksana_llm {

// Kunlun INT8 MatMul layer for W4A8_KLX quantized models.
// Uses klx_int8_scaled_mm for W8 layers (Attention/Dense/SharedExpert)
// and klx_w4a8_per_chn_scaled_mm for W4 layers (MoE Experts via MoE layer).
// During inference: performs dynamic per-token INT8 activation quantization
// followed by INT8 GEMM with per-channel weight scale dequantization.
class KunlunInt8MatMulLayer : public MatMulBaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const Tensor* input, const Tensor* weight, Tensor* output,
                         const LayerExecContext& exec_ctx = {}) override;

  virtual size_t GetWorkspaceSize() override;

 private:
  template <typename T>
  Status ForwardT(const Tensor& input, const Tensor& weight, Tensor& output, const LayerExecContext& exec_ctx);
};

}  // namespace ksana_llm
