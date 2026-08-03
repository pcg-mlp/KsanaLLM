/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/situ_mul_layer.h"

#include "configure/environment.h"
#include "csrc/kernels/nvidia/moe_utils/moe_activation.h"

namespace ksana_llm {

Status SituMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  params_ = std::any_cast<const SituMulLayerParameters>(parameters.front());
  KLLM_LOG_INFO << fmt::format("Rank[{}] SituMulLayer Init with {}", rank_, params_.ToString());
  return Status();
}

template <typename T>
Status SituMulLayer::ForwardT(const Tensor& fused_gate_up, Tensor& output, const LayerExecContext& exec_ctx) {
  const size_t m = fused_gate_up.shape[0];
  const size_t n = fused_gate_up.shape[1];
  const size_t inter_size = n / 2;
  const size_t elements_num = m * n;

  llm_kernels::nvidia::InvokeSituAndMul<T, false>(
      reinterpret_cast<const T*>(fused_gate_up.GetPtr<void>()), reinterpret_cast<T*>(output.GetPtr<void>()), nullptr,
      elements_num, inter_size, params_.situ_beta, params_.situ_linear_beta, GetComputeStream(exec_ctx));

  output.shape = fused_gate_up.shape;
  output.shape[1] = inter_size;
  output.dtype = fused_gate_up.dtype;
  return Status();
}

Status SituMulLayer::Forward(const Tensor* fused_gate_up, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *fused_gate_up, *output, exec_ctx);
}

}  // namespace ksana_llm
