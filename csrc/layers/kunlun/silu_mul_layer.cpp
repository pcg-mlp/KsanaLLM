/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/silu_mul_layer.h"
#include "device/kunlun/kunlun_device_context_impl.h"

#include "kernels/kunlun/basic_kernel_wrapper.h"

namespace ksana_llm {

Status SiluMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

template <typename T>
Status SiluMulLayer::ForwardFusedT(const Tensor& fused_gate_up, Tensor& output, const LayerExecContext& exec_ctx) {
  void* stream = context_->device->GetPrimaryComputeStream().Get();
  xdnn::Context* xctx = static_cast<xdnn::Context*>(context_->device->ext->GetXdnnContext(stream));

  // Row-based split: fused_gate_up [m, 2*n] -> output [m, n]
  // Splits input into left/right halves, output = silu(left) * right
  int64_t m = static_cast<int64_t>(fused_gate_up.shape[0]);
  int64_t n = static_cast<int64_t>(fused_gate_up.shape[1]) / 2;

  const T* input = fused_gate_up.GetPtr<T>();
  T* out = output.GetPtr<T>();

  int ret = XpuRowBasedSiluMul<T>(xctx, input, out, m, n);
  if (ret != 0) {
    return Status(RET_RUNTIME_FAILED, "SiluMul kernel failed in SiluMulLayer.");
  }

  output.shape = fused_gate_up.shape;
  output.shape[1] = static_cast<size_t>(n);
  output.dtype = fused_gate_up.dtype;
  return Status();
}

template <typename T>
Status SiluMulLayer::ForwardSplitT(const Tensor& gate, const Tensor& up, Tensor& output,
                                   const LayerExecContext& exec_ctx) {
  namespace xpukernel = xpytorch::xpu::api;
  void* stream = context_->device->GetPrimaryComputeStream().Get();
  auto* xpuctx = static_cast<xpukernel::Context*>(context_->device->ext->GetXpuKernelContext(stream));

  // Two-input mode: output = silu(gate) * up
  int64_t m = static_cast<int64_t>(gate.shape[0]);
  int64_t n = static_cast<int64_t>(gate.shape[1]);

  const T* input = gate.GetPtr<T>();
  const T* up_ptr = up.GetPtr<T>();
  T* out = output.GetPtr<T>();

  int ret = XpuSiluMul<T>(xpuctx, input, up_ptr, out, m, n);
  if (ret != 0) {
    return Status(RET_RUNTIME_FAILED, "SiluMul kernel failed in SiluMulLayer.");
  }

  output.shape = gate.shape;
  output.dtype = gate.dtype;
  return Status();
}

Status SiluMulLayer::Forward(const Tensor* fused_gate_up, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardFusedT, *fused_gate_up, *output, exec_ctx);
}

Status SiluMulLayer::Forward(const Tensor* gate, const Tensor* up, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardSplitT, *gate, *up, *output, exec_ctx);
}

}  // namespace ksana_llm
