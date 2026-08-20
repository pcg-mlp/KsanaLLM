/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/matmul_layer.h"
#include "device/kunlun/kunlun_device_context_impl.h"

#include "kernels/kunlun/basic_kernel_wrapper.h"

namespace ksana_llm {

Status MatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                         std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;
  return Status();
}

size_t MatMulLayer::GetWorkspaceSize() { return 0; }

Status MatMulLayer::Forward(const Tensor* input, const Tensor* weight, Tensor* output,
                            const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *input, *weight, *output, exec_ctx);
}

template <typename T>
Status MatMulLayer::ForwardT(const Tensor& input, const Tensor& weight, Tensor& output,
                             const LayerExecContext& exec_ctx) {
  // Kunlun XPU: weight 保持原始 [n, k] 布局（跳过 Permute2D），通过 w_trans=true
  // 告诉 xBLAS 内层 k 维归约时 weight 访问 stride=1。这与 xSGLang I28Args 的
  // weight 转置优化对齐——xSGLang 在 process_weights_after_loading 中做
  // layer.weight = layer.weight.t() 后，xBLAS 以 trans_b=true 方式访问 weight。
  //
  // input: x [m, k]
  // weight: w [n, k] (HF 原始布局，未经 Permute2D)
  // output: y [m, n]
  xdnn::Context* xctx = static_cast<xdnn::Context*>(
      context_->device->ext->GetXdnnContext(context_->device->GetPrimaryComputeStream().Get()));

  int64_t m = static_cast<int64_t>(input.shape[0]);
  int64_t k = static_cast<int64_t>(input.shape[1]);
  int64_t n = static_cast<int64_t>(weight.shape[0]);  // w is [n, k], so n = dim 0

  const T* x = input.GetPtr<T>();
  const T* w = weight.GetPtr<T>();
  T* y = output.GetPtr<T>();

  // w_trans=true: xBLAS 将 weight 视为 [n,k] 的转置 → 逻辑 [k,n]，
  // 内层沿 k 维归约时 weight 的访存 stride=1（因为 weight 在内存中是 [n,k] 行主序，k 维是内层维）。
  // 这与 xSGLang 的 I28Args 优化一致。
  int ret = XpuFcFusion<T, T, T>(xctx, x, w, y, m, n, k,
                                 /*x_trans=*/false, /*w_trans=*/true,
                                 /*x_maxptr=*/nullptr, /*w_maxptr=*/nullptr, /*y_maxptr=*/nullptr);
  if (ret != 0) {
    return Status(RET_RUNTIME_FAILED, "XpuFcFusion failed in MatMulLayer.");
  }

  output.shape = {static_cast<size_t>(m), static_cast<size_t>(n)};
  output.dtype = input.dtype;
  return Status();
}

}  // namespace ksana_llm
