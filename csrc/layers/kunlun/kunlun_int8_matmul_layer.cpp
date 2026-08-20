/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/kunlun_int8_matmul_layer.h"
#include "device/kunlun/kunlun_device_context_impl.h"

#include "kernels/kunlun/quant_gemm_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {

Status KunlunInt8MatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                   std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  KLLM_LOG_INFO << "KunlunInt8MatMulLayer initialized (W4A8_KLX quantization support).";
  return Status();
}

size_t KunlunInt8MatMulLayer::GetWorkspaceSize() { return 0; }

Status KunlunInt8MatMulLayer::Forward(const Tensor* input, const Tensor* weight, Tensor* output,
                                      const LayerExecContext& exec_ctx) {
  DISPATCH_BY_2_DTYPE(inter_data_type_, ForwardT, *input, *weight, *output, exec_ctx);
}

template <typename T>
Status KunlunInt8MatMulLayer::ForwardT(const Tensor& input, const Tensor& weight, Tensor& output,
                                       const LayerExecContext& exec_ctx) {
  // Input convention (from Linear::Forward):
  //   input:  x [m, k] (bf16/fp16) — activation
  //   weight: w [n, k] (int8) — weight (trans_b=true, 物理 shape [N, K])
  //           weight.weight_scales → Tensor* pointing to scale [n] (fp32, per-channel)
  // Output:
  //   output: y [m, n] (bf16/fp16)

  KLLM_CHECK_WITH_INFO(weight.weight_scales != nullptr,
                       "KunlunInt8MatMulLayer: weight tensor must have weight_scales bound");

  xdnn::Context* xctx = static_cast<xdnn::Context*>(
      context_->device->ext->GetXdnnContext(context_->device->GetPrimaryComputeStream().Get()));

  int64_t m = static_cast<int64_t>(input.shape[0]);
  int64_t k = static_cast<int64_t>(input.shape[1]);
  int64_t n = static_cast<int64_t>(weight.shape[0]);

  const T* x = input.GetPtr<T>();
  const int8_t* w = weight.GetPtr<int8_t>();
  // Scale accessed via weight_scales pointer (bound during weight post-processing)
  const float* scale_w = weight.weight_scales->GetPtr<float>();
  T* y = output.GetPtr<T>();

  if (m == 0) {
    output.shape = {0, static_cast<size_t>(n)};
    output.dtype = inter_data_type_;
    return Status();
  }

  int ret = XpuInt8ScaledMM<T>(xctx, x, w, scale_w, y, m, n, k, /*bias=*/nullptr);
  if (ret != 0) {
    KLLM_LOG_ERROR << "XpuInt8ScaledMM failed with ret=" << ret << " (m=" << m << ", n=" << n << ", k=" << k << ")";
    return Status(RET_RUNTIME_FAILED, "XpuInt8ScaledMM failed in KunlunInt8MatMulLayer.");
  }

  output.shape = {static_cast<size_t>(m), static_cast<size_t>(n)};
  output.dtype = inter_data_type_;
  return Status();
}

}  // namespace ksana_llm
