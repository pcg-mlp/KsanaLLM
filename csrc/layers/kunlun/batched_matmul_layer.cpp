/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/batched_matmul_layer.h"
#include "device/kunlun/kunlun_device_context_impl.h"

#include "kernels/kunlun/batched_matmul_wrapper.h"

namespace ksana_llm {

Status BatchedMatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;
  return Status();
}

size_t BatchedMatMulLayer::GetWorkspaceSize() { return 0; }

Status BatchedMatMulLayer::Forward(const Tensor* a, const Tensor* b, Tensor* output,
                                   const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *a, *b, *output, exec_ctx);
}

template <typename T>
Status BatchedMatMulLayer::ForwardT(const Tensor& a, const Tensor& b, Tensor& output,
                                    const LayerExecContext& exec_ctx) {
  // Input convention for Kunlun backend:
  //   a: A [M, B, K]
  //   b: B [B, N, K]  (weight stored in [B, N, K] for xblas::fc_batch trans_b=true)
  // Output:
  //   output: C [M, B, N]
  KLLM_CHECK_WITH_INFO(a.shape.size() == 3, "a shape size should be 3.");
  KLLM_CHECK_WITH_INFO(b.shape.size() == 3, "b shape size should be 3.");
  KLLM_CHECK_WITH_INFO(a.shape[1] == b.shape[0], "input batch size should be equal.");
  KLLM_CHECK_WITH_INFO(a.shape[2] == b.shape[2], "input k dim should be equal.");

  int64_t m = static_cast<int64_t>(a.shape[0]);
  int64_t bs = static_cast<int64_t>(a.shape[1]);
  int64_t k = static_cast<int64_t>(a.shape[2]);
  int64_t n = static_cast<int64_t>(b.shape[1]);  // weight dim1 is N

  if (m == 0) {
    output.shape = {0, static_cast<size_t>(bs), static_cast<size_t>(n)};
    output.dtype = a.dtype;
    return Status();
  }

  xdnn::Context* xctx = static_cast<xdnn::Context*>(
      context_->device->ext->GetXdnnContext(context_->device->GetPrimaryComputeStream().Get()));

  int ret = 0;
  if (b.dtype == TYPE_INT8) {
    // INT8 weight path: activation (bf16/fp16) dynamically quantized, then INT8 BMM
    KLLM_CHECK_WITH_INFO(b.weight_scales != nullptr,
                         "BatchedMatMulLayer INT8 path: weight tensor must have weight_scales bound.");
    const T* a_ptr = a.GetPtr<T>();
    const int8_t* b_ptr = b.GetPtr<int8_t>();
    const float* b_scales = b.weight_scales->GetPtr<float>();
    T* c = output.GetPtr<T>();
    ret = XpuBatchedMatMulInt8<T>(xctx, a_ptr, b_ptr, b_scales, c, m, n, k, bs);
  } else {
    // Same-type path: bf16×bf16 or fp16×fp16
    const T* a_ptr = a.GetPtr<T>();
    const T* b_ptr = b.GetPtr<T>();
    T* c = output.GetPtr<T>();
    ret = XpuBatchedMatMul<T>(xctx, a_ptr, b_ptr, c, m, n, k, bs);
  }

  if (ret != 0) {
    KLLM_LOG_ERROR << "BatchedMatMulLayer failed with ret=" << ret << " (m=" << m << ", bs=" << bs << ", n=" << n
                   << ", k=" << k << ")";
    return Status(RET_RUNTIME_FAILED, "XpuBatchedMatMul failed in BatchedMatMulLayer.");
  }

  output.shape = {static_cast<size_t>(m), static_cast<size_t>(bs), static_cast<size_t>(n)};
  output.dtype = a.dtype;
  return Status();
}

}  // namespace ksana_llm
