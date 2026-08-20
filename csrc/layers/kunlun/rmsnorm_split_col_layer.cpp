/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/kunlun_rmsnorm_split_col_layer.h"

#include "device/kunlun/kunlun_device_context_impl.h"
#include "kernels/kunlun/attention_kernel_wrapper.h"

namespace ksana_llm {

Status KunlunRmsnormSplitColLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                        std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int idx = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[idx++]);
  q_dim_ = std::any_cast<const int64_t>(parameters[idx++]);
  kv_dim_ = std::any_cast<const int64_t>(parameters[idx++]);
  rot_dim_ = std::any_cast<const int64_t>(parameters[idx++]);
  return Status();
}

Status KunlunRmsnormSplitColLayer::Forward(const std::vector<Tensor>& input_tensors,
                                           std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_2_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status KunlunRmsnormSplitColLayer::ForwardT(const std::vector<Tensor>& input_tensors,
                                            std::vector<Tensor>& output_tensors) {
  xdnn::Context* xctx = static_cast<xdnn::Context*>(
      context_->device->ext->GetXdnnContext(context_->device->GetPrimaryComputeStream().Get()));

  const T* input = input_tensors[0].GetPtr<T>();
  // Norm weights (q_a_layernorm / kv_a_layernorm) are no longer cast to fp32 on load
  // (see commit 5416bef); they keep their native activation dtype T (bf16/fp16). Read
  // them as T so the fused rmsnorm_split_col kernel gets the correct byte layout —
  // reading bf16 weights through a float* mixes adjacent elements and yields NaN.
  const T* q_weight = input_tensors[1].GetPtr<T>();
  const T* kv_weight = input_tensors[2].GetPtr<T>();
  T* output_q = output_tensors[0].GetPtr<T>();
  T* output_kv = output_tensors[1].GetPtr<T>();
  T* output_latent = output_tensors[2].GetPtr<T>();
  int64_t rows = static_cast<int64_t>(input_tensors[0].shape[0]);

  int ret = XpuRmsnormSplitCol<T, T>(xctx, input, q_weight, kv_weight, output_q, output_kv, output_latent, rows, q_dim_,
                                     kv_dim_, rot_dim_, rms_norm_eps_);
  if (ret != 0) {
    return Status(RET_RUNTIME_FAILED, "XpuRmsnormSplitCol failed");
  }
  return Status();
}

}  // namespace ksana_llm
