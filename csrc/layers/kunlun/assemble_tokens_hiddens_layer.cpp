/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/assemble_tokens_hidden_layer.h"

#include <vector>

#include "kernels/kunlun/basic_kernel_wrapper.h"

namespace xpukernel = xpytorch::xpu::api;

namespace ksana_llm {

Status AssembleTokensHiddenLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                       std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

template <typename T>
Status AssembleTokensHiddenLayer::ForwardT(const Tensor& hidden_states, const Tensor& accepted_token_indices,
                                           Tensor& output) {
  // index_select along axis=0: output[i] = hidden_states[index[i], :]
  //
  // Replaces the previous StreamSynchronize -> D2H(indices) -> StreamSynchronize
  // -> host loop of per-row D2D pattern (two host syncs on the compute stream
  // per step) with a single on-device index_select kernel. Indices are stored
  // as uint64_t; they are non-negative row offsets so reinterpret_cast to
  // int64_t is safe.
  auto* xpuctx = static_cast<xpukernel::Context*>(
      context_->device->ext->GetXpuKernelContext(context_->device->GetPrimaryComputeStream().Get()));

  const int64_t total_token_num = static_cast<int64_t>(hidden_states.shape[0]);
  const int64_t hidden_size = static_cast<int64_t>(hidden_states.shape[1]);
  const int64_t num_logits = static_cast<int64_t>(accepted_token_indices.shape[0]);

  // Publish output metadata before launching the kernel: both shape and dtype
  // are known from the inputs, and on the error path the caller must not be
  // left inspecting the previous forward's stale shape.
  output.shape = {static_cast<size_t>(num_logits), static_cast<size_t>(hidden_size)};
  output.dtype = hidden_states.dtype;

  if (num_logits == 0) {
    return Status();
  }

  const T* src = hidden_states.GetPtr<T>();
  const int64_t* idx = reinterpret_cast<const int64_t*>(accepted_token_indices.GetPtr<void>());
  T* dst = output.GetPtr<T>();

  if (num_logits == 0) {
    return Status();
  }

  int ret = XpuIndexSelect<T, int64_t>(xpuctx, src, idx, dst, {total_token_num, hidden_size}, num_logits, /*axis=*/0);
  if (ret != 0) {
    KLLM_LOG_ERROR << "AssembleTokensHiddenLayer: XpuIndexSelect failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "XpuIndexSelect failed in AssembleTokensHiddenLayer.");
  }
  return Status();
}

Status AssembleTokensHiddenLayer::Forward(const Tensor* hidden_states, const Tensor* accepted_token_indices,
                                          Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *hidden_states, *accepted_token_indices, *output);
}

}  // namespace ksana_llm
