/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/mhc_layer.h"

#include "utils/logger.h"

namespace ksana_llm {

Status MhcLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}

size_t MhcLayer::GetMhcPreNumSplits(int token_num, int hc_hidden_dim) { return 0; }

template <typename T>
Status MhcLayer::PreForwardT(const Tensor& hidden, const Tensor& fn_w, const Tensor& scale_w, const Tensor& base_w,
                             Tensor& out, Tensor& mix_h, Tensor& norm_buf, Tensor& h_post, Tensor& h_res) {
  KLLM_THROW("MhcLayer not implement in Ascend.");
  return Status();
}

template <typename T>
Status MhcLayer::PreForwardWithNormT(const Tensor& hidden, const Tensor& fn_w, const Tensor& scale_w,
                                     const Tensor& base_w, const Tensor& norm_w, Tensor& out, Tensor& mix_h,
                                     Tensor& norm_buf, Tensor& h_post, Tensor& h_res) {
  KLLM_THROW("MhcLayer not implement in Ascend.");
  return Status();
}

template <typename T>
Status MhcLayer::PostForwardT(const Tensor& hidden, const Tensor& h_post, const Tensor& h_res, Tensor& output) {
  KLLM_THROW("MhcLayer not implement in Ascend.");
  return Status();
}

template <typename T>
Status MhcLayer::InitialForwardT(const Tensor& hidden, Tensor& output) {
  KLLM_THROW("MhcLayer not implement in Ascend.");
  return Status();
}

template <typename T>
Status MhcLayer::FinalForwardT(const Tensor& hidden, const Tensor& fn_w, const Tensor& scale_w, const Tensor& base_w,
                               Tensor& out, Tensor& mix_h, Tensor& norm_buf) {
  KLLM_THROW("MhcLayer not implement in Ascend.");
  return Status();
}

Status MhcLayer::ForwardPre(const Tensor* hidden, const Tensor* fn_w, const Tensor* scale_w, const Tensor* base_w,
                            Tensor* out, Tensor* mix_h, Tensor* norm_buf, Tensor* h_post, Tensor* h_res,
                            const LayerExecContext& exec_ctx) {
  KLLM_THROW("MhcLayer not implement in Ascend.");
  return Status();
}

Status MhcLayer::ForwardPreWithNorm(const Tensor* hidden, const Tensor* fn_w, const Tensor* scale_w,
                                    const Tensor* base_w, const Tensor* norm_w, Tensor* out, Tensor* mix_h,
                                    Tensor* norm_buf, Tensor* h_post, Tensor* h_res, const LayerExecContext& exec_ctx) {
  KLLM_THROW("MhcLayer not implement in Ascend.");
  return Status();
}

Status MhcLayer::ForwardPost(const Tensor* hidden, const Tensor* h_post, const Tensor* h_res, Tensor* output,
                             const LayerExecContext& exec_ctx) {
  KLLM_THROW("MhcLayer not implement in Ascend.");
  return Status();
}

Status MhcLayer::ForwardInitial(const Tensor* hidden, Tensor* output, const LayerExecContext& exec_ctx) {
  KLLM_THROW("MhcLayer not implement in Ascend.");
  return Status();
}

Status MhcLayer::ForwardFinal(const Tensor* hidden, const Tensor* fn_w, const Tensor* scale_w, const Tensor* base_w,
                              Tensor* out, Tensor* mix_h, Tensor* norm_buf, const LayerExecContext& exec_ctx) {
  KLLM_THROW("MhcLayer not implement in Ascend.");
  return Status();
}

}  // namespace ksana_llm
