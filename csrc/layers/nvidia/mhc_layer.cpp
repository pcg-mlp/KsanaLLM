/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/mhc_layer.h"

#include "csrc/kernels/nvidia/expand/expand.h"
#include "csrc/kernels/nvidia/mhc/mhc_kernel.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "kernels/nvidia/basic_kernel_wrapper.h"
#include "kernels/nvidia/deepseek_deepgemm_wrapper.h"
#include "kernels/nvidia/triton_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {

Status MhcLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  hc_mult_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  KLLM_CHECK_WITH_INFO(hc_mult_ == 4, "MhcLayer requires hc_mult == 4");
  hc_sinkhorn_iters_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  hc_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  hidden_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  max_token_num_ = runtime_config.max_step_token_num;
  return Status();
}

size_t MhcLayer::GetMhcPreNumSplits(int token_num, int hc_hidden_dim) {
  constexpr int block_m = 64, block_k = 64;
  const int grid_size = std::max(1, (token_num + block_m - 1) / block_m);
  const int num_block_k = (hc_hidden_dim + block_k - 1) / block_k;
  const int num_splits = std::max(1, std::min(llm_kernels::utils::GetSMCount() / grid_size, num_block_k / 4));
  // Round down to nearest supported num_splits.
  constexpr size_t supported_num_splits[] = {1, 2, 16, 32, 64};
  auto it = std::upper_bound(std::begin(supported_num_splits), std::end(supported_num_splits),
                             static_cast<size_t>(num_splits));
  return *(--it);
}

// PreForwardT: Down-project input before attn/mlp
template <typename T>
Status MhcLayer::PreForwardT(const Tensor& hidden, const Tensor& fn_w, const Tensor& scale_w, const Tensor& base_w,
                             Tensor& out, Tensor& mix_h, Tensor& norm_buf, Tensor& h_post, Tensor& h_res) {
  KLLM_CHECK(hidden.shape.size() == 2);
  KLLM_CHECK(fn_w.shape.size() == 2);
  KLLM_CHECK(hidden.shape[1] == fn_w.shape[1]);

  const size_t token_num = hidden.shape[0];
  const size_t num_splits = GetMhcPreNumSplits(token_num, hidden.shape[1]);
  mix_h.shape = {num_splits, token_num, fn_w.shape[0]};
  norm_buf.shape = {num_splits, token_num};
  auto wrapper = ksana_llm::nvidia::DeepSeekDeepGEMMWrapper::GetInstance();
  wrapper->Tf32HcPreNormGemm(hidden, fn_w, mix_h, norm_buf, num_splits, rank_,
                             context_->device->GetPrimaryComputeStream().Get());

  Singleton<TritonWrapper>::GetInstance()->InvokeMhcPreBigFuseKernel<T>(
      mix_h.GetPtr<void>(), norm_buf.GetPtr<void>(), scale_w.GetPtr<void>(), base_w.GetPtr<void>(),
      hidden.GetPtr<void>(), h_post.GetPtr<void>(), h_res.GetPtr<void>(), out.GetPtr<void>(), token_num, hidden_dim_,
      rms_norm_eps_, hc_eps_, /*hc_sinkhorn_eps*/ hc_eps_,
      /*hc_post_mult_value*/ 2, hc_sinkhorn_iters_, num_splits, hc_mult_,
      context_->device->GetPrimaryComputeStream().Get());

  out.shape = {token_num, hidden_dim_};
  out.dtype = hidden.dtype;
  return Status();
}

// PreForwardWithNormT: Down-project + fused RMSNorm
template <typename T>
Status MhcLayer::PreForwardWithNormT(const Tensor& hidden, const Tensor& fn_w, const Tensor& scale_w,
                                     const Tensor& base_w, const Tensor& norm_w, Tensor& out, Tensor& mix_h,
                                     Tensor& norm_buf, Tensor& h_post, Tensor& h_res) {
  KLLM_CHECK(hidden.shape.size() == 2);
  KLLM_CHECK(fn_w.shape.size() == 2);
  KLLM_CHECK(hidden.shape[1] == fn_w.shape[1]);

  const size_t token_num = hidden.shape[0];
  const size_t num_splits = GetMhcPreNumSplits(token_num, hidden.shape[1]);
  mix_h.shape = {num_splits, token_num, fn_w.shape[0]};
  norm_buf.shape = {num_splits, token_num};
  auto wrapper = ksana_llm::nvidia::DeepSeekDeepGEMMWrapper::GetInstance();
  wrapper->Tf32HcPreNormGemm(hidden, fn_w, mix_h, norm_buf, num_splits, rank_,
                             context_->device->GetPrimaryComputeStream().Get());

  Singleton<TritonWrapper>::GetInstance()->InvokeMhcPreBigFuseWithNormKernel<T>(
      mix_h.GetPtr<void>(), norm_buf.GetPtr<void>(), scale_w.GetPtr<void>(), base_w.GetPtr<void>(),
      hidden.GetPtr<void>(), norm_w.GetPtr<void>(), h_post.GetPtr<void>(), h_res.GetPtr<void>(), out.GetPtr<void>(),
      token_num, hidden_dim_, rms_norm_eps_, hc_eps_, /*hc_sinkhorn_eps*/ hc_eps_,
      /*hc_post_mult_value*/ 2, hc_sinkhorn_iters_, num_splits, hc_mult_,
      context_->device->GetPrimaryComputeStream().Get());

  out.shape = {token_num, hidden_dim_};
  out.dtype = hidden.dtype;
  return Status();
}

// PostForwardT: Combine input and residual after attn/mlp
template <typename T>
Status MhcLayer::PostForwardT(const Tensor& hidden, const Tensor& h_post, const Tensor& h_res, Tensor& output) {
  KLLM_CHECK(hidden.shape.size() == 2);

  const size_t token_num = hidden.shape[0];
  llm_kernels::nvidia::mhc_post(
      /*out=*/output.GetPtr<void>(), hidden.GetPtr<void>(),
      /*residual=*/output.GetPtr<void>(), h_post.GetPtr<float>(), h_res.GetPtr<float>(), token_num, hidden_dim_,
      context_->device->GetPrimaryComputeStream().Get());

  output.shape = {token_num, hc_mult_ * hidden_dim_};
  output.dtype = hidden.dtype;
  return Status();
}

// InitialForwardT: Up-project input after embedding lookup
template <typename T>
Status MhcLayer::InitialForwardT(const Tensor& hidden, Tensor& output) {
  KLLM_CHECK(hidden.shape.size() == 2);

  const size_t token_num = hidden.shape[0];
  const size_t hidden_size = hidden.shape[1];
  llm_kernels::nvidia::InvokeExpand<T>(hidden.GetPtr<const T>(), output.GetPtr<T>(), token_num, hc_mult_, hidden_size,
                                       context_->device->GetPrimaryComputeStream().Get());

  output.shape = {token_num, hc_mult_ * hidden_size};
  output.dtype = hidden.dtype;
  return Status();
}

// FinalForwardT: Down-project input before lmhead
template <typename T>
Status MhcLayer::FinalForwardT(const Tensor& hidden, const Tensor& fn_w, const Tensor& scale_w, const Tensor& base_w,
                               Tensor& out, Tensor& mix_h, Tensor& norm_buf) {
  KLLM_CHECK(hidden.shape.size() == 2);
  KLLM_CHECK(fn_w.shape.size() == 2);
  KLLM_CHECK(hidden.shape[1] == fn_w.shape[1]);
  KLLM_CHECK(fn_w.shape[0] == 2 * hc_mult_);

  const size_t token_num = hidden.shape[0];
  const size_t num_splits = GetMhcPreNumSplits(token_num, hidden.shape[1]);
  mix_h.shape = {num_splits, token_num, fn_w.shape[0]};
  norm_buf.shape = {num_splits, token_num};
  auto wrapper = ksana_llm::nvidia::DeepSeekDeepGEMMWrapper::GetInstance();
  wrapper->Tf32HcPreNormGemm(hidden, fn_w, mix_h, norm_buf, num_splits, rank_,
                             context_->device->GetPrimaryComputeStream().Get());

  Singleton<TritonWrapper>::GetInstance()->InvokeMhcHeadBigFuseKernel<T>(
      mix_h.GetPtr<void>(), norm_buf.GetPtr<void>(), scale_w.GetPtr<void>(), base_w.GetPtr<void>(),
      hidden.GetPtr<void>(), out.GetPtr<void>(), token_num, hidden_dim_, rms_norm_eps_, hc_eps_, num_splits, hc_mult_,
      context_->device->GetPrimaryComputeStream().Get());

  out.shape = {token_num, hidden_dim_};
  out.dtype = hidden.dtype;
  return Status();
}

Status MhcLayer::ForwardPre(const Tensor* hidden, const Tensor* fn_w, const Tensor* scale_w, const Tensor* base_w,
                            Tensor* out, Tensor* mix_h, Tensor* norm_buf, Tensor* h_post, Tensor* h_res,
                            const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, PreForwardT, *hidden, *fn_w, *scale_w, *base_w, *out, *mix_h, *norm_buf,
                      *h_post, *h_res);
}

Status MhcLayer::ForwardPreWithNorm(const Tensor* hidden, const Tensor* fn_w, const Tensor* scale_w,
                                    const Tensor* base_w, const Tensor* norm_w, Tensor* out, Tensor* mix_h,
                                    Tensor* norm_buf, Tensor* h_post, Tensor* h_res, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, PreForwardWithNormT, *hidden, *fn_w, *scale_w, *base_w, *norm_w, *out, *mix_h,
                      *norm_buf, *h_post, *h_res);
}

Status MhcLayer::ForwardPost(const Tensor* hidden, const Tensor* h_post, const Tensor* h_res, Tensor* output,
                             const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, PostForwardT, *hidden, *h_post, *h_res, *output);
}

Status MhcLayer::ForwardInitial(const Tensor* hidden, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, InitialForwardT, *hidden, *output);
}

Status MhcLayer::ForwardFinal(const Tensor* hidden, const Tensor* fn_w, const Tensor* scale_w, const Tensor* base_w,
                              Tensor* out, Tensor* mix_h, Tensor* norm_buf, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, FinalForwardT, *hidden, *fn_w, *scale_w, *base_w, *out, *mix_h, *norm_buf);
}

}  // namespace ksana_llm
