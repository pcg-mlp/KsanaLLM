/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/mhc_layer.h"

#include "csrc/kernels/nvidia/expand/expand.h"
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

Status MhcLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx) {
  if (input_tensors.size() == 5 && output_tensors.size() == 5) {
    DISPATCH_BY_3_DTYPE(inter_data_type_, PreForwardWithNormT, input_tensors, output_tensors);
  } else if (input_tensors.size() == 4 && output_tensors.size() == 5) {
    DISPATCH_BY_3_DTYPE(inter_data_type_, PreForwardT, input_tensors, output_tensors);
  } else if (input_tensors.size() == 3 && output_tensors.size() == 1) {
    DISPATCH_BY_3_DTYPE(inter_data_type_, PostForwardT, input_tensors, output_tensors);
  } else if (input_tensors.size() == 1 && output_tensors.size() == 1) {
    DISPATCH_BY_3_DTYPE(inter_data_type_, InitialForwardT, input_tensors, output_tensors);
  } else {
    KLLM_CHECK(input_tensors.size() == 4 && output_tensors.size() == 3);
    DISPATCH_BY_3_DTYPE(inter_data_type_, FinalForwardT, input_tensors, output_tensors);
  }
  return Status();
}

// PreForwardT: Down-project input before attn/mlp
// Inputs:
//   input_tensors[0]: hidden_states [t, hc * d] T
//   input_tensors[1]: fn_weight [(2 + hc) * hc, hc * d] FP32
//   input_tensors[2]: scale_weight [3] FP32
//   input_tensors[3]: base_weight [(2 + hc) * hc] FP32
// Outputs:
//   output_tensors[0]: projected_out [t, d] T
//   output_tensors[1]: mix_h_buffer [t, (2 + hc) * hc] FP32
//   output_tensors[2]: norm_buffer [t] FP32
//   output_tensors[3]: h_post_buffer [t, hc] FP32
//   output_tensors[4]: h_res_buffer [t, hc * hc] FP32
template <typename T>
Status MhcLayer::PreForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK(input_tensors[0].shape.size() == 2);
  KLLM_CHECK(input_tensors[1].shape.size() == 2);
  KLLM_CHECK(input_tensors[0].shape[1] == input_tensors[1].shape[1]);

  const size_t token_num = input_tensors[0].shape[0];
  const size_t num_splits = GetMhcPreNumSplits(token_num, input_tensors[0].shape[1]);
  output_tensors[1].shape = {num_splits, token_num, input_tensors[1].shape[0]};
  output_tensors[2].shape = {num_splits, token_num};
  // Call DeepGEMM wrapper
  auto wrapper = ksana_llm::nvidia::DeepSeekDeepGEMMWrapper::GetInstance();
  wrapper->Tf32HcPreNormGemm(input_tensors[0], input_tensors[1], output_tensors[1], output_tensors[2], num_splits,
                             rank_, context_->device->GetPrimaryComputeStream().Get());

  // Call TileLang wrapper
  Singleton<TritonWrapper>::GetInstance()->InvokeMhcPreBigFuseKernel<T>(
      output_tensors[1].GetPtr<void>(), output_tensors[2].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
      input_tensors[3].GetPtr<void>(), input_tensors[0].GetPtr<void>(), output_tensors[3].GetPtr<void>(),
      output_tensors[4].GetPtr<void>(), output_tensors[0].GetPtr<void>(), token_num, hidden_dim_, rms_norm_eps_,
      hc_eps_, /*hc_sinkhorn_eps*/ hc_eps_,
      /*hc_post_mult_value*/ 2, hc_sinkhorn_iters_, num_splits, hc_mult_,
      context_->device->GetPrimaryComputeStream().Get());

  output_tensors[0].shape = {token_num, hidden_dim_};
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

// PreForwardWithNormT: Down-project input before attn/mlp, with fused RMSNorm
// Inputs:
//   input_tensors[0]: hidden_states [t, hc * d] T
//   input_tensors[1]: fn_weight [(2 + hc) * hc, hc * d] FP32
//   input_tensors[2]: scale_weight [3] FP32
//   input_tensors[3]: base_weight [(2 + hc) * hc] FP32
//   input_tensors[4]: norm_weight [d] T (RMSNorm weight)
// Outputs:
//   output_tensors[0]: projected_out [t, d] T  (already RMSNorm'd)
//   output_tensors[1]: mix_h_buffer [t, (2 + hc) * hc] FP32
//   output_tensors[2]: norm_buffer [t] FP32
//   output_tensors[3]: h_post_buffer [t, hc] FP32
//   output_tensors[4]: h_res_buffer [t, hc * hc] FP32
template <typename T>
Status MhcLayer::PreForwardWithNormT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK(input_tensors[0].shape.size() == 2);
  KLLM_CHECK(input_tensors[1].shape.size() == 2);
  KLLM_CHECK(input_tensors[0].shape[1] == input_tensors[1].shape[1]);

  const size_t token_num = input_tensors[0].shape[0];
  const size_t num_splits = GetMhcPreNumSplits(token_num, input_tensors[0].shape[1]);
  output_tensors[1].shape = {num_splits, token_num, input_tensors[1].shape[0]};
  output_tensors[2].shape = {num_splits, token_num};
  // Call DeepGEMM wrapper
  auto wrapper = ksana_llm::nvidia::DeepSeekDeepGEMMWrapper::GetInstance();
  wrapper->Tf32HcPreNormGemm(input_tensors[0], input_tensors[1], output_tensors[1], output_tensors[2], num_splits,
                             rank_, context_->device->GetPrimaryComputeStream().Get());

  // Call fused TileLang wrapper (MHC pre + RMSNorm in one kernel)
  Singleton<TritonWrapper>::GetInstance()->InvokeMhcPreBigFuseWithNormKernel<T>(
      output_tensors[1].GetPtr<void>(), output_tensors[2].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
      input_tensors[3].GetPtr<void>(), input_tensors[0].GetPtr<void>(), input_tensors[4].GetPtr<void>(),
      output_tensors[3].GetPtr<void>(), output_tensors[4].GetPtr<void>(), output_tensors[0].GetPtr<void>(), token_num,
      hidden_dim_, rms_norm_eps_, hc_eps_, /*hc_sinkhorn_eps*/ hc_eps_,
      /*hc_post_mult_value*/ 2, hc_sinkhorn_iters_, num_splits, hc_mult_,
      context_->device->GetPrimaryComputeStream().Get());

  output_tensors[0].shape = {token_num, hidden_dim_};
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

// PostForwardT: Combine input and residual after attn/mlp
// Inputs:
//   input_tensors[0]: hidden_states [t, d] T
//   input_tensors[1]: h_post_buffer [t, hc] FP32
//   input_tensors[2]: h_res_buffer [t, hc * hc] FP32
// Outputs:
//   output_tensors[0]: combined_out [t, hc * d] T
template <typename T>
Status MhcLayer::PostForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK(input_tensors[0].shape.size() == 2);

  const size_t token_num = input_tensors[0].shape[0];
  // Call TileLang wrapper
  Singleton<TritonWrapper>::GetInstance()->InvokeMhcPostFwdKernel<T>(
      input_tensors[2].GetPtr<void>(), output_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(),
      input_tensors[0].GetPtr<void>(), output_tensors[0].GetPtr<void>(), token_num, hc_mult_, hidden_dim_,
      /*n_thr*/ 128, /*h_blk*/ 1024, context_->device->GetPrimaryComputeStream().Get());

  output_tensors[0].shape = {token_num, hc_mult_ * hidden_dim_};
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

// InitialForwardT: Up-project input after embedding lookup
// Inputs:
//   input_tensors[0]: hidden_states [t, d] T
// Outputs:
//   output_tensors[0]: expanded_out [t, hc, d] T
template <typename T>
Status MhcLayer::InitialForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK(input_tensors[0].shape.size() == 2);

  const size_t token_num = input_tensors[0].shape[0];
  const size_t hidden_size = input_tensors[0].shape[1];
  llm_kernels::nvidia::InvokeExpand<T>(input_tensors[0].GetPtr<const T>(), output_tensors[0].GetPtr<T>(), token_num,
                                       hc_mult_, hidden_size, context_->device->GetPrimaryComputeStream().Get());

  output_tensors[0].shape = {token_num, hc_mult_ * hidden_size};
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

// FinalForwardT: Down-project input before lmhead
// Inputs:
//   input_tensors[0]: hidden_states [t, hc * d] T
//   input_tensors[1]: fn_weight [2 * hc, hc * d] FP32
//                     NOTE: Pad from hc to 2*hc to work around DeepGEMM kernel limitations
//   input_tensors[2]: scale_weight [1] FP32
//   input_tensors[3]: base_weight [hc] FP32
// Outputs:
//   output_tensors[0]: projected_out [t, d] T
//   output_tensors[1]: mix_h_buffer [t, 2 * hc] FP32
//   output_tensors[2]: norm_buffer [t] FP32
template <typename T>
Status MhcLayer::FinalForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK(input_tensors[0].shape.size() == 2);
  KLLM_CHECK(input_tensors[1].shape.size() == 2);
  KLLM_CHECK(input_tensors[0].shape[1] == input_tensors[1].shape[1]);
  KLLM_CHECK(input_tensors[1].shape[0] == 2 * hc_mult_);  // Verify padding

  const size_t token_num = input_tensors[0].shape[0];
  const size_t num_splits = GetMhcPreNumSplits(token_num, input_tensors[0].shape[1]);
  output_tensors[1].shape = {num_splits, token_num, input_tensors[1].shape[0]};
  output_tensors[2].shape = {num_splits, token_num};
  // Call DeepGEMM wrapper
  auto wrapper = ksana_llm::nvidia::DeepSeekDeepGEMMWrapper::GetInstance();
  wrapper->Tf32HcPreNormGemm(input_tensors[0], input_tensors[1], output_tensors[1], output_tensors[2], num_splits,
                             rank_, context_->device->GetPrimaryComputeStream().Get());

  // Call TileLang wrapper
  Singleton<TritonWrapper>::GetInstance()->InvokeMhcHeadBigFuseKernel<T>(
      output_tensors[1].GetPtr<void>(), output_tensors[2].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
      input_tensors[3].GetPtr<void>(), input_tensors[0].GetPtr<void>(), output_tensors[0].GetPtr<void>(), token_num,
      hidden_dim_, rms_norm_eps_, hc_eps_, num_splits, hc_mult_, context_->device->GetPrimaryComputeStream().Get());

  output_tensors[0].shape = {token_num, hidden_dim_};
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm
