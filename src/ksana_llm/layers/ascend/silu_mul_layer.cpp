/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/silu_mul_layer.h"
#include "csrc/kernels/ascend/activation/activation.h"
#include "csrc/kernels/ascend/elementwise/elementwise.h"
#include "csrc/utils/ascend/common.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

template <typename T>
Status SiluMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int64_t seq_len = static_cast<int64_t>(input_tensors[0].shape[0]);
  int64_t ffn_size = static_cast<int64_t>(input_tensors[0].shape[1]);
  std::vector<int64_t> silu_output_shape = {1, seq_len, ffn_size};
  aclTensor* silu_output = nullptr;
  void* silu_output_buf_ptr = output_tensors[0].GetPtr<void>();
  llm_kernels::utils::CreateAclTensorWithData(silu_output_shape, &silu_output_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &silu_output);
  aclTensor* silu_input = nullptr;
  void* silu_input_buf_ptr = input_tensors[0].GetPtr<void>();
  llm_kernels::utils::CreateAclTensorWithData(silu_output_shape, &silu_input_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &silu_input);
  llm_kernels::ascend::Silu(silu_input, &silu_output, context_->GetComputeStreams()[rank_].Get(), GetWorkSpaceFunc());

  aclTensor* gated_weight = nullptr;
  void* gated_weight_buf_ptr = input_tensors[1].GetPtr<void>();
  std::vector<int64_t> gated_weight_shape = {1, seq_len, ffn_size};
  llm_kernels::utils::CreateAclTensorWithData(gated_weight_shape, &gated_weight_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &gated_weight);
  aclTensor* mul_output = nullptr;
  std::vector<int64_t> mul_output_shape = {1, seq_len, ffn_size};
  llm_kernels::utils::CreateAclTensorWithData(mul_output_shape, &silu_output_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &mul_output);
  llm_kernels::ascend::Mul(gated_weight, silu_output, &mul_output, context_->GetComputeStreams()[rank_].Get(),
                           GetWorkSpaceFunc());

  ACL_CHECK(aclDestroyTensor(mul_output));
  ACL_CHECK(aclDestroyTensor(gated_weight));
  ACL_CHECK(aclDestroyTensor(silu_input));
  ACL_CHECK(aclDestroyTensor(silu_output));
  return Status();
}
template class SiluMulLayer<float>;
template class SiluMulLayer<float16>;
}  // namespace ksana_llm
