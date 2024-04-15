/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/silu_mul_layer.h"
#include "csrc/kernels/ascend/activation/activation.h"
#include "csrc/kernels/ascend/elementwise/elementwise.h"
#include "csrc/utils/ascend/common.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

namespace ksana_llm {

Status SiluMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int64_t seq_len = static_cast<int64_t>(input_tensors[0].shape[0]);
  int64_t ffn_size = static_cast<int64_t>(input_tensors[0].shape[1]);
  std::vector<int64_t> silu_output_shape = {1, seq_len, ffn_size};
  aclTensor* silu_output = nullptr;
  void* silu_output_buf_ptr = input_tensors[0].GetPtr<void>();
  llm_kernels::utils::CreateAclTensorWithData(silu_output_shape, &silu_output_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &silu_output);
  aclTensor* silu_input = nullptr;
  void* silu_input_buf_ptr = input_tensors[0].GetPtr<void>();
  llm_kernels::utils::CreateAclTensorWithData(silu_output_shape, &silu_input_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &silu_input);
  WorkSpaceFunc f = GetWorkSpaceFunc();
  constexpr uint64_t workspace_size = 1073741824ull;
  void* ws_addr_ptr = nullptr;
  f(workspace_size, &ws_addr_ptr);
  llm_kernels::ascend::Silu(silu_input, &silu_output, &ws_addr_ptr, workspace_size,
                            context_->GetComputeStreams()[rank_].Get());

  aclTensor* gated_weight = nullptr;
  void* gated_weight_buf_ptr = input_tensors[1].GetPtr<void>();
  std::vector<int64_t> gated_weight_shape = {1, seq_len, ffn_size};
  llm_kernels::utils::CreateAclTensorWithData(gated_weight_shape, &gated_weight_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &gated_weight);
  aclTensor* mul_output = nullptr;
  std::vector<int64_t> mul_output_shape = {1, seq_len, ffn_size};
  llm_kernels::utils::CreateAclTensorWithData(mul_output_shape, &silu_output_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &mul_output);
  llm_kernels::ascend::Mul(gated_weight, silu_output, &mul_output, ws_addr_ptr, workspace_size,
                           context_->GetComputeStreams()[rank_].Get());

  aclDestroyTensor(mul_output);
  aclDestroyTensor(gated_weight);
  aclDestroyTensor(silu_input);
  aclDestroyTensor(silu_output);
  return Status();
}
}  // namespace ksana_llm
