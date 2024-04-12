/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/layernorm_layer.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/kernels/ascend/layernorm/layernorm.h"
#include "csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

Status LayernormLayer::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  NLLM_LOG_DEBUG << fmt::format("rms_norm_eps {}", rms_norm_eps_);
  return Status();
}

Status LayernormLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  size_t seq_len = input_tensors[0].shape[0];
  size_t hidden_size = input_tensors[0].shape[1];
  std::vector<int64_t> lm_input_shape = {1, seq_len, hidden_size};
  aclTensor* lm_input_tensor_ptr = nullptr;
  void* lm_input_tensor_buf_ptr = input_tensors[0].GetPtr<void>();
  llm_kernels::utils::CreateAclTensorWithData(lm_input_shape, &lm_input_tensor_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &lm_input_tensor_ptr);

  std::vector<int64_t> lm_weight_shape = {1, 1, hidden_size};
  aclTensor* lm_weight_tensor_ptr = nullptr;
  void* lm_weight_tensor_buf_ptr = input_tensors[1].GetPtr<void>();
  llm_kernels::utils::CreateAclTensorWithData(lm_weight_shape, &lm_weight_tensor_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &lm_weight_tensor_ptr);

  aclTensor* lm_output_tensor_ptr = nullptr;
  void* lm_output_tensor_buf_ptr = output_tensors[0].GetPtr<void>();
  llm_kernels::utils::CreateAclTensorWithData(lm_input_shape, &lm_output_tensor_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &lm_output_tensor_ptr);

  uint64_t workspace_size = 0ull;
  WorkSpaceFunc f = GetWorkSpaceFunc();
  void* ws_addr_ptr = nullptr;
  f(workspace_size, &ws_addr_ptr);
  llm_kernels::ascend::RMSLayerNorm(lm_input_tensor_ptr, lm_weight_tensor_ptr, &lm_output_tensor_ptr, ws_addr_ptr,
                                    workspace_size, context_->GetComputeStreams()[rank_].Get());

  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;

  aclDestroyTensor(lm_input_tensor_ptr);
  aclDestroyTensor(lm_weight_tensor_ptr);
  aclDestroyTensor(lm_output_tensor_ptr);

  return Status();
}
}  // namespace ksana_llm
