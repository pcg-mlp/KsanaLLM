/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/layernorm_layer.h"
#include <cstdint>
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/kernels/ascend/layernorm/layernorm.h"
#include "csrc/kernels/ascend/rmsnorm/rmsnorm.h"
#include "csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

template <typename T>
Status LayernormLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  int parameter_index = 0;
  context_ = context;
  rank_ = rank;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  return Status();
}

template <typename T>
Status LayernormLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int64_t batch_size = input_tensors[0].shape[0];
  int64_t seq_len = input_tensors[0].shape[1];
  int64_t hidden_size = input_tensors[0].shape[2];
  size_t workspace_needed = batch_size * seq_len * hidden_size * sizeof(float) * 3;
  // NOTE(karlluo): allocate the workspace for float32
  if (workspace_block_id_ == -1 || workspace_size_ == 0) {
    workspace_size_ = workspace_needed;
    GetBlockManager()->AllocateContiguous(workspace_size_, workspace_block_id_);
  }
  // NOTE(karlluo): not enough, reallocate
  if (workspace_size_ < workspace_needed) {
    GetBlockManager()->FreeContiguous(workspace_block_id_);
    GetBlockManager()->AllocateContiguous(workspace_needed, workspace_block_id_);
    workspace_size_ = workspace_needed;
  }
  void* workspace_buf_ptr;
  GetBlockManager()->GetContiguousPtr(workspace_block_id_, workspace_buf_ptr);

  std::vector<int64_t> lm_input_shape = {batch_size, seq_len, hidden_size};
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

  // TODO(karlluo): support beta
  T* beta_ptr = nullptr;
  llm_kernels::ascend::InvokeRmsLayerNorm<T>(
      (T*)lm_output_tensor_buf_ptr, (T*)lm_input_tensor_buf_ptr, (T*)lm_weight_tensor_buf_ptr, beta_ptr, rms_norm_eps_,
      batch_size * seq_len, hidden_size, context_->GetComputeStreams()[rank_].Get(), GetWorkSpaceFunc());

  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;

  ACL_CHECK(aclDestroyTensor(lm_input_tensor_ptr));
  ACL_CHECK(aclDestroyTensor(lm_weight_tensor_ptr));
  ACL_CHECK(aclDestroyTensor(lm_output_tensor_ptr));

  return Status();
}
template class LayernormLayer<float>;
template class LayernormLayer<float16>;
}  // namespace ksana_llm
