/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/matmul_layer.h"

#include "csrc/kernels/ascend/matmul/matmul.h"
#include "csrc/utils/ascend/common.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

template <typename T>
Status MatMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // TODO(karlluo): implement llm_kernels::ascend::MatMul
  size_t k = input_tensors[1].shape[0];
  size_t n = input_tensors[1].shape[1];
  size_t m = input_tensors[0].shape[0];

  std::vector<int64_t> matmul_input_shape = {1ul, m, k};  /*m, k*/
  std::vector<int64_t> matmul_weight_shape = {k, n};      /*k, n*/
  std::vector<int64_t> matmul_output_shape = {1ul, m, n}; /*m, n*/
  aclTensor* matmul_input = nullptr;
  aclTensor* matmul_weight = nullptr;
  aclTensor* matmul_output = nullptr;
  void* matmul_input_buf_ptr = input_tensors[0].GetPtr<void>();
  void* matmul_weight_buf_ptr = input_tensors[1].GetPtr<void>();
  void* matmul_output_buf_ptr = output_tensors[0].GetPtr<void>();
  llm_kernels::utils::CreateAclTensorWithData(matmul_input_shape, &matmul_input_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &matmul_input);
  llm_kernels::utils::CreateAclTensorWithData(matmul_weight_shape, &matmul_weight_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &matmul_weight);
  llm_kernels::utils::CreateAclTensorWithData(matmul_output_shape, &matmul_output_buf_ptr, aclDataType::ACL_FLOAT16,
                                              aclFormat::ACL_FORMAT_ND, &matmul_output);

  int mm_type = 0;
  llm_kernels::ascend::MatMul(matmul_input, matmul_weight, mm_type, &matmul_output,
                              context_->GetComputeStreams()[rank_].Get(), GetWorkSpaceFunc());

  output_tensors[0].shape[0] = 1ul;
  output_tensors[0].shape[1] = m;
  output_tensors[0].shape[2] = n;
  output_tensors[0].ResetDeviceTensor(matmul_output);

  ACL_CHECK(aclDestroyTensor(matmul_input));
  ACL_CHECK(aclDestroyTensor(matmul_weight));
  return Status();
}
template class MatMulLayer<float>;
template class MatMulLayer<float16>;
}  // namespace ksana_llm
