/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "csrc/kernels/ascend/embedding/embedding.h"
#include "csrc/kernels/ascend/permute/permute.h"
#include "csrc/kernels/ascend/pointwise/pointwise.h"
#include "csrc/kernels/ascend/reshape/reshape.h"
#include "csrc/kernels/ascend/transpose/transpose.h"

#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

void LookupEmbedding(const aclTensor* input_ids, const aclTensor* embedding_table, const aclTensor* position_table,
                     aclTensor* output, aclrtStream stream, WorkSpaceFunc ws_func) {
  llm_kernels::ascend::LookupEmbedding(input_ids, embedding_table, position_table, output, stream, ws_func);
}

aclDataType CastDataTypeToAclDataType(const DataType dtype) {
  switch (dtype) {
    case DataType::TYPE_FP16:
      return aclDataType::ACL_FLOAT16;
    case DataType::TYPE_FP32:
      return aclDataType::ACL_FLOAT;
    default:
      return aclDataType::ACL_FLOAT;
  }
}

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr) {
  Tensor* input_normal_tensor_ptr = reinterpret_cast<Tensor*>(&tensor);
  aclTensor* input_tensor_ptr = input_normal_tensor_ptr->GetDeviceTensor();
  std::vector<int64_t> input_shape = GetAclTensorShape(input_tensor_ptr);
  void* output_buffer_space_ptr = tensor.GetPtr<void>();
  aclTensor* reshaped_output_tensor = nullptr;
  llm_kernels::utils::CreateAclTensorWithData(input_shape, &(output_buffer_space_ptr),
                                              CastDataTypeToAclDataType(target_dtype), aclFormat::ACL_FORMAT_ND,
                                              &reshaped_output_tensor);
  llm_kernels::ascend::Cast(input_tensor_ptr, CastDataTypeToAclDataType(target_dtype), &reshaped_output_tensor,
                            stream.Get(), GetWorkSpaceFunc());
  tensor.ResetDeviceTensor(reshaped_output_tensor);
  tensor.dtype = target_dtype;
  ACL_CHECK(aclDestroyTensor(input_tensor_ptr));
  return Status();
}

Status Permute(Tensor& input_tensor, Tensor& output_tensor, const std::vector<size_t>& permutation, Stream& stream,
               void* workspace_ptr) {
  std::vector<size_t> input_shape = input_tensor.shape;
  aclTensor* input_acl_tensor = input_tensor.GetDeviceTensor();
  void* output_buf_ptr = output_tensor.GetPtr<void>();
  std::vector<int64_t> dims(permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    dims[i] = static_cast<int64_t>(permutation[i]);
  }
  aclTensor* permute_output = nullptr;
  void* input_acl_tensor_buf_ptr = input_tensor.GetPtr<void>();
  llm_kernels::ascend::Permute(input_acl_tensor, &input_acl_tensor_buf_ptr, &permute_output, dims, stream.Get(),
                               GetWorkSpaceFunc());
  int64_t* output_t_shape_ptr = nullptr;
  std::vector<int64_t> output_shape(input_shape.size(), 0);
  uint64_t output_t_dims_num = 0;
  ACL_CHECK_RET(aclGetViewShape(permute_output, &output_t_shape_ptr, &output_t_dims_num));
  for (uint64_t i = 0; i < output_t_dims_num; ++i) {
    output_shape[i] = output_t_shape_ptr[i];
    output_tensor.shape[i] = static_cast<size_t>(output_shape[i]);
  }
  aclDataType output_dtype;
  ACL_CHECK(aclGetDataType(input_acl_tensor, &output_dtype));
  aclTensor* output_acl_tensor = nullptr;
  llm_kernels::utils::CreateAclTensorWithData(output_shape, &output_buf_ptr, output_dtype, aclFormat::ACL_FORMAT_ND,
                                              &output_acl_tensor);
  llm_kernels::ascend::Copy(permute_output, &output_acl_tensor, stream.Get(), llm_kernels::utils::GetTestWorkSpaceFunc);
  output_tensor.ResetDeviceTensor(output_acl_tensor);
  ACL_CHECK(aclDestroyTensor(permute_output));
  return Status();
}

}  // namespace ksana_llm
