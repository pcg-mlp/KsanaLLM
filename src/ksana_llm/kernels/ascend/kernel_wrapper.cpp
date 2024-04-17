/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "csrc/kernels/ascend/embedding/embedding.h"
#include "csrc/kernels/ascend/permute/permute.h"
#include "csrc/kernels/ascend/pointwise/pointwise.h"

#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/kernels/permute.h"
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
  uint64_t workspace_size = 0ull;
  aclTensor* output = output_tensor.GetDeviceTensor();
  std::vector<int64_t> dims(permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    dims[i] = static_cast<int64_t>(permutation[i]);
  }
  void* input_workspace_ptr = input_tensor.GetPtr<void>();
  llm_kernels::ascend::Permute(input_tensor.GetDeviceTensor(), &input_workspace_ptr, &output, dims, stream.Get(),
                               GetWorkSpaceFunc());
  return Status();
}

}  // namespace ksana_llm
