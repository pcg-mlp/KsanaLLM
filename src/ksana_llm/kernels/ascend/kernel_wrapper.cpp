/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "csrc/kernels/ascend/pointwise/pointwise.h"
#include "csrc/kernels/ascend/transpose/transpose.h"

#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/kernels/permute.h"

namespace ksana_llm {

aclDataType CastDataTypeToAclDataType(const DataType dtype) {
  switch (dtype) {
    case DataType::TYPE_FP16:
      return aclDataType::ACL_FLOAT16;
    default:
      return aclDataType::ACL_FLOAT;
  }
}

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr) {
  uint64_t workspace_size = 0ull;
  aclTensor* output = tensor.GetDeviceTensor();
  llm_kernels::ascend::Cast(tensor.GetDeviceTensor(), CastDataTypeToAclDataType(target_dtype), &output, &workspace_ptr,
                            workspace_size, stream.Get());
  return Status();
}

Status Permute(Tensor& input_tensor, Tensor& output_tensor, const std::vector<size_t>& permutation,
               Stream& stream, void* workspace_ptr) {
  uint64_t workspace_size = 0ull;
  aclTensor* output = output_tensor.GetDeviceTensor();
  llm_kernels::ascend::Transpose(input_tensor.GetDeviceTensor(), &output, &workspace_ptr, workspace_size, stream.Get());
  return Status();
}

}  // namespace ksana_llm