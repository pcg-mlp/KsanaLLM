/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "csrc/kernels/ascend/embedding/embedding.h"
#include "csrc/kernels/ascend/permute/permute.h"
#include "csrc/kernels/ascend/pointwise/pointwise.h"

#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/kernels/permute.h"

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
  aclTensor* output = tensor.GetDeviceTensor();
  llm_kernels::ascend::Cast(tensor.GetDeviceTensor(), CastDataTypeToAclDataType(target_dtype), &output, stream.Get(),
                            GetWorkSpaceFunc());
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
