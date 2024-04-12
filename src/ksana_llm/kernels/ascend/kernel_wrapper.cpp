/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "csrc/kernels/ascend/permute/permute.h"
#include "csrc/kernels/ascend/pointwise/pointwise.h"

#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/kernels/lookup_embedding.h"
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

Status Permute(Tensor& input_tensor, Tensor& output_tensor, const std::vector<size_t>& permutation, Stream& stream,
               void* workspace_ptr) {
  uint64_t workspace_size = 0ull;
  aclTensor* output = output_tensor.GetDeviceTensor();
  std::vector<int64_t> dims(permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    dims[i] = static_cast<int64_t>(permutation[i]);
  }
  void* input_workspace_ptr = nullptr;
  GetBlockManager()->GetContiguousPtr(input_tensor.GetBlockId(), input_workspace_ptr);
  llm_kernels::ascend::Permute(input_tensor.GetDeviceTensor(), input_workspace_ptr, &output, dims, stream.Get());
  return Status();
}

Status LookupEmbedding(const void* ids, const void* offset, const void* emb, const void* pos, void* output,
                       int vocab_size, int hidden_size, int bs, int step, int vocab_id, Stream& stream,
                       void* workspace_ptr) {
  // TODO(karlluo): compat ascend lookup
  // void llm::ascend::LookupEmbedding(aclTensor** gatherOutput, const int64_t vocab_size, const int64_t hidden_size,
  //                     const int64_t seq_len, void** embedding_table, void** input_ids, void** maxDev_a,
  //                     aclOpExecutor& executor, aclrtStream& stream);
  return Status();
}

}  // namespace ksana_llm