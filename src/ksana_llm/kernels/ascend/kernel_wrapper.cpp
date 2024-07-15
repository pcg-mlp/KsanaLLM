/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/kernels/ascend/argmax/argmax.h"
#include "csrc/kernels/ascend/embedding/embedding.h"
#include "csrc/kernels/ascend/permute/permute.h"
#include "csrc/kernels/ascend/pointwise/pointwise.h"
#include "csrc/kernels/ascend/transpose/transpose.h"

#include "ksana_llm/kernels/argmax.h"
#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

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

void LookupEmbedding(const aclTensor* input_ids, const aclTensor* embedding_table, const aclTensor* position_table,
                     aclTensor* output, aclrtStream stream, WorkSpaceFunc ws_func) {
  llm_kernels::ascend::LookupEmbedding(input_ids, embedding_table, position_table, output, stream, ws_func);
}

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr) {
  if (tensor.dtype == DataType::TYPE_BF16 || target_dtype == DataType::TYPE_BF16) {
    throw std::invalid_argument("Invalid cast compute type, only support float16 to float32 or float32 to float16.");
  } else if (tensor.dtype == target_dtype) {
    // No need to convert
  } else {
    throw std::runtime_error(
        fmt::format("CastInplace from type {} to {} is not yet implement", tensor.dtype, target_dtype));
  }
  tensor.dtype = target_dtype;
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

template <typename T>
Status ArgMax(const T* input, const uint32_t* ids_offset, const int32_t batch_size, const int32_t vocab_size,
              uint32_t* result, Stream& stream, void* workspace_ptr) {
  if (ids_offset != nullptr) {
    throw std::runtime_error("Not supported ids offset");
    return Status(RET_RUNTIME, "argmax not supported ids offset");
  }
  llm_kernels::ascend::InvokeArgmax<T>(input, ids_offset, batch_size, vocab_size, result, stream.Get(),
                                       llm_kernels::utils::GetTestWorkSpaceFunc);
  return Status();
}

#define INSTANTIATE_ARG_MAX(T)                                                                 \
  template Status ArgMax(const T* input, const uint32_t* ids_offset, const int32_t batch_size, \
                         const int32_t vocab_size, uint32_t* result, Stream& stream, void* workspace_ptr);

INSTANTIATE_ARG_MAX(float);
INSTANTIATE_ARG_MAX(float16);

#undef INSTANTIATE_ARG_MAX

}  // namespace ksana_llm
