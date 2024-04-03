/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "csrc/kernels/ascend/pointwise/pointwise.h"

#include "ksana_llm/kernels/cast.h"

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
  void* wsDev = nullptr;
  uint64_t wsSize = 0;
  aclTensor* output = tensor.GetDeviceTensor();
  llm_kernels::ascend::Cast(tensor.GetDeviceTensor(), CastDataTypeToAclDataType(target_dtype), &output, &wsDev, wsSize,
                            stream.Get());
  return Status();
}

}  // namespace ksana_llm