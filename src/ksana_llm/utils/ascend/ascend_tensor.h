/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/common_tensor.h"
#include "ksana_llm/utils/device_types.h"

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

namespace ksana_llm {

template <>
struct DeviceTensorTypeTraits<DEVICE_TYPE_ASCEND> {
  typedef aclTensor* value_type;
};

template <>
void TensorT<DEVICE_TYPE_ASCEND>::InitializeDeviceTensor();

template <>
aclTensor* TensorT<DEVICE_TYPE_ASCEND>::GetDeviceTensor();

template <>
aclTensor* TensorT<DEVICE_TYPE_ASCEND>::ResetDeviceTensor(const DataType dtype, const std::vector<int64_t> shape);

// Print acl tensor information.
void PrintAclTensorMeta(const aclTensor* tensor, const std::string& name);

}  // namespace ksana_llm
