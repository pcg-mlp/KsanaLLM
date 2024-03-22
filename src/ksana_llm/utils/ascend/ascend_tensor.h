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

}  // namespace ksana_llm
