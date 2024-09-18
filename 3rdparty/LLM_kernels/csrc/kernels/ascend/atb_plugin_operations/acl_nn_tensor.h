/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>
#include <string>

using UpdateDataPtrFunc = std::function<aclnnStatus(aclOpExecutor *, const size_t, aclTensor *, void *)>;

namespace llm_kernels {
namespace ascend {

class AclNNTensor {
 public:
  atb::Tensor atb_tensor;
  atb::SVector<int64_t> strides = {};
  aclTensor *tensor = nullptr;
  int tensor_idx = -1;  // aclTensor在aclExecutor中的index
};
}  // namespace ascend
}  // namespace llm_kernels