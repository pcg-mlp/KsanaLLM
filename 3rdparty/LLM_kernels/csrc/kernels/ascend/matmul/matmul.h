/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"
#include "csrc/utils/ascend/common.h"

constexpr uint32_t CUBE_CORE_NUM = 24;

namespace llm_kernels {
namespace ascend {

aclError InvokeAclNNMatMul(const aclTensor* input, const aclTensor* weight,
                           const llm_kernels::utils::ACLNNMatmulComputeType cube_math_type, aclTensor** output,
                           aclrtStream& stream, void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels
