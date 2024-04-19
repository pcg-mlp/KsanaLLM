/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

namespace llm_kernels {
namespace ascend {

void ArgMax(const aclTensor* argMaxInput, const int64_t argMaxDim, const bool argMaxKeepdim, aclTensor** argMaxOutput,
            aclrtStream& stream, void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels
