/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <unordered_map>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

namespace llm_kernels {
namespace ascend {

void RMSLayerNorm(const aclTensor* input, const aclTensor* weight, float eps, aclTensor** output, aclrtStream& stream,
                  void (*ws_func)(size_t, void**), void* workspace_buf_ptr = nullptr);

}  // namespace ascend
}  // namespace llm_kernels
