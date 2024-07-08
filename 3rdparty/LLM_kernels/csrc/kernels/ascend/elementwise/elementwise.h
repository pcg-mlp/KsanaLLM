/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

namespace llm_kernels {
namespace ascend {

// output = input + scalar1*scalar2
void Adds(const aclTensor* input, const aclScalar* scalar1, const aclScalar* scalar2, aclTensor** addOutput,
          aclrtStream& stream, void (*ws_func)(size_t, void**));

void Mul(const aclTensor* mulInput1, const aclTensor* mulInput2, aclTensor** mulOutput, aclrtStream& stream,
         void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels
