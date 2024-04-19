/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

namespace llm_kernels {
namespace ascend {

void Cast(const aclTensor* castInput, const aclDataType castToType, aclTensor** castOutput, aclrtStream& stream,
          void (*ws_func)(size_t, void**));

void Pow(const aclTensor* input, const float powExponentValue, aclTensor** output, aclrtStream& stream,
         void (*ws_func)(size_t, void**));

void Mean(const aclTensor* input, std::vector<int64_t>& meanDimData, const bool keepdim, aclDataType dtype,
          aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**));

void Neg(const aclTensor* input, aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**));

void InplaceDiv(const aclTensor* input, aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**));

void InplaceSqrt(aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels
