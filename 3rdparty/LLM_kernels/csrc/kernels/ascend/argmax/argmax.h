/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

namespace llm_kernels {
namespace ascend {

template <typename T>
void InvokeArgmax(const T* input, const uint32_t* ids_offset, const int32_t batch_size, const int32_t vocab_size,
                  uint32_t* result, aclrtStream& stream, void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels
