/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <stdint.h>

#include "acl/acl.h"

#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

template <typename T>
void InvokeSiluMul(const T* input, const T* weight, const size_t m, const size_t n, T* output, aclrtStream stream,
                   llm_kernels::utils::WorkSpaceFunc ws_func);

}  // namespace ascend
}  // namespace llm_kernels