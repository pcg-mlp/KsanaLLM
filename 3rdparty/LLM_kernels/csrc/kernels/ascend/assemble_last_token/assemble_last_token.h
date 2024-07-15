/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <vector>

#include "acl/acl.h"

namespace llm_kernels {
namespace ascend {

template <typename DTYPE>
void InvokeAssembleLastToken(DTYPE* input, size_t* ids_offsets, size_t* prefix_offsets, const int32_t batch_size,
                             const int32_t hidden_units_num, DTYPE* output, aclrtStream& stream,
                             void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels