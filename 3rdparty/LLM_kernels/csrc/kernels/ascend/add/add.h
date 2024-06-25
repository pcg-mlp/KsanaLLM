/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <vector>

#include "acl/acl.h"

namespace llm_kernels {
namespace ascend {

template <typename DTYPE>
void InvokeAdd(DTYPE* input, DTYPE* output, DTYPE alpha, DTYPE* bias, uint32_t hidden_units_num, uint32_t total_seq_len,
               aclrtStream& stream, void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels