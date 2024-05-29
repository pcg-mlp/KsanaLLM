/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

#include "csrc/kernels/ascend/cast/cast_tiling.h"

namespace llm_kernels {
namespace ascend {

template <typename SRC_DTYPE, typename DST_DTYPE>
void InvokeCast(SRC_DTYPE* input, DST_DTYPE* output, uint32_t seq_len, uint32_t hidden_units_num, aclrtStream& stream,
                void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels