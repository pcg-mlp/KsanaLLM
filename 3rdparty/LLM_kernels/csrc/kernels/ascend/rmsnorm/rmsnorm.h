/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <stdint.h>

#include "acl/acl.h"

namespace llm_kernels {
namespace ascend {

template <typename DTYPE>
void InvokeRmsLayerNorm(DTYPE* out, DTYPE* input, DTYPE* gamma, DTYPE* beta, const float layernorm_eps, const int32_t m,
                        const int32_t n, aclrtStream& stream, void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels