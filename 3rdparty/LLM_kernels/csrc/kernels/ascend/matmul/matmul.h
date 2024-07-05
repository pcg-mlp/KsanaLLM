/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

constexpr uint32_t CUBE_CORE_NUM = 24;

namespace llm_kernels {
namespace ascend {

aclError MatMul(const aclTensor* input, const aclTensor* weight, const int8_t matmulCubeMathType, aclTensor** output,
                aclrtStream& stream, void (*ws_func)(size_t, void**));

template <typename DTYPE>
void InvokeMatMul(const size_t m, const size_t n, const size_t k, DTYPE* input_device, DTYPE* weight_device,
                  DTYPE* bias_device, DTYPE* output_device, aclrtStream& stream, void (*ws_func)(size_t, void**));

#ifdef WITH_ACL_ATB
template <typename DTYPE>
void InvokeATBMatMul(const size_t m, const size_t n, const size_t k, DTYPE* input_device, DTYPE* weight_device,
                  DTYPE* bias_device, DTYPE* output_device, aclrtStream& stream, void (*ws_func)(size_t, void**));
#endif

}  // namespace ascend
}  // namespace llm_kernels
