/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "atb/atb_infer.h"
#include "atb/operation.h"
#include "utils.h"

namespace llm_kernels {
namespace ascend {

const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;
const int NUM4 = 4;
const int NUM5 = 5;
const int NUM6 = 6;

atb::SVector<int64_t> GetCopyTensorStride(atb::Dims &tensor_dims);

atb::Tensor SqueezeBatchSeq(atb::Tensor atb_tensor);

}  // namespace ascend
}  // namespace llm_kernels