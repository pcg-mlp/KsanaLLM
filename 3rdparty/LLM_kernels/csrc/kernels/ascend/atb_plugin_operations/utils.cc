/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "utils.h"
#include <cstring>
#include <sstream>

namespace llm_kernels {
namespace ascend {

atb::SVector<int64_t> GetCopyTensorStride(atb::Dims &tensor_dims) {
  atb::SVector<int64_t> tmp_strides(tensor_dims.dimNum, 1);
  for (int64_t i = static_cast<int64_t>(tensor_dims.dimNum) - 2; i >= 0; i--) {
    tmp_strides[i] = tensor_dims.dims[i + 1] * tmp_strides[i + 1];
  }
  return tmp_strides;
}

atb::Tensor SqueezeBatchSeq(atb::Tensor atb_tensor) {
  if (atb_tensor.desc.shape.dimNum == DIM3) {
    atb_tensor.desc.shape.dimNum = DIM2;
    atb_tensor.desc.shape.dims[DIM0] = atb_tensor.desc.shape.dims[DIM0] * atb_tensor.desc.shape.dims[DIM1];
    atb_tensor.desc.shape.dims[DIM1] = atb_tensor.desc.shape.dims[DIM2];
  }
  return atb_tensor;
}

}  // namespace ascend
}  // namespace llm_kernels
