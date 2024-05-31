/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>

namespace llm_kernels {
namespace ascend {

// The data type definitions.
enum PermuteDataType { FLOAT16 = 0, FLOAT32 = 1 };

// The permute tiling define, support at most 6 dimensions.
struct PermuteTilingData {
  // The input dims.
  uint32_t dim0;
  uint32_t dim1;
  uint32_t dim2;
  uint32_t dim3;
  uint32_t dim4;
  uint32_t dim5;

  // The input strides.
  uint32_t stride0;
  uint32_t stride1;
  uint32_t stride2;
  uint32_t stride3;
  uint32_t stride4;
  uint32_t stride5;

  // The new dim order.
  uint32_t new_idx0;
  uint32_t new_idx1;
  uint32_t new_idx2;
  uint32_t new_idx3;
  uint32_t new_idx4;
  uint32_t new_idx5;

  // The strides for new tensor.
  uint32_t new_stride0;
  uint32_t new_stride1;
  uint32_t new_stride2;
  uint32_t new_stride3;
  uint32_t new_stride4;
  uint32_t new_stride5;

  // The tiling block length and total length, num of elments.
  uint32_t block_length;
  uint32_t total_length;

  // The used aicore number.
  uint32_t used_core_num;

  // The tiling key, used to specify data type.
  uint32_t tiling_key;
};

}  // namespace ascend
}  // namespace llm_kernels
