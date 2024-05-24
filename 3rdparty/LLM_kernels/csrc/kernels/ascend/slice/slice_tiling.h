/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>

namespace llm_kernels {
namespace ascend {

// The tiling struct for slice.
struct SliceTilingData {
  // The start offset.
  uint32_t start;

  // The head size of model.
  uint32_t length;

  // Advance bytes at every step.
  uint32_t step;

  // Repeat times
  uint32_t times;

  // The block size in every step.
  uint32_t block_size;

  // The block size of tail block in every step.
  uint32_t tail_block_size;

  // The block size of every step.
  uint32_t step_block_num;

  // The used aicore block number.
  uint32_t used_core_num;
};

}  // namespace ascend
}  // namespace llm_kernels
