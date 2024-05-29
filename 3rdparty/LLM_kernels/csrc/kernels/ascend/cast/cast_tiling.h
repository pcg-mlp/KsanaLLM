/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <stdint.h>

namespace llm_kernels {
namespace ascend {

constexpr uint32_t CAST_TILE_NUM = 2;

struct CastTilingConfig {
  uint32_t total_elem_num = 0;
  uint32_t block_elem_num = 0;
  uint32_t tile_num = 2;
};

}  // namespace ascend
}  // namespace llm_kernels