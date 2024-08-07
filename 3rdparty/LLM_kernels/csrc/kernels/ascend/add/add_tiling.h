/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

namespace llm_kernels {
namespace ascend {

constexpr uint32_t ADD_TILE_NUM = 1;

struct AddTilingConfig {
  uint32_t total_elem_num = 0;
  uint32_t block_elem_num = 0;
  uint32_t tile_num = 1;
  // TODO(karlluo): maybe support alpha
  float alpha = 1.0f;
};

}  // namespace ascend
}  // namespace llm_kernels