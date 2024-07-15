/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cstdint>

namespace llm_kernels {
namespace ascend {

struct AssembleLastTokenTiling {
  uint32_t batch_size;
  uint32_t hidden_units_num;
  uint32_t tile_num;
};

}  // namespace ascend
}  // namespace llm_kernels