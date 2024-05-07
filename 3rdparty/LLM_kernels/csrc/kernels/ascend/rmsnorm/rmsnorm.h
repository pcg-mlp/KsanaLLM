/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <stdint.h>

namespace llm_kernels {
namespace ascend {

// NOTE(karlluo): same as RmsNormTiling from kernel_tiling/kernel_tiling.h at host
struct RmsNormTilingConfig {
  uint32_t bLength = 0;
  uint32_t sLength = 0;
  uint32_t hLength = 0;
  uint32_t originalHLength = 0;
  uint32_t reciprocalOfHLength = 0;
  uint32_t mainBshLength = 0;
  uint32_t mainBsLength = 0;
  uint32_t mainBsLengthAlign = 0;
  uint32_t loopRound = 0;
  uint32_t inputTailPos = 0;
  uint32_t tailBshLength = 0;
  uint32_t tailBsLength = 0;
};

}  // namespace ascend
}  // namespace llm_kernels