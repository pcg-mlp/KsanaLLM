/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <stdint.h>

namespace llm_kernels {
namespace ascend {

// NOTE(karlluo): for global memeory write retrict, each block handle 64B ~ 16 * argmax value(uint32_t)
// ref:
// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha003/apiref/opdevgapi/atlasascendc_api_07_0169.html
// if output less than 16 argmax value, we handle it to one block
constexpr uint32_t ARGMAX_SINGLE_BLOCK_CAPACITY = 16;

struct ArgmaxConfigTiling {
  uint32_t batch_size = 0;
  uint32_t vocab_size = 0;
  uint32_t tile_num = 2;
  uint32_t block_handle_num = 0;
};

}  // namespace ascend
}  // namespace llm_kernels