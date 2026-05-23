/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <cstdint>

namespace llm_kernels {
namespace nvidia {


enum class ForwardMode : int32_t {
  CompressExtend = 0,
  CompressDecode = 1,
  DefaultForward = 2,
};

}  // namespace nvidia
}  // namespace llm_kernels
