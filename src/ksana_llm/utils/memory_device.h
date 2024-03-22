/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/device_types.h"

namespace ksana_llm {

// The memory device.
enum MemoryDevice {
  // CPU
  MEMORY_CPU = MEMORY_DEVICE_HOST,

  // CPU with pinned memory.
  MEMORY_CPU_PINNED = MEMORY_DEVICE_PINNED,

  // NVIDIA GPU
  MEMORY_GPU = MEMORY_DEVICE_NVIDIA,

  // HUAWEI Ascend
  MEMORY_ASCEND = MEMORY_DEVICE_ASCEND
};

}  // namespace ksana_llm
