/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

namespace ksana_llm {

// The memory device.
enum MemoryDevice {
  // CPU
  MEMORY_CPU,

  // CPU with pinned memory.
  MEMORY_CPU_PINNED,

  // NVIDIA GPU
  MEMORY_GPU,

  // HUAWEI Ascend
  MEMORY_ASCEND
};

}  // namespace ksana_llm