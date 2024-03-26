/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <cstdint>

#include "ksana_llm/utils/memory_device.h"

namespace ksana_llm {

uint32_t GetDeviceNumber(MemoryDevice device_type = MemoryDevice::MEMORY_GPU);

} // namespace ksana_llm