/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

// All supported device type.
#define DEVICE_TYPE_NVIDIA 0
#define DEVICE_TYPE_ASCEND 1

// Unknown device type.
#define DEVICE_TYPE_UNKNOWN -1

// All supported memory device.
#define MEMORY_DEVICE_HOST 0
#define MEMORY_DEVICE_NVIDIA 1
#define MEMORY_DEVICE_ASCEND 2
#define MEMORY_DEVICE_PINNED 3

// Unknown memory device.
#define MEMORY_UNKNOWN -1

#ifdef ENABLE_CUDA
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_NVIDIA
#  define ACTIVE_MEMORY_DEVICE MEMORY_NVIDIA
#elif defined(ENABLE_ACL)
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_ASCEND
#  define ACTIVE_MEMORY_DEVICE MEMORY_ASCEND
#else
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_UNKNOWN
#  define ACTIVE_MEMORY_DEVICE MEMORY_UNKNOWN
#endif

namespace ksana_llm {

// A dummy class used as a real defined class.
struct DummyClass {};

} // namespace ksana_llm

