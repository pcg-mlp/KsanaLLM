/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <mutex>
#include "numerous_llm/block_manager/device_allocator.h"
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/utils/environment.h"

namespace numerous_llm {

// The device allocator for nvidia card.
// All the method must be thread-safe.
class NvidiaDeviceAllocator : public DeviceAllocator {
 public:
  NvidiaDeviceAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context, int device_id);
  ~NvidiaDeviceAllocator() {}

 private:
  // allocate memory
  virtual void AllocateMemory(void** memory_ptr, size_t bytes) override;

  // Free memory.
  virtual void FreeMemory(void* memory_ptr) override;
};

}  // namespace numerous_llm
