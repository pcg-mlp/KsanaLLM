/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/block_manager/base_allocator.h"
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/utils/environment.h"

namespace numerous_llm {

// The host allocator implement.
// All the method must be thread-safe.
class HostAllocator : public BaseAllocator {
 public:
  HostAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context);
  virtual ~HostAllocator() {}

 private:
  // allocate memory
  virtual void AllocateMemory(void** memory_ptr, size_t bytes) override;

  // Free memory.
  virtual void FreeMemory(void* memory_ptr) override;
};

}  // namespace numerous_llm
