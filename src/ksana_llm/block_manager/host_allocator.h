/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/block_manager/base_allocator.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

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

}  // namespace ksana_llm
