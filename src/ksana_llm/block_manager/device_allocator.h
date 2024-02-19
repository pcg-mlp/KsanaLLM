/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "c10/core/impl/VirtualGuardImpl.h"
#include "ksana_llm/block_manager/base_allocator.h"
#include "ksana_llm/block_manager/memory_block.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The base class of all device allocator.
class DeviceAllocator : public BaseAllocator {
 public:
  DeviceAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context, int device_id);
  virtual ~DeviceAllocator();

 protected:
  // The device index for current allocator.
  int device_id_;
};

}  // namespace ksana_llm
