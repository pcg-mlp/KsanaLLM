/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/block_manager/ascend_allocator.h"
#include "ksana_llm/block_manager/device_allocator.h"
#include "ksana_llm/block_manager/memory_block.h"
#include "ksana_llm/utils/ascend/acl_utils.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

AscendDeviceAllocator::AscendDeviceAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context,
                                             int device_id)
    : DeviceAllocator(allocator_config, context, device_id) {}

void AscendDeviceAllocator::AllocateMemory(void** memory_ptr, size_t bytes) {
  ACL_CHECK(aclrtMalloc(memory_ptr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
}

void AscendDeviceAllocator::FreeMemory(void* memory_ptr) { ACL_CHECK(aclrtFree(memory_ptr)); }

}  // namespace ksana_llm
