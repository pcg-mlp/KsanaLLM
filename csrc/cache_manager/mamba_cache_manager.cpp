/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/
#include "cache_manager/mamba_cache_manager.h"

#include <cinttypes>
#include <vector>

#include "utils/string_utils.h"

namespace ksana_llm {

MambaCacheManager::MambaCacheManager(std::shared_ptr<BlockAllocatorGroupInterface> mamba_block_allocator_group)
    : mamba_block_allocator_group_(mamba_block_allocator_group) {}

Status MambaCacheManager::AllocateRequestMambaBlock(int64_t req_id, int& mamba_kv_cache_block_id) {
  mamba_kv_cache_block_id = -1;
  auto it = req_to_mamba_block_.find(req_id);
  if (it != req_to_mamba_block_.end()) {
    if (it->second->owner_req_id == req_id) {
      mamba_kv_cache_block_id = it->second->memory_block_id;
      return Status();
    }
    // req_id 复用残留的旧条目：回收物理块后重建，避免请求永久卡死。
    FreePhysicalBlock(it->second->memory_block_id);
    req_to_mamba_block_.erase(it);
  }

  int new_block_id = -1;
  Status status = AllocatePhysicalBlock(new_block_id);
  if (!status.OK()) {
    return status;
  }

  auto cached_block = std::make_unique<MambaCachedBlock>();
  cached_block->memory_block_id = new_block_id;
  cached_block->owner_req_id = req_id;
  req_to_mamba_block_[req_id] = std::move(cached_block);
  mamba_kv_cache_block_id = new_block_id;
  return Status();
}

Status MambaCacheManager::FreeRequestMambaBlock(int64_t req_id) {
  auto it = req_to_mamba_block_.find(req_id);
  if (it == req_to_mamba_block_.end()) {
    return Status();
  }

  auto& cached_block = it->second;
  if (cached_block->owner_req_id != req_id) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Invalid working Mamba block for req %" PRId64 ".", req_id));
  }

  const int mamba_kv_cache_block_id = cached_block->memory_block_id;
  Status status = FreePhysicalBlock(mamba_kv_cache_block_id);
  if (status.OK()) {
    req_to_mamba_block_.erase(it);
  }
  return status;
}

int MambaCacheManager::GetRequestMambaBlockId(int64_t req_id) const {
  auto it = req_to_mamba_block_.find(req_id);
  return it != req_to_mamba_block_.end() ? it->second->memory_block_id : -1;
}

Status MambaCacheManager::AllocatePhysicalBlock(int& mamba_kv_cache_block_id) {
  mamba_kv_cache_block_id = -1;
  auto allocator = mamba_block_allocator_group_->GetDeviceBlockAllocator(0);
  std::vector<int> allocated_block_ids;
  Status status = allocator->AllocateBlocks(1, allocated_block_ids);
  if (!status.OK()) {
    return status;
  }
  if (allocated_block_ids.empty()) {
    return Status(RET_RUNTIME_FAILED, "Allocator returned empty block list.");
  }

  // Slot 0 is the fixed NULL sentinel (kernel contract); usable slots are [1, N].
  mamba_kv_cache_block_id = static_cast<int>(allocated_block_ids[0]) + 1;
  return Status();
}

Status MambaCacheManager::FreePhysicalBlock(int mamba_kv_cache_block_id) {
  if (mamba_kv_cache_block_id <= 0) {
    return Status(RET_INVALID_ARGUMENT, "Invalid Mamba block id.");
  }
  return mamba_block_allocator_group_->GetDeviceBlockAllocator(0)->FreeBlocks({mamba_kv_cache_block_id - 1});
}
}  // namespace ksana_llm
