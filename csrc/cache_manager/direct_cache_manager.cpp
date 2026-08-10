/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "cache_manager/direct_cache_manager.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include "runtime/request_state.h"
#include "utils/memory_utils.h"
#include "utils/ret_code.h"
#include "utils/status.h"
#include "utils/string_utils.h"

namespace ksana_llm {

DirectCacheManager::DirectCacheManager(const CacheManagerConfig& cache_manager_config,
                                       std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group,
                                       std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group,
                                       std::shared_ptr<BlockAllocatorGroupInterface> mamba_block_allocator_group)
    : BaseCacheManager<DirectCachedBlock, DirectCachedRequest>(cache_manager_config, block_allocator_group) {
  cache_manager_config_ = cache_manager_config;

  // Initialize SWACacheManager only when SWA is configured.
  if (swa_block_allocator_group != nullptr && cache_manager_config_.sliding_window_size > 0) {
    const bool is_contiguous_swa_mode = !cache_manager_config_.enable_non_contiguous_swa_mode;
    swa_cache_manager_ =
        std::make_unique<SWACacheManager>(swa_block_allocator_group, cache_manager_config_.sliding_window_size,
                                          cache_manager_config_.block_token_num, is_contiguous_swa_mode);
  }

  if (mamba_block_allocator_group != nullptr) {
    mamba_cache_manager_ = std::make_unique<MambaCacheManager>(mamba_block_allocator_group);
  }
}

void DirectCacheManager::InitializeCachedBlocks() {
  size_t total_device_block_num = block_allocator_group_->GetDeviceBlockAllocator()->GetFreeBlockNumber();

  for (size_t i = 0; i < total_device_block_num; ++i) {
    DirectCachedBlock* cached_block = CreateCachedBlock(i);

    // allocate memory block on device0 and reuse its result for all devices.
    std::vector<int> blocks;
    block_allocator_group_->GetDeviceBlockAllocator(0)->AllocateBlocks(1, blocks);
    cached_block->memory_block_id = blocks[0];
    free_cached_blocks_.push(cached_block);
  }
  KLLM_LOG_DEBUG << "DirectCacheManager initialized, device num:" << block_device_num_
                 << ", device block num:" << free_cached_blocks_.size()
                 << ", host block num:" << block_allocator_group_->GetHostBlockAllocator()->GetFreeBlockNumber();
}

std::shared_ptr<BlockAllocatorGroupInterface> DirectCacheManager::GetBlockAllocatorGroup() const {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::GetBlockAllocatorGroup();
}

size_t DirectCacheManager::GetUsableBlockNumber() { return free_cached_blocks_.size(); }

size_t DirectCacheManager::GetRequestUsableBlockNumber(int64_t req_id) { return free_cached_blocks_.size(); }

size_t DirectCacheManager::GetHostFreeBlockNumber() {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::GetHostFreeBlockNumber();
}

size_t DirectCacheManager::GetUsedBlockNumber() {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::GetDeviceUsedBlockNumber() - GetUsableBlockNumber();
}

size_t DirectCacheManager::GetRequestStepBlockNumber(int64_t req_id, size_t input_token_lens, size_t mtp_step_num) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_THROW(FormatStr("Get step block num of req error, req %d is not exist.", req_id));
  }
  const size_t block_token_num = cache_manager_config_.block_token_num;
  // input 和 mtp_step_num 分别按 block 向上取整，保证 MTP lookahead 使用独立 block。
  // hook fast-path 与 backend fallback 都走这个入口，避免 pd_v2 dispatcher 两侧 block 计数不一致。
  const size_t input_blocks = (input_token_lens + block_token_num) / block_token_num;
  const size_t mtp_blocks = (mtp_step_num + block_token_num - 1) / block_token_num;
  const size_t total_require_block_num = input_blocks + mtp_blocks;
  return total_require_block_num <= it->second->cached_blocks.size()
             ? 0
             : total_require_block_num - it->second->cached_blocks.size();
}

size_t DirectCacheManager::GetRequestStepBlockNumberForOneNextToken(int64_t req_id) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_THROW(FormatStr("Get step block number for req %d error, req not exist.", req_id));
  }
  return ((it->second->kvcached_token_num + 1) % cache_manager_config_.block_token_num == 0) ? 1 : 0;
}

Status DirectCacheManager::GetRequestPrefixBlockNumber(int64_t req_id, const std::vector<int>& input_token_ids,
                                                       size_t check_token_num, size_t& shared_block_num,
                                                       size_t& unique_block_num, size_t& shared_token_num) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    CreateCachedRequest(req_id);
  }

  shared_token_num = 0;
  shared_block_num = 0;
  const size_t block_token_num = cache_manager_config_.block_token_num;
  unique_block_num = (check_token_num + block_token_num) / block_token_num;
  return Status();
}

Status DirectCacheManager::GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(
    int64_t req_id, const std::vector<int>& input_token_ids, size_t check_token_num, size_t& fa_shared_block_num,
    size_t& fa_unique_block_num, size_t& fa_shared_token_num, size_t& swa_shared_block_num,
    size_t& swa_unique_block_num, size_t& swa_shared_token_num) {
  // DirectCacheManager has no prefix cache; all shared counts are 0.
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    CreateCachedRequest(req_id);
  }

  fa_shared_block_num = 0;
  fa_shared_token_num = 0;
  swa_shared_block_num = 0;
  swa_shared_token_num = 0;
  const size_t block_token_num = cache_manager_config_.block_token_num;
  fa_unique_block_num = (check_token_num + block_token_num) / block_token_num;
  swa_unique_block_num = fa_unique_block_num;
  return Status();
}

DirectCachedBlock* DirectCacheManager::CreateEmptyCachedBlock() {
  DirectCachedBlock* cached_block = new DirectCachedBlock();

  return cached_block;
}

DirectCachedBlock* DirectCacheManager::CreateCachedBlock(size_t block_id) {
  DirectCachedBlock* cached_block = CreateEmptyCachedBlock();
  cached_block->block_id = block_id;

  return cached_block;
}

Status DirectCacheManager::AllocateRequestBlocks(int64_t req_id, size_t block_num, std::vector<int>& req_block_ids,
                                                 std::vector<int>& swa_req_block_ids,
                                                 std::vector<size_t>& swa_block_idx_offsets,
                                                 int& mamba_kv_cache_block_id) {
  if (block_num > free_cached_blocks_.size()) {
    return Status(RET_OUT_OF_DEVICE_MEMORY,
                  FormatStr("Allocate %d blocks for req %d error, no more free blocks.", block_num, req_id));
  }

  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_LOG_ERROR << "Error in AllocateRequestBlocks for req id " << req_id << " is not exist.";
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("Error in AllocateRequestBlocks for req id %d error, is not exist.", req_id));
  }

  DirectCachedRequest* cached_request = it->second.get();

  // SWA resource check BEFORE FA allocation to avoid any rollback.
  // Compute the SWA positions that should be bound in this round, using the known
  // post-allocation FA count (old_size + block_num) without actually allocating first.
  std::vector<int> positions_to_allocate;
  if (swa_cache_manager_ != nullptr) {
    // swa_right_block = total FA blocks AFTER this allocation round.
    const size_t swa_right_block = cached_request->cached_blocks.size() + block_num;
    const size_t sliding_window_blocks =
        (cache_manager_config_.sliding_window_size + cache_manager_config_.block_token_num - 1) /
            cache_manager_config_.block_token_num +
        1;
    const size_t minimum_swa_block_num = std::min(block_num, sliding_window_blocks);

    if (swa_cache_manager_->GetAllocatableSWACache(req_id, cached_request->cached_blocks.size(), swa_right_block,
                                                   minimum_swa_block_num, /*disable_swa_block_num_adjust*/ false,
                                                   positions_to_allocate,
                                                   cached_request->swa_window_fa_block_ids) < 0) {
      return Status(RET_OUT_OF_DEVICE_MEMORY,
                    FormatStr("GetAllocatableSWACache for req %d failed, no enough SWA blocks.", req_id));
    }
  }

  mamba_kv_cache_block_id = -1;
  if (mamba_cache_manager_ != nullptr) {
    Status status = mamba_cache_manager_->AllocateRequestMambaBlock(req_id, mamba_kv_cache_block_id);
    if (!status.OK()) {
      return Status(status.GetCode(),
                    FormatStr("Allocate mamba working block for req %d failed: %s", req_id,
                              status.GetMessage().c_str()));
    }
  }

  // FA block allocation (SWA availability already confirmed above).
  for (size_t i = 0; i < block_num; ++i) {
    DirectCachedBlock* cached_block = free_cached_blocks_.front();
    free_cached_blocks_.pop();

    // Fill memory block ids.
    req_block_ids.push_back(cached_block->memory_block_id);

    // record block ids.
    cached_request->cached_blocks.push_back(cached_block);
  }

  if (swa_cache_manager_ == nullptr) {
    return Status();
  }

  // Allocate SWA blocks at the queried positions and record them in the window tracking arrays.
  for (int pos : positions_to_allocate) {
    const size_t fa_block_id = cached_request->cached_blocks[pos]->block_id;
    int swa_mem_id = swa_cache_manager_->AllocateSWABlock(req_id, fa_block_id, static_cast<size_t>(pos));
    if (swa_mem_id < 0) {
      return Status(RET_OUT_OF_DEVICE_MEMORY, FormatStr("AllocateSWABlock at pos %d for req %d failed.", pos, req_id));
    }
    cached_request->swa_window_fa_block_ids.push_back(fa_block_id);
    cached_request->swa_window_block_idxs.push_back(static_cast<size_t>(pos));
  }

  // Fill output parameters directly from the per-block index arrays.
  // swa_block_idx_offsets[i] is the FA block global index for swa_req_block_ids[i].
  swa_req_block_ids.clear();
  swa_block_idx_offsets.clear();
  for (size_t i = 0; i < cached_request->swa_window_fa_block_ids.size(); ++i) {
    int mem_id = swa_cache_manager_->GetSWAMemoryBlockId(cached_request->swa_window_fa_block_ids[i]);
    if (mem_id >= 0) {
      swa_req_block_ids.push_back(mem_id);
      swa_block_idx_offsets.push_back(cached_request->swa_window_block_idxs[i]);
    }
  }

  KLLM_LOG_SCHEDULER << "AllocateRequestBlocks, req_id=" << req_id << ", block_num=" << block_num
                     << ", req_block_num=" << cached_request->cached_blocks.size()
                     << ", swa_block_idx_offsets.size=" << swa_block_idx_offsets.size() << ", "
                     << Vector2Str(swa_block_idx_offsets);

  return Status();
}

void DirectCacheManager::DestroyFinishedRequest(int64_t req_id) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_LOG_ERROR << "DestroyFinishedRequest error, req " << req_id << " is not found in cached queue.";
    return;
  }

  DirectCachedRequest* cached_request = it->second.get();

  for (size_t i = 0; i < cached_request->cached_blocks.size(); ++i) {
    free_cached_blocks_.push(cached_request->cached_blocks[i]);
  }

  // Remove blocks from request.
  cached_request->cached_blocks.clear();

  // Release all SWA blocks via SWACacheManager.
  // DirectCacheManager does not support prefix cache reuse, so blocks are freed
  // immediately rather than being marked reclaimable.
  // 注意: AllocateSWABlock 时会把 req_id 加入 block->active_req_ids; 直接调
  // FreeSWABlock 会因检测到非空 active_req_ids 而报 lifecycle warn。先调
  // SetReclaimable 把当前 req 从 active_req_ids erase, FreeSWABlock 即走 silent path。
  if (mamba_cache_manager_ != nullptr) {
    mamba_cache_manager_->FreeRequestMambaBlock(req_id);
  }

  if (swa_cache_manager_ != nullptr) {
    for (size_t fa_block_id : cached_request->swa_window_fa_block_ids) {
      swa_cache_manager_->SetReclaimable(req_id, fa_block_id);
      swa_cache_manager_->FreeSWABlock(fa_block_id);
    }
    cached_request->swa_window_fa_block_ids.clear();
    cached_request->swa_window_block_idxs.clear();
  }

  cached_requests_.erase(it);
}

Status DirectCacheManager::UpdateRequestTokens(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                                               size_t shareable_kvcache_token_num, std::vector<int>& req_block_ids,
                                               bool& block_merged, std::vector<int>& swa_req_block_ids,
                                               std::vector<size_t>& swa_block_idx_offsets) {
  block_merged = false;
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_LOG_ERROR << "Error in UpdateRequestTokens for req id " << req_id << " is not exist.";
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("Error in UpdateRequestTokens for req id %d error, is not exist.", req_id));
  }
  it->second->kvcached_token_num = kvcached_token_ids.size();

  // SWA block sliding: skip if SWACacheManager is not configured.
  if (swa_cache_manager_ == nullptr) {
    return Status();
  }

  DirectCachedRequest* cached_request = it->second.get();
  const size_t block_token_num = cache_manager_config_.block_token_num;

  // Compute the SWA window left boundary block index.
  // The left boundary token is: max(shareable_kvcache_token_num - sliding_window_size + 1, 0).
  // The caller (AllocateRequestBlocks) is responsible for allocating sufficient SWA blocks on the right;
  // UpdateRequestTokens only slides the left boundary to free blocks that have fallen out of the window.
  const size_t sliding_window_size = cache_manager_config_.sliding_window_size;
  size_t swa_left_block = 0;
  if (shareable_kvcache_token_num >= sliding_window_size) {
    const size_t swa_left_token = shareable_kvcache_token_num - sliding_window_size + 1;
    swa_left_block = swa_left_token / block_token_num;
  }

  // Release SWA blocks whose FA block index < swa_left_block.
  // DirectCacheManager does not support prefix cache reuse, so blocks are freed immediately.
  // 先 SetReclaimable 再 FreeSWABlock，避免 lifecycle warn（同 DestroyFinishedRequest）。
  auto& ids = cached_request->swa_window_fa_block_ids;
  auto& idxs = cached_request->swa_window_block_idxs;
  while (!ids.empty() && idxs.front() < swa_left_block) {
    swa_cache_manager_->SetReclaimable(req_id, ids.front());
    swa_cache_manager_->FreeSWABlock(ids.front());
    ids.erase(ids.begin());
    idxs.erase(idxs.begin());
  }

  // Fill output parameters directly from the per-block index arrays.
  // swa_block_idx_offsets[i] is the FA block global index for swa_req_block_ids[i].
  swa_req_block_ids.clear();
  swa_block_idx_offsets.clear();
  for (size_t i = 0; i < ids.size(); ++i) {
    int mem_id = swa_cache_manager_->GetSWAMemoryBlockId(ids[i]);
    if (mem_id >= 0) {
      swa_req_block_ids.push_back(mem_id);
      swa_block_idx_offsets.push_back(idxs[i]);
    }
  }

  return Status();
}

bool DirectCacheManager::IsPrefixCachingEnabled() {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::IsPrefixCachingEnabled();
}

}  // namespace ksana_llm
