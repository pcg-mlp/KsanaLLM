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
                                       std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group)
    : BaseCacheManager<DirectCachedBlock, DirectCachedRequest>(cache_manager_config, block_allocator_group) {
  cache_manager_config_ = cache_manager_config;
}

DirectCacheManager::~DirectCacheManager() { KLLM_LOG_DEBUG << "DirectCacheManager destroyed."; }

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

size_t DirectCacheManager::GetFutureFreeBlockNumber() {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::GetFutureFreeBlockNumber();
}

size_t DirectCacheManager::GetUsableBlockNumber() { return free_cached_blocks_.size(); }

size_t DirectCacheManager::GetRequestUsableBlockNumber(int64_t req_id) { return free_cached_blocks_.size(); }

size_t DirectCacheManager::GetHostFreeBlockNumber() {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::GetHostFreeBlockNumber();
}

size_t DirectCacheManager::GetUsedBlockNumber() {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::GetDeviceUsedBlockNumber() - GetUsableBlockNumber();
}

size_t DirectCacheManager::GetRequestStepBlockNumber(int64_t req_id, size_t input_token_lens) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_THROW(FormatStr("Get step block num of req error, req %d is not exist.", req_id));
  }
  const size_t block_token_num = cache_manager_config_.block_token_num;
  const size_t total_require_block_num = (input_token_lens + block_token_num) / block_token_num;
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

DirectCachedBlock* DirectCacheManager::CreateEmptyCachedBlock() {
  DirectCachedBlock* cached_block = new DirectCachedBlock();

  return cached_block;
}

DirectCachedBlock* DirectCacheManager::CreateCachedBlock(size_t block_id) {
  DirectCachedBlock* cached_block = CreateEmptyCachedBlock();
  cached_block->block_id = block_id;

  return cached_block;
}

Status DirectCacheManager::AllocateRequestBlocks(int64_t req_id, size_t block_num, std::vector<int>& req_block_ids) {
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

  // Try to allocate from free list.
  for (size_t i = 0; i < block_num; ++i) {
    DirectCachedBlock* cached_block = free_cached_blocks_.front();
    free_cached_blocks_.pop();

    // Fill memory block ids.
    req_block_ids.push_back(cached_block->memory_block_id);

    // record block ids.
    cached_request->cached_blocks.push_back(cached_block);
  }

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
  cached_requests_.erase(it);
}

Status DirectCacheManager::UpdateRequestTokens(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                                               size_t shareable_kvcache_token_num, std::vector<int>& req_block_ids,
                                               bool& block_merged) {
  block_merged = false;
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_LOG_ERROR << "Error in UpdateRequestTokens for req id " << req_id << " is not exist.";
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("Error in UpdateRequestTokens for req id %d error, is not exist.", req_id));
  }
  it->second->kvcached_token_num = kvcached_token_ids.size();

  return Status();
}

Status DirectCacheManager::GetRequestFreeableBlockNum(int64_t req_id, size_t& block_num) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_LOG_ERROR << "Error in GetRequestFreeableBlockNum for req id " << req_id << " is not exist.";
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("Error in GetRequestFreeableBlockNum for req id %d error, is not exist.", req_id));
  }

  block_num = it->second->cached_blocks.size();
  return Status();
}

Status DirectCacheManager::GetRequestNeededBlockNumForOneNextToken(int64_t req_id, size_t& block_num) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_LOG_ERROR << "Error in GetRequestNeededBlockNumForOneNextToken for req id " << req_id << " is not exist.";
    return Status(
        RET_RUNTIME_FAILED,
        FormatStr("Error in GetRequestNeededBlockNumForOneNextToken for req id %d error, is not exist.", req_id));
  }

  block_num = it->second->cached_blocks.size();

  // Make sure there is enough blocks for next step.
  block_num += GetRequestStepBlockNumberForOneNextToken(req_id);

  return Status();
}

void DirectCacheManager::SetSwapCallbackFunc(AsyncSwapCallbackFunc callback) { swap_callback_func_ = callback; }

Status DirectCacheManager::SwapoutRequestAsync(int64_t req_id, size_t& swapped_block_num, size_t& free_block_num,
                                               std::vector<int>& swapped_memory_block_ids,
                                               std::vector<int>& host_block_ids) {
  {
    if (!swapin_task_queue_.empty() || !swapin_cached_block_buffer_.empty() || !finish_swapin_request_.empty()) {
      return Status(RET_RUNTIME_FAILED, FormatStr("Cannot swapout req %d, some swapin jobs is in progress.", req_id));
    }
  }

  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_LOG_ERROR << "Error in SwapoutRequestAsync for req id " << req_id << " is not exist.";
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("Error in SwapoutRequestAsync for req id %d error, is not exist.", req_id));
  }

  // Note: here must from tail to head. If swap out failed, do not change anything.
  std::vector<DirectCachedBlock*> dev_swapout_blocks;
  for (auto it2 = it->second->cached_blocks.rbegin(); it2 != it->second->cached_blocks.rend(); ++it2) {
    DirectCachedBlock* cb = *it2;
    dev_swapout_blocks.push_back(cb);

    // Assume every device have same memory block id.
    swapped_memory_block_ids.push_back(cb->memory_block_id);
  }

  free_block_num = 0;
  swapped_block_num = dev_swapout_blocks.size();
  if (block_allocator_group_->GetHostBlockAllocator()->GetFreeBlockNumber() < swapped_block_num) {
    return Status(RET_OUT_OF_DEVICE_MEMORY,
                  FormatStr("Swap out req %d error, no more host blocks, needed: %d, free: %d.", req_id,
                            swapped_block_num, block_allocator_group_->GetHostBlockAllocator()->GetFreeBlockNumber()));
  }

  host_block_ids.clear();
  host_block_ids.reserve(dev_swapout_blocks.size());
  std::vector<DirectCachedBlock*> host_swapout_blocks;
  for (size_t i = 0; i < dev_swapout_blocks.size(); ++i) {
    DirectCachedBlock* cached_block = CreateEmptyCachedBlock();
    std::vector<int> host_blocks;
    block_allocator_group_->GetHostBlockAllocator()->AllocateBlocks(1, host_blocks);
    cached_block->memory_block_id = host_blocks[0];
    host_swapout_blocks.push_back(cached_block);
    host_block_ids.push_back(host_blocks[0]);

    // Append new cached block to buffer list.
    swapout_cached_block_buffer_[req_id].push_back(cached_block);
  }

  SubmitSwapoutAsyncTask(req_id, swapped_block_num, std::move(dev_swapout_blocks), std::move(host_swapout_blocks),
                         swap_callback_func_);

  return Status();
}

Status DirectCacheManager::SwapinRequestAsync(int64_t req_id, size_t& block_num, std::vector<int>& req_block_ids,
                                              std::vector<int>& swapped_memory_block_ids,
                                              std::vector<int>& host_block_ids) {
  if (!swapout_task_queue_.empty() || !swapout_cached_block_buffer_.empty() || !finish_swapout_request_.empty()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Swap in req %d error, some swapout jobs is in progress.", req_id));
  }

  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_LOG_ERROR << "Error in SwapinRequestAsync for req id " << req_id << " is not exist.";
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("Error in SwapinRequestAsync for req id %d error, is not exist.", req_id));
  }

  host_block_ids.clear();
  host_block_ids.reserve(it->second->cached_blocks.size());
  std::vector<DirectCachedBlock*> swapin_host_blocks;
  for (DirectCachedBlock* cb : it->second->cached_blocks) {
    swapin_host_blocks.push_back(cb);
    host_block_ids.push_back(cb->memory_block_id);
  }

  // Allocate block for next step.
  size_t step_block_num = GetRequestStepBlockNumberForOneNextToken(req_id);

  // Check whether enough memory exist, do not change anything swapin failed.
  size_t swapin_block_num = swapin_host_blocks.size() + step_block_num;
  if (free_cached_blocks_.size() < swapin_block_num) {
    return Status(RET_OUT_OF_DEVICE_MEMORY, FormatStr("Swap in req %d error, No more free blocks.", req_id));
  }
  block_num = swapin_host_blocks.size();

  std::vector<DirectCachedBlock*> swapin_dev_blocks;
  for (size_t i = 0; i < swapin_host_blocks.size(); ++i) {
    // Pick a empty cached block, replace it.
    DirectCachedBlock* cached_block = free_cached_blocks_.front();
    free_cached_blocks_.pop();
    swapin_dev_blocks.push_back(cached_block);

    // Assume every device have same memory block id.
    swapped_memory_block_ids.push_back(cached_block->memory_block_id);

    // Append to buffer list.
    swapin_cached_block_buffer_[req_id].push_back(swapin_host_blocks[i]);
  }

  // Allocate next step block.
  for (size_t i = 0; i < step_block_num; ++i) {
    DirectCachedBlock* cached_block = free_cached_blocks_.front();
    free_cached_blocks_.pop();

    // Fill memory block ids.
    req_block_ids.push_back(cached_block->memory_block_id);

    // Not trace request info before a block is merge to tree.
    it->second->cached_blocks.push_back(cached_block);
  }

  SubmitSwapinAsyncTask(req_id, block_num, std::move(swapin_dev_blocks), std::move(swapin_host_blocks),
                        swap_callback_func_);

  return Status();
}

Status DirectCacheManager::WaitSwapoutRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking) {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::WaitSwapoutRequests(req_ids, left_req_num, blocking);
}

Status DirectCacheManager::WaitSwapinRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking) {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::WaitSwapinRequests(req_ids, left_req_num, blocking);
}

Status DirectCacheManager::WaitSwapoutRequests(const std::unordered_set<int64_t>& waiting_req_ids,
                                               std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking) {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::WaitSwapoutRequests(waiting_req_ids, req_ids,
                                                                                       left_req_num, blocking);
}
Status DirectCacheManager::WaitSwapinRequests(const std::unordered_set<int64_t>& waiting_req_ids,
                                              std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking) {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::WaitSwapinRequests(waiting_req_ids, req_ids,
                                                                                      left_req_num, blocking);
}

Status DirectCacheManager::MergeSwapoutRequest(int64_t req_id) {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::MergeSwapoutRequest(req_id);
}

Status DirectCacheManager::MergeSwapinRequest(int64_t req_id, std::vector<int>& req_block_ids) {
  std::vector<DirectCachedBlock*> swapin_blocks;

  swapin_blocks.swap(swapin_cached_block_buffer_[req_id]);
  swapin_cached_block_buffer_.erase(req_id);

  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_LOG_ERROR << "Error in MergeSwapinRequest for req id " << req_id << " is not exist.";
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("Error in MergeSwapinRequest for req id %d error, is not exist.", req_id));
  }

  // Note: must from begin to end.
  for (size_t i = 0; i < it->second->cached_blocks.size(); ++i) {
    // Merge every node to the tree.
    DirectCachedBlock* cb = it->second->cached_blocks[i];

    // Update internal memory block ids of unfilled block.
    req_block_ids[i] = cb->memory_block_id;
  }

  // Remove from finished queue.
  auto it2 = std::find(finish_swapin_request_.begin(), finish_swapin_request_.end(), req_id);
  if (it2 != finish_swapin_request_.end()) {
    finish_swapin_request_.erase(it2);
  }

  return Status();
}

void DirectCacheManager::DestroySwappedRequest(int64_t req_id) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return;
  }

  DirectCachedRequest* cached_request = it->second.get();
  for (DirectCachedBlock* cb : cached_request->cached_blocks) {
    block_allocator_group_->GetHostBlockAllocator()->FreeBlocks({cb->memory_block_id});
    delete cb;
  }

  cached_requests_.erase(it);
}

bool DirectCacheManager::IsPrefixCachingEnabled() {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::IsPrefixCachingEnabled();
}

}  // namespace ksana_llm
