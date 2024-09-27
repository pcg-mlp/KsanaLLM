/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/cache_manager/direct_cache_manager.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include "ksana_llm/runtime/request_state.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

DirectCacheManager::DirectCacheManager(const CacheManagerConfig& cache_manager_config)
    : BaseCacheManager<DirectCachedBlock, DirectCachedRequest>(cache_manager_config) {
  cache_manager_config_ = cache_manager_config;
}

DirectCacheManager::~DirectCacheManager() {}

void DirectCacheManager::InitializeCachedBlocks() {
  size_t total_device_block_num = GetBlockManager()->GetDeviceFreeBlockNumber();

  for (size_t i = 0; i < total_device_block_num; ++i) {
    DirectCachedBlock* cached_block = CreateCachedBlock(i);

    // allocate memory block on every device.
    for (size_t j = 0; j < cache_manager_config_.tensor_para_size; ++j) {
      std::vector<int> blocks;
      GetBlockManager()->SetDeviceId(j);
      GetBlockManager()->AllocateBlocks(1, blocks);
      cached_block->memory_block_ids[j] = blocks[0];
    }
    free_cached_blocks_.push(cached_block);
  }
  KLLM_LOG_DEBUG << "DirectCacheManager initialized, device num:" << cache_manager_config_.tensor_para_size
                 << ", device block num:" << free_cached_blocks_.size()
                 << ", host block num:" << GetBlockManager()->GetHostFreeBlockNumber();
}

size_t DirectCacheManager::GetFutureBlockNumber() {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::GetFutureBlockNumber();
}

size_t DirectCacheManager::GetUsableBlockNumber() { return free_cached_blocks_.size(); }

size_t DirectCacheManager::GetRequestUsableBlockNumber(int64_t req_id) { return free_cached_blocks_.size(); }

size_t DirectCacheManager::GetHostFreeBlockNumber() {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::GetHostFreeBlockNumber();
}

size_t DirectCacheManager::GetRequestStepBlockNumber(int64_t req_id) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_THROW(FormatStr("Get step block number for req %d error, req not exist.", req_id));
  }
  return (it->second->output_token_num % cache_manager_config_.block_token_num == 0) ? 1 : 0;
}

Status DirectCacheManager::GetRequestPrefixBlockNumber(int64_t req_id, const std::vector<int>& input_token_ids,
                                                       size_t& shared_block_num, size_t& unique_block_num,
                                                       size_t& shared_token_num) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    CreateCachedRequest(req_id);
  }

  shared_token_num = 0;
  shared_block_num = 0;
  size_t block_token_num = cache_manager_config_.block_token_num;
  unique_block_num = (input_token_ids.size() + block_token_num) / block_token_num;
  return Status();
}

DirectCachedBlock* DirectCacheManager::CreateEmptyCachedBlock() {
  DirectCachedBlock* cached_block = new DirectCachedBlock();
  cached_block->memory_block_ids.resize(cache_manager_config_.tensor_para_size);

  return cached_block;
}

DirectCachedBlock* DirectCacheManager::CreateCachedBlock(size_t block_id) {
  DirectCachedBlock* cached_block = CreateEmptyCachedBlock();
  cached_block->block_id = block_id;

  return cached_block;
}

Status DirectCacheManager::AllocateRequestBlocks(int64_t req_id, size_t block_num,
                                                 std::vector<std::vector<int>>& req_block_ids) {
  if (block_num > free_cached_blocks_.size()) {
    return Status(RET_OUT_OF_MEMORY,
                  FormatStr("Allocate %d blocks for req %d error, no more free blocks.", block_num, req_id));
  }

  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME, FormatStr("Allocate block for req %d error, req not exist.", req_id));
  }

  DirectCachedRequest* cached_request = it->second.get();

  // Try to allocate from free list.
  for (size_t i = 0; i < block_num; ++i) {
    DirectCachedBlock* cached_block = free_cached_blocks_.front();
    free_cached_blocks_.pop();

    // Fill memory block ids,
    for (size_t j = 0; j < cache_manager_config_.tensor_para_size; ++j) {
      req_block_ids[j].push_back(cached_block->memory_block_ids[j]);
    }

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

Status DirectCacheManager::UpdateRequestTokens(int64_t req_id, const std::vector<int>& token_ids,
                                               std::vector<std::vector<int>>& req_block_ids) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME, FormatStr("Update request token error, req %d is not found.", req_id));
  }

  it->second->output_token_num = token_ids.size();
  return Status();
}

Status DirectCacheManager::GetRequestFreeableBlockNum(int64_t req_id, size_t& block_num) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME, FormatStr("Get freeable block num of req error, req %d is not exist.", req_id));
  }

  block_num = it->second->cached_blocks.size();
  return Status();
}

Status DirectCacheManager::GetRequestNeededBlockNum(int64_t req_id, size_t& block_num) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME, FormatStr("Get needed block num of req error, req %d is not exist.", req_id));
  }

  block_num = it->second->cached_blocks.size();

  // Make sure there is enough blocks for next step.
  block_num += GetRequestStepBlockNumber(req_id);

  return Status();
}

Status DirectCacheManager::SwapoutRequestAsync(int64_t req_id, size_t& swapped_block_num, size_t& free_block_num) {
  {
    if (!swapin_task_queue_.empty() || !swapin_cached_block_buffer_.empty() || !finish_swapin_request_.empty()) {
      return Status(RET_RUNTIME, FormatStr("Cannot swapout req %d, some swapin jobs is in progress.", req_id));
    }
  }

  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME, FormatStr("Swapout req %d error, req is not exist.", req_id));
  }

  // Note: here must from tail to head. If swap out failed, do not change anything.
  std::vector<DirectCachedBlock*> dev_swapout_blocks;
  for (auto it2 = it->second->cached_blocks.rbegin(); it2 != it->second->cached_blocks.rend(); ++it2) {
    DirectCachedBlock* cb = *it2;
    dev_swapout_blocks.push_back(cb);
  }

  swapped_block_num = dev_swapout_blocks.size();
  if (GetBlockManager()->GetHostFreeBlockNumber() < cache_manager_config_.tensor_para_size * swapped_block_num) {
    return Status(RET_OUT_OF_MEMORY, FormatStr("Swap out req %d error, no more host blocks, needed: %d, free: %d.",
                                               req_id, cache_manager_config_.tensor_para_size * swapped_block_num,
                                               GetBlockManager()->GetHostFreeBlockNumber()));
  }

  std::vector<DirectCachedBlock*> host_swapout_blocks;
  for (size_t i = 0; i < dev_swapout_blocks.size(); ++i) {
    DirectCachedBlock* cached_block = CreateEmptyCachedBlock();
    GetBlockManager()->AllocateHostBlocks(cache_manager_config_.tensor_para_size, cached_block->memory_block_ids);
    host_swapout_blocks.push_back(cached_block);

    // Append new cached block to buffer list.
    swapout_cached_block_buffer_[req_id].push_back(cached_block);
  }

  swapout_task_queue_[req_id] = threadpool_->Submit([=] {
    for (size_t i = 0; i < dev_swapout_blocks.size(); ++i) {
      SwapoutCachedBlock(dev_swapout_blocks[i], host_swapout_blocks[i]);
    }
  });

  return Status();
}

Status DirectCacheManager::SwapinRequestAsync(int64_t req_id, size_t& block_num,
                                              std::vector<std::vector<int>>& req_block_ids) {
  if (!swapout_task_queue_.empty() || !swapout_cached_block_buffer_.empty() || !finish_swapout_request_.empty()) {
    return Status(RET_RUNTIME, FormatStr("Swap in req %d error, some swapout jobs is in progress.", req_id));
  }

  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME, FormatStr("Swap in req %d error, req is not exist.", req_id));
  }

  std::vector<DirectCachedBlock*> swapin_host_blocks;
  for (DirectCachedBlock* cb : it->second->cached_blocks) {
    swapin_host_blocks.push_back(cb);
  }

  // Allocate block for next step.
  size_t step_block_num = GetRequestStepBlockNumber(req_id);

  // Check whether enough memory exist, do not change anything swapin failed.
  size_t swapin_block_num = swapin_host_blocks.size() + step_block_num;
  if (free_cached_blocks_.size() < swapin_block_num) {
    return Status(RET_OUT_OF_MEMORY, FormatStr("Swap in req %d error, No more free blocks.", req_id));
  }
  block_num = swapin_host_blocks.size();

  std::vector<DirectCachedBlock*> swapin_dev_blocks;
  for (size_t i = 0; i < swapin_host_blocks.size(); ++i) {
    // Pick a empty cached block, replace it.
    DirectCachedBlock* cached_block = free_cached_blocks_.front();
    free_cached_blocks_.pop();
    swapin_dev_blocks.push_back(cached_block);

    // Append to buffer list.
    swapin_cached_block_buffer_[req_id].push_back(swapin_host_blocks[i]);
  }

  // Allocate next step block.
  for (size_t i = 0; i < step_block_num; ++i) {
    DirectCachedBlock* cached_block = free_cached_blocks_.front();
    free_cached_blocks_.pop();

    // Fill memory block ids,
    for (size_t j = 0; j < cache_manager_config_.tensor_para_size; ++j) {
      req_block_ids[j].push_back(cached_block->memory_block_ids[j]);
    }

    // Not trace request info before a block is merge to tree.
    it->second->cached_blocks.push_back(cached_block);
  }

  swapin_task_queue_[req_id] = threadpool_->Submit([=] {
    for (size_t i = 0; i < swapin_host_blocks.size(); ++i) {
      SwapinCachedBlock(swapin_dev_blocks[i], swapin_host_blocks[i]);
    }
  });

  return Status();
}

Status DirectCacheManager::WaitSwapoutRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking) {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::WaitSwapoutRequests(req_ids, left_req_num, blocking);
}

Status DirectCacheManager::WaitSwapinRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking) {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::WaitSwapinRequests(req_ids, left_req_num, blocking);
}

Status DirectCacheManager::MergeSwapoutRequest(int64_t req_id) {
  return BaseCacheManager<DirectCachedBlock, DirectCachedRequest>::MergeSwapoutRequest(req_id);
}

Status DirectCacheManager::MergeSwapinRequest(int64_t req_id, std::vector<std::vector<int>>& req_block_ids) {
  std::vector<DirectCachedBlock*> swapin_blocks;

  swapin_blocks.swap(swapin_cached_block_buffer_[req_id]);
  swapin_cached_block_buffer_.erase(req_id);

  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME, FormatStr("Merge swapped req %d error, req is not exist.", req_id));
  }

  // Note: must from begin to end.
  for (size_t i = 0; i < it->second->cached_blocks.size(); ++i) {
    // Merge every node to the tree.
    DirectCachedBlock* cb = it->second->cached_blocks[i];

    // Update internal memory block ids of unfilled block.
    for (size_t j = 0; j < cache_manager_config_.tensor_para_size; ++j) {
      req_block_ids[j][i] = cb->memory_block_ids[j];
    }
  }

  // Remove from finished queue.
  auto it2 = std::find(finish_swapin_request_.begin(), finish_swapin_request_.end(), req_id);
  if (it2 != finish_swapin_request_.end()) {
    finish_swapin_request_.erase(it2);
  }

  return Status();
}

void DirectCacheManager::DestroySwapedRequest(int64_t req_id) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return;
  }

  DirectCachedRequest* cached_request = it->second.get();
  for (DirectCachedBlock* cb : cached_request->cached_blocks) {
    GetBlockManager()->FreeHostBlocks(cb->memory_block_ids);
    delete cb;
  }

  cached_requests_.erase(it);
}

}  // namespace ksana_llm
