/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <future>
#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

template <class CachedBlockType, class CachedRequestType>
class BaseCacheManager {
 public:
  explicit BaseCacheManager(const CacheManagerConfig& cache_manager_config) {
    cache_manager_config_ = cache_manager_config;
    threadpool_ = std::make_shared<ThreadPool>(cache_manager_config_.swap_threadpool_size);
    threadpool_->Start();
  }

  ~BaseCacheManager() { threadpool_->Stop(); }

  // The value is from block mamanger.
  size_t GetHostFreeBlockNumber() { return GetBlockManager()->GetHostFreeBlockNumber(); }

  // Get block number that not usable now, but will be usable in future.
  // That is, the blocks used by swapout, but not merged yet.
  size_t GetFutureBlockNumber() {
    size_t future_block_num = 0;
    for (auto pair : swapout_cached_block_buffer_) {
      future_block_num += pair.second.size();
    }
    return future_block_num;
  }

  // Waiting until at lease on swap out/in task done, return the pending task number.
  Status WaitSwapoutRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking = true) {
    return WaitTaskDone(swapout_task_queue_, finish_swapout_request_, req_ids, left_req_num, blocking);
  }

  Status WaitSwapinRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking = true) {
    return WaitTaskDone(swapin_task_queue_, finish_swapin_request_, req_ids, left_req_num, blocking);
  }

  // Merge the swapped out blocks to free list, no need to get host block ids.
  // The swapout of the request's block must be done before call this.
  Status MergeSwapoutRequest(int64_t req_id) {
    std::vector<CachedBlockType*> swapout_blocks;

    swapout_blocks.swap(swapout_cached_block_buffer_[req_id]);
    swapout_cached_block_buffer_.erase(req_id);

    // Move reusable node to free list.
    for (CachedBlockType* cb : swapout_blocks) {
      free_cached_blocks_.push_back(cb);
    }

    // Remove from finished queue.
    auto it = std::find(finish_swapout_request_.begin(), finish_swapout_request_.end(), req_id);
    if (it != finish_swapout_request_.end()) {
      finish_swapout_request_.erase(it);
    }

    return Status();
  }

 protected:
  // Create a new cached request instance.
  CachedRequestType* CreateCachedRequest(int64_t req_id) {
    // New created request is in waiting state, and have no cached blocks.PrefixCacheManager
    CachedRequestType* cached_request = new CachedRequestType();
    cached_request->req_id = req_id;

    // Append to request list.
    cached_requests_[req_id] = cached_request;

    return cached_request;
  }

  // Wait until at least one request is done.
  Status WaitTaskDone(std::unordered_map<int64_t, std::future<void>>& task_queue,
                      std::vector<int64_t>& finish_task_request, std::vector<int64_t>& req_ids, size_t& left_task_num,
                      bool blocking = true) {
    // Return immediately if no pending task.
    req_ids.clear();
    if (task_queue.empty()) {
      left_task_num = 0;
      return Status();
    }

    for (auto& pair : task_queue) {
      // Fetch all finished task.
      if (pair.second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        pair.second.get();
        req_ids.push_back(pair.first);
        continue;
      }

      // Return immediately if in unblocking mode.
      if (!blocking) {
        break;
      }

      // Waiting the first one if on task done.
      if (req_ids.empty()) {
        try {
          pair.second.get();
        } catch (const std::exception& e) {
          KLLM_LOG_FATAL << "Exception in swapout, info: " << e.what();
        }
        req_ids.push_back(pair.first);
        continue;
      }

      // return immediately if any task done.
      break;
    }

    for (auto id : req_ids) {
      task_queue.erase(id);
      finish_task_request.push_back(id);
    }

    left_task_num = task_queue.size();
    return Status();
  }

  // Swap out/in a cached block.
  Status SwapoutCachedBlock(CachedBlockType* dev_cached_block, CachedBlockType* host_cached_block) {
    // For swapout, every block is process by only one thread, no need to protect.
    std::vector<int> dev_block_ids = dev_cached_block->memory_block_ids;

    // Swap out cached block to host.
    for (size_t i = 0; i < cache_manager_config_.tensor_para_size; ++i) {
      GetBlockManager()->SetDeviceId(i);

      GetBlockManager()->SwapOut(host_cached_block->memory_block_ids[i], dev_cached_block->memory_block_ids[i]);
      dev_cached_block->memory_block_ids[i] = host_cached_block->memory_block_ids[i];
    }
    dev_cached_block->is_device_location = false;

    // Reuse the memory blocks.
    host_cached_block->block_id = dev_cached_block->block_id;
    host_cached_block->memory_block_ids = dev_block_ids;
    host_cached_block->is_device_location = true;

    return Status();
  }

  Status SwapinCachedBlock(CachedBlockType* dev_cached_block, CachedBlockType* host_cached_block) {
    // Check whether current block has been processed by another thread.
    if (host_cached_block->is_device_location) {
      return Status();
    }

    // Swap in cached block to dev.
    for (size_t i = 0; i < cache_manager_config_.tensor_para_size; ++i) {
      GetBlockManager()->SetDeviceId(i);
      GetBlockManager()->SwapIn(dev_cached_block->memory_block_ids[i], host_cached_block->memory_block_ids[i]);

      // Free the host memory.
      GetBlockManager()->FreeHostBlocks({host_cached_block->memory_block_ids[i]});

      host_cached_block->memory_block_ids[i] = dev_cached_block->memory_block_ids[i];
    }
    host_cached_block->block_id = dev_cached_block->block_id;
    host_cached_block->is_device_location = true;

    // Remove usless cached block.
    delete dev_cached_block;

    return Status();
  }

 protected:
  CacheManagerConfig cache_manager_config_;

  // The cached blocks that have no computed data, and could be reused freely.
  std::list<CachedBlockType*> free_cached_blocks_;

  // All requests, the key is request id.
  std::unordered_map<int64_t, CachedRequestType*> cached_requests_;

  // Threadpool used to swap in/out.
  std::shared_ptr<ThreadPool> threadpool_ = nullptr;

  // The swap out/in requests, all blocks have copied.
  std::vector<int64_t> finish_swapout_request_;
  std::vector<int64_t> finish_swapin_request_;

  // The future queue for swap out/in, the key is requst id.
  std::unordered_map<int64_t, std::future<void>> swapout_task_queue_;
  std::unordered_map<int64_t, std::future<void>> swapin_task_queue_;

  // The cached blocks that have been swapped out/in by async task.
  // The key is request id.
  std::unordered_map<int64_t, std::vector<CachedBlockType*>> swapout_cached_block_buffer_;
  std::unordered_map<int64_t, std::vector<CachedBlockType*>> swapin_cached_block_buffer_;
};

}  // namespace ksana_llm
