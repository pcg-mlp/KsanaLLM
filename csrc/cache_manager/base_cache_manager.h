/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "cache_manager/cache_manager_interface.h"
#include "configure/environment.h"
#include "runtime/threadpool.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/status.h"
#include "utils/string_utils.h"

namespace ksana_llm {

// Used to describe memory block swappness.
struct RequestMemoryBlockSwappinessTask {
  std::vector<int> device_memory_blocks;

  // Eevery device block need multiple host blocks.
  std::vector<std::vector<int>> host_memory_blocks;
};

template <class CachedBlockType, class CachedRequestType>
class BaseCacheManager {
 public:
  explicit BaseCacheManager(const CacheManagerConfig& cache_manager_config,
                            std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group) {
    cache_manager_config_ = cache_manager_config;
    block_allocator_group_ = block_allocator_group;

    block_device_num_ = block_allocator_group_->GetBlockAllocatorDevices().size();

    // Ensure threadpool size is at least block_device_num_ + 1 to avoid deadlock.
    // SwapoutCachedBlocksByDevice is called from a threadpool task and internally submits
    // block_device_num_ more tasks, so we need at least block_device_num_ + 1 threads.
    size_t min_threadpool_size = block_device_num_ + 1;
    size_t actual_threadpool_size = std::max(cache_manager_config_.swap_threadpool_size, min_threadpool_size);
    threadpool_ = std::make_shared<ThreadPool>(actual_threadpool_size);
    threadpool_->Start();
  }

  ~BaseCacheManager() { threadpool_->Stop(); }

  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup() const { return block_allocator_group_; }

  // The value is from block mamanger.
  size_t GetHostFreeBlockNumber() { return block_allocator_group_->GetHostBlockAllocator()->GetFreeBlockNumber(); }

  size_t GetDeviceUsedBlockNumber() { return block_allocator_group_->GetDeviceBlockAllocator()->GetUsedBlockNumber(); }

  // Get block number that not usable now, but will be usable in future.
  // That is, the blocks used by swapout, but not merged yet.
  size_t GetFutureFreeBlockNumber() {
    size_t future_block_num = 0;
    for (const auto& [req_id, swapout_blocks] : swapout_cached_block_buffer_) {
      future_block_num += swapout_blocks.size();
    }
    return future_block_num;
  }

  // Waiting until at least one swap out/in task done, return the pending task number.
  Status WaitSwapoutRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking = true) {
    std::unordered_set<int64_t> empty_req_ids;
    return WaitTaskDone(swapout_task_queue_, empty_req_ids, finish_swapout_request_, req_ids, left_req_num, blocking);
  }

  Status WaitSwapinRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking = true) {
    std::unordered_set<int64_t> empty_req_ids;
    return WaitTaskDone(swapin_task_queue_, empty_req_ids, finish_swapin_request_, req_ids, left_req_num, blocking);
  }

  // Waiting until at least one swap out/in tasks in waiting_req_ids done, return the pending task number.
  Status WaitSwapoutRequests(const std::unordered_set<int64_t>& waiting_req_ids, std::vector<int64_t>& req_ids,
                             size_t& left_req_num, bool blocking = true) {
    return WaitTaskDone(swapout_task_queue_, waiting_req_ids, finish_swapout_request_, req_ids, left_req_num, blocking);
  }

  Status WaitSwapinRequests(const std::unordered_set<int64_t>& waiting_req_ids, std::vector<int64_t>& req_ids,
                            size_t& left_req_num, bool blocking = true) {
    return WaitTaskDone(swapin_task_queue_, waiting_req_ids, finish_swapin_request_, req_ids, left_req_num, blocking);
  }

  // Merge the swapped out blocks to free list, no need to get host block ids.
  // The swapout of the request's block must be done before call this.
  Status MergeSwapoutRequest(int64_t req_id) {
    std::vector<CachedBlockType*> swapout_blocks;

    swapout_blocks.swap(swapout_cached_block_buffer_[req_id]);
    swapout_cached_block_buffer_.erase(req_id);

    // Move reusable node to free list.
    for (CachedBlockType* cb : swapout_blocks) {
      free_cached_blocks_.push(cb);
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
    auto cached_request = std::make_unique<CachedRequestType>();
    cached_request->req_id = req_id;

    // Get a raw pointer from the unique pointer of CachedRequestType without ownership.
    CachedRequestType* cached_request_ptr = cached_request.get();
    // Append to request list.
    cached_requests_.emplace(req_id, std::move(cached_request));

    return cached_request_ptr;
  }

  // Wait until at least one request in waiting_req_ids is done. If waiting_req_ids is empty, wait for any task.
  Status WaitTaskDone(std::unordered_map<int64_t, std::future<void>>& task_queue,
                      const std::unordered_set<int64_t>& waiting_req_ids, std::vector<int64_t>& finish_task_request,
                      std::vector<int64_t>& req_ids, size_t& left_task_num, bool blocking = true) {
    // Return immediately if no pending task.
    req_ids.clear();
    if (task_queue.empty()) {
      left_task_num = 0;
      if (waiting_req_ids.empty()) {
        req_ids = finish_task_request;
      } else {
        for (auto id : finish_task_request) {
          if (waiting_req_ids.find(id) != waiting_req_ids.end()) {
            req_ids.push_back(id);
          }
        }
      }
      return Status();
    }

    for (auto& pair : task_queue) {
      if (!waiting_req_ids.empty() && waiting_req_ids.find(pair.first) == waiting_req_ids.end()) {
        continue;
      }

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

  void SubmitSwapoutAsyncTask(int64_t req_id, size_t swapped_block_num,
                              std::vector<CachedBlockType*> dev_swapout_blocks,
                              std::vector<CachedBlockType*> host_swapout_blocks, AsyncSwapCallbackFunc callback) {
    swapout_task_queue_[req_id] = threadpool_->Submit([this, req_id, swapped_block_num, callback,
                                                       dev_swapout_blocks = std::move(dev_swapout_blocks),
                                                       host_swapout_blocks = std::move(host_swapout_blocks)]() mutable {
      Status status;
      try {
        status = SwapoutCachedBlocksByDevice(dev_swapout_blocks, host_swapout_blocks);
      } catch (const std::exception& e) {
        status = Status(RET_RUNTIME_FAILED, FormatStr("Swapout async req %d exception: %s", req_id, e.what()));
      }

      if (callback != nullptr) {
        callback(req_id, true, swapped_block_num, status);
      }
    });
  }

  void SubmitSwapinAsyncTask(int64_t req_id, size_t swapped_block_num, std::vector<CachedBlockType*> swapin_dev_blocks,
                             std::vector<CachedBlockType*> swapin_host_blocks, AsyncSwapCallbackFunc callback) {
    swapin_task_queue_[req_id] = threadpool_->Submit([this, req_id, swapped_block_num, callback,
                                                      swapin_dev_blocks = std::move(swapin_dev_blocks),
                                                      swapin_host_blocks = std::move(swapin_host_blocks)]() mutable {
      Status status;
      try {
        status = SwapinCachedBlocksByDevice(swapin_dev_blocks, swapin_host_blocks);
      } catch (const std::exception& e) {
        status = Status(RET_RUNTIME_FAILED, FormatStr("Swapin async req %d exception: %s", req_id, e.what()));
      }

      if (callback != nullptr) {
        callback(req_id, false, swapped_block_num, status);
      }
    });
  }

  // Swap out cached blocks for a specific device.
  // The input dev_cached_blocks and host_cached_blocks are organized as [block_index].
  // This method only processes blocks for the specified device_id, calling BatchSwapOut.
  Status SwapoutCachedBlockByDevice(size_t device_id, std::vector<CachedBlockType*>& dev_cached_blocks,
                                    std::vector<CachedBlockType*>& host_cached_blocks) {
    if (dev_cached_blocks.size() != host_cached_blocks.size()) {
      return Status(RET_RUNTIME_FAILED, "SwapoutCachedBlockByDevice: dev and host block count mismatch.");
    }

    if (device_id >= block_device_num_) {
      return Status(RET_RUNTIME_FAILED,
                    FormatStr("SwapoutCachedBlockByDevice: device_id %d exceeds block_device_num %d.", device_id,
                              block_device_num_));
    }

    size_t block_count = dev_cached_blocks.size();
    if (block_count == 0) {
      return Status();
    }

    // Collect device and host block ids for the specified device.
    std::vector<int> device_block_ids_for_device;
    std::vector<int> host_block_ids_for_device;
    device_block_ids_for_device.reserve(block_count);
    host_block_ids_for_device.reserve(block_count);

    for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
      device_block_ids_for_device.push_back(dev_cached_blocks[block_idx]->memory_block_id);
      host_block_ids_for_device.push_back(host_cached_blocks[block_idx]->memory_block_id);
    }

    // Batch swap out for the specified device.
    Status status =
        block_allocator_group_->BatchSwapOut(device_id, host_block_ids_for_device, device_block_ids_for_device);
    if (!status.OK()) {
      return status;
    }

    return Status();
  }

  // Swap out all cached blocks by iterating over devices concurrently.
  // This method uses threadpool to parallelize SwapoutCachedBlockByDevice for each device.
  Status SwapoutCachedBlocksByDevice(std::vector<CachedBlockType*>& dev_cached_blocks,
                                     std::vector<CachedBlockType*>& host_cached_blocks) {
    if (dev_cached_blocks.size() != host_cached_blocks.size()) {
      return Status(RET_RUNTIME_FAILED, "SwapoutCachedBlocksByDevice: dev and host block count mismatch.");
    }

    size_t block_count = dev_cached_blocks.size();
    if (block_count == 0) {
      return Status();
    }

    // Save original device block ids before swap for later reuse.
    std::vector<int> original_dev_block_ids(block_count);
    for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
      original_dev_block_ids[block_idx] = dev_cached_blocks[block_idx]->memory_block_id;
    }

    // Submit swap tasks for each device concurrently.
    std::vector<std::future<Status>> futures;
    futures.reserve(block_device_num_);
    for (size_t device_id = 0; device_id < block_device_num_; ++device_id) {
      futures.push_back(threadpool_->Submit([this, device_id, &dev_cached_blocks, &host_cached_blocks]() -> Status {
        return SwapoutCachedBlockByDevice(device_id, dev_cached_blocks, host_cached_blocks);
      }));
    }

    // Wait for all device swap tasks to complete and check status.
    Status final_status;
    for (size_t device_id = 0; device_id < block_device_num_; ++device_id) {
      try {
        Status status = futures[device_id].get();
        if (!status.OK() && final_status.OK()) {
          final_status = status;
        }
      } catch (const std::exception& e) {
        KLLM_LOG_FATAL << "Exception in SwapoutCachedBlocksByDevice for device " << device_id << ": " << e.what();
        if (final_status.OK()) {
          final_status =
              Status(RET_RUNTIME_FAILED, FormatStr("Exception in swap for device %d: %s", device_id, e.what()));
        }
      }
    }

    if (!final_status.OK()) {
      return final_status;
    }

    // Finalize: update is_device_location and reuse the memory blocks.
    for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
      dev_cached_blocks[block_idx]->is_device_location = false;
      dev_cached_blocks[block_idx]->memory_block_id = host_cached_blocks[block_idx]->memory_block_id;

      // Reuse the memory blocks.
      host_cached_blocks[block_idx]->block_id = dev_cached_blocks[block_idx]->block_id;
      host_cached_blocks[block_idx]->memory_block_id = original_dev_block_ids[block_idx];
      host_cached_blocks[block_idx]->is_device_location = true;
    }

    return Status();
  }

  // Swap in cached blocks for a specific device.
  // The input dev_cached_blocks and host_cached_blocks are organized as [block_index].
  // This method only processes blocks for the specified device_id, calling BatchSwapIn.
  Status SwapinCachedBlockByDevice(size_t device_id, std::vector<CachedBlockType*>& dev_cached_blocks,
                                   std::vector<CachedBlockType*>& host_cached_blocks) {
    if (dev_cached_blocks.size() != host_cached_blocks.size()) {
      return Status(RET_RUNTIME_FAILED, "SwapinCachedBlockByDevice: dev and host block count mismatch.");
    }

    if (device_id >= block_device_num_) {
      return Status(RET_RUNTIME_FAILED,
                    FormatStr("SwapinCachedBlockByDevice: device_id %d exceeds block_device_num %d.", device_id,
                              block_device_num_));
    }

    size_t block_count = dev_cached_blocks.size();
    if (block_count == 0) {
      return Status();
    }

    // Collect device and host block ids for the specified device.
    std::vector<int> device_block_ids_for_device;
    std::vector<int> host_block_ids_for_device;
    device_block_ids_for_device.reserve(block_count);
    host_block_ids_for_device.reserve(block_count);

    for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
      device_block_ids_for_device.push_back(dev_cached_blocks[block_idx]->memory_block_id);
      host_block_ids_for_device.push_back(host_cached_blocks[block_idx]->memory_block_id);
    }

    // Batch swap in for the specified device.
    Status status =
        block_allocator_group_->BatchSwapIn(device_id, device_block_ids_for_device, host_block_ids_for_device);
    if (!status.OK()) {
      return status;
    }
    return Status();
  }

  // Swap in all cached blocks by iterating over devices concurrently.
  // This method uses threadpool to parallelize SwapinCachedBlockByDevice for each device.
  Status SwapinCachedBlocksByDevice(std::vector<CachedBlockType*>& dev_cached_blocks,
                                    std::vector<CachedBlockType*>& host_cached_blocks) {
    if (dev_cached_blocks.size() != host_cached_blocks.size()) {
      return Status(RET_RUNTIME_FAILED, "SwapinCachedBlocksByDevice: dev and host block count mismatch.");
    }

    size_t block_count = dev_cached_blocks.size();
    if (block_count == 0) {
      return Status();
    }

    // Check whether blocks have already been processed.
    // For swapin, every block maybe processed by multiple threads, so we need to check.
    bool all_processed = true;
    for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
      if (!host_cached_blocks[block_idx]->is_device_location) {
        all_processed = false;
        break;
      }
    }
    if (all_processed) {
      return Status();
    }

    // Collect host block ids for one-time free after swap.
    std::vector<int> host_block_ids_to_free;
    host_block_ids_to_free.reserve(block_count);
    for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
      host_block_ids_to_free.push_back(host_cached_blocks[block_idx]->memory_block_id);
    }

    // Submit swap tasks for each device concurrently.
    std::vector<std::future<Status>> futures;
    futures.reserve(block_device_num_);
    for (size_t device_id = 0; device_id < block_device_num_; ++device_id) {
      futures.push_back(threadpool_->Submit([this, device_id, &dev_cached_blocks, &host_cached_blocks]() -> Status {
        return SwapinCachedBlockByDevice(device_id, dev_cached_blocks, host_cached_blocks);
      }));
    }

    // Wait for all device swap tasks to complete and check status.
    Status final_status;
    for (size_t device_id = 0; device_id < block_device_num_; ++device_id) {
      try {
        Status status = futures[device_id].get();
        if (!status.OK() && final_status.OK()) {
          final_status = status;
        }
      } catch (const std::exception& e) {
        KLLM_LOG_FATAL << "Exception in SwapinCachedBlocksByDevice for device " << device_id << ": " << e.what();
        if (final_status.OK()) {
          final_status =
              Status(RET_RUNTIME_FAILED, FormatStr("Exception in swapin for device %d: %s", device_id, e.what()));
        }
      }
    }

    if (!final_status.OK()) {
      return final_status;
    }

    // Free host blocks once per cached block.
    for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
      block_allocator_group_->GetHostBlockAllocator()->FreeBlocks({host_block_ids_to_free[block_idx]});
    }

    // Finalize: update is_device_location and delete dev_cached_blocks.
    for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
      host_cached_blocks[block_idx]->block_id = dev_cached_blocks[block_idx]->block_id;
      host_cached_blocks[block_idx]->memory_block_id = dev_cached_blocks[block_idx]->memory_block_id;
      host_cached_blocks[block_idx]->is_device_location = true;

      // Remove useless cached block.
      delete dev_cached_blocks[block_idx];
    }

    return Status();
  }

  bool IsPrefixCachingEnabled() { return cache_manager_config_.enable_prefix_caching; }

 protected:
  CacheManagerConfig cache_manager_config_;

  // The block allocators for this cache manager instance.
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_ = nullptr;

  // The devices of current cache manager.
  size_t block_device_num_;

  // The cached blocks that have no computed data, and could be reused freely.
  std::queue<CachedBlockType*> free_cached_blocks_;

  // All requests, the key is request id.
  std::unordered_map<int64_t, std::unique_ptr<CachedRequestType>> cached_requests_;

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
