/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "runtime/schedule_output.h"
#include "runtime/threadpool.h"
#include "utils/status.h"

namespace ksana_llm {

class ExecutorBlockManager {
 public:
  ExecutorBlockManager(int dp_idx, int device_idx_in_dp_group,
                       std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager,
                       size_t swap_threadpool_size);
  ~ExecutorBlockManager();

  Status SwapoutRequestMemoryBlockAsync(
      const std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>>& swapout_req_block_ids);
  void WaitSwapoutRequestMemoryBlock(const std::vector<std::vector<int64_t>>& merged_swapout_req_ids);
  Status SwapinRequestMemoryBlockAsync(
      const std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>>& swapin_req_block_ids);
  void WaitSwapinRequestMemoryBlock(const std::vector<std::vector<int64_t>>& merged_swapin_req_ids);

 private:
  bool IsDpIndexValid(size_t data_size) const;
  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup();
  Status SwapoutRequestMemoryBlockAsyncInternal(int64_t req_id, const SwapReqBlockInfo& block_info);
  Status SwapinRequestMemoryBlockAsyncInternal(int64_t req_id, const SwapReqBlockInfo& block_info);
  Status WaitSwapoutRequestMemoryBlockInternal(const std::vector<int64_t>& req_ids);
  Status WaitSwapinRequestMemoryBlockInternal(const std::vector<int64_t>& req_ids);
  Status WaitSwappinessRequestMemoryBlockInternal(const std::vector<int64_t>& req_ids);

 private:
  const int dp_idx_;
  const int device_idx_in_dp_group_;
  std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager_ = nullptr;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_ = nullptr;
  std::shared_ptr<ThreadPool> threadpool_ = nullptr;
  const size_t swap_threadpool_size_;
  std::unordered_map<int64_t, std::future<void>> request_memory_block_swap_result_;
};

}  // namespace ksana_llm
