
/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include "executor_block_manager.h"

#include <algorithm>
#include <sstream>

#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ksana_llm {

ExecutorBlockManager::ExecutorBlockManager(int dp_idx, int device_idx_in_dp_group,
                                           std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager,
                                           size_t swap_threadpool_size)
    : dp_idx_(dp_idx),
      device_idx_in_dp_group_(device_idx_in_dp_group),
      block_allocator_manager_(std::move(block_allocator_manager)),
      swap_threadpool_size_(swap_threadpool_size) {
  size_t actual_threadpool_size = std::max<size_t>(swap_threadpool_size_, 1);
  threadpool_ = std::make_shared<ThreadPool>(actual_threadpool_size);
  threadpool_->Start();
}

ExecutorBlockManager::~ExecutorBlockManager() {
  if (threadpool_ != nullptr) {
    threadpool_->Stop();
  }
}

Status ExecutorBlockManager::SwapoutRequestMemoryBlockAsync(
    const std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>>& swapout_req_block_ids) {
  if (swapout_req_block_ids.empty()) {
    return Status();
  }
  if (!IsDpIndexValid(swapout_req_block_ids.size())) {
    return Status(RET_RUNTIME_FAILED, "dp_idx out of range");
  }

  auto block_allocator_group = GetBlockAllocatorGroup();
  if (block_allocator_group == nullptr) {
    return Status(RET_RUNTIME_FAILED, "block_allocator_group is null");
  }
  const auto& dp_swapout_req_block_ids = swapout_req_block_ids[dp_idx_];
  if (dp_swapout_req_block_ids.empty()) {
    return Status();
  }

  std::stringstream so_ss;
  for (auto it = dp_swapout_req_block_ids.begin(); it != dp_swapout_req_block_ids.end(); ++it) {
    so_ss << it->first << ", ";
    Status status = SwapoutRequestMemoryBlockAsyncInternal(it->first, it->second);
    if (!status.OK()) {
      return status;
    }
  }
  KLLM_LOG_INFO << "dp_idx=" << dp_idx_ << ", SwapoutRequestMemoryBlockAsync req_ids=(" << so_ss.str() << ")";
  return Status();
}

void ExecutorBlockManager::WaitSwapoutRequestMemoryBlock(
    const std::vector<std::vector<int64_t>>& merged_swapout_req_ids) {
  if (merged_swapout_req_ids.empty()) {
    return;
  }
  if (!IsDpIndexValid(merged_swapout_req_ids.size())) {
    return;
  }

  const auto& dp_merged_swapout_req_ids = merged_swapout_req_ids[dp_idx_];
  if (dp_merged_swapout_req_ids.empty()) {
    return;
  }

  KLLM_LOG_INFO << "WaitSwapoutRequestMemoryBlock dp_idx=" << dp_idx_
                << ", req num=" << dp_merged_swapout_req_ids.size()
                << ", ids=" << Vector2Str(dp_merged_swapout_req_ids);
  auto block_allocator_group = GetBlockAllocatorGroup();
  if (block_allocator_group == nullptr) {
    KLLM_LOG_ERROR << "WaitSwapoutRequestMemoryBlock failed, block_allocator_group is null.";
    return;
  }
  Status status = WaitSwapoutRequestMemoryBlockInternal(dp_merged_swapout_req_ids);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "WaitSwapoutRequestMemoryBlock failed, dp_idx=" << dp_idx_;
  }
}

Status ExecutorBlockManager::SwapinRequestMemoryBlockAsync(
    const std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>>& swapin_req_block_ids) {
  if (swapin_req_block_ids.empty()) {
    return Status();
  }
  if (!IsDpIndexValid(swapin_req_block_ids.size())) {
    return Status(RET_RUNTIME_FAILED, "dp_idx out of range");
  }

  auto block_allocator_group = GetBlockAllocatorGroup();
  if (block_allocator_group == nullptr) {
    return Status(RET_RUNTIME_FAILED, "block_allocator_group is null");
  }
  const auto& dp_swapin_req_block_ids = swapin_req_block_ids[dp_idx_];
  if (dp_swapin_req_block_ids.empty()) {
    return Status();
  }

  std::stringstream si_ss;
  for (auto it = dp_swapin_req_block_ids.begin(); it != dp_swapin_req_block_ids.end(); ++it) {
    si_ss << it->first << ", ";
    Status status = SwapinRequestMemoryBlockAsyncInternal(it->first, it->second);
    if (!status.OK()) {
      return status;
    }
  }
  KLLM_LOG_INFO << "dp_idx=" << dp_idx_ << ", SwapinRequestMemoryBlockAsync req_ids=(" << si_ss.str() << ")";
  return Status();
}

void ExecutorBlockManager::WaitSwapinRequestMemoryBlock(
    const std::vector<std::vector<int64_t>>& merged_swapin_req_ids) {
  if (merged_swapin_req_ids.empty()) {
    return;
  }
  if (!IsDpIndexValid(merged_swapin_req_ids.size())) {
    return;
  }

  const auto& dp_merged_swapin_req_ids = merged_swapin_req_ids[dp_idx_];
  if (dp_merged_swapin_req_ids.empty()) {
    return;
  }

  KLLM_LOG_INFO << "WaitSwapinRequestMemoryBlock dp_idx=" << dp_idx_ << ", req num=" << dp_merged_swapin_req_ids.size()
                << ", ids=" << Vector2Str(dp_merged_swapin_req_ids);
  auto block_allocator_group = GetBlockAllocatorGroup();
  if (block_allocator_group == nullptr) {
    KLLM_LOG_ERROR << "WaitSwapinRequestMemoryBlock failed, block_allocator_group is null.";
    return;
  }
  Status status = WaitSwapinRequestMemoryBlockInternal(dp_merged_swapin_req_ids);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "WaitSwapinRequestMemoryBlock failed, dp_idx=" << dp_idx_;
  }
}

bool ExecutorBlockManager::IsDpIndexValid(size_t data_size) const {
  return data_size > 0 && dp_idx_ >= 0 && static_cast<size_t>(dp_idx_) < data_size;
}

std::shared_ptr<BlockAllocatorGroupInterface> ExecutorBlockManager::GetBlockAllocatorGroup() {
  if (block_allocator_manager_ == nullptr) {
    return nullptr;
  }
  block_allocator_group_ = block_allocator_manager_->GetBlockAllocatorGroup(dp_idx_);
  return block_allocator_group_;
}

Status ExecutorBlockManager::SwapoutRequestMemoryBlockAsyncInternal(int64_t req_id,
                                                                    const SwapReqBlockInfo& block_info) {
  const auto& dev_block_ids = block_info.dev_block_ids;
  const auto& host_block_ids = block_info.host_block_ids;
  if (host_block_ids.size() != dev_block_ids.size()) {
    return Status(RET_RUNTIME_FAILED, "swapout host_block_ids size mismatch");
  }

  request_memory_block_swap_result_[req_id] = threadpool_->Submit([=] {
    for (size_t i = 0; i < dev_block_ids.size(); ++i) {
      block_allocator_group_->SwapOut(device_idx_in_dp_group_, host_block_ids[i], dev_block_ids[i]);
    }
  });

  return Status();
}

Status ExecutorBlockManager::SwapinRequestMemoryBlockAsyncInternal(int64_t req_id, const SwapReqBlockInfo& block_info) {
  const auto& dev_block_ids = block_info.dev_block_ids;
  const auto& host_block_ids = block_info.host_block_ids;
  if (host_block_ids.size() != dev_block_ids.size()) {
    return Status(RET_RUNTIME_FAILED, "swapin host_block_ids size mismatch");
  }

  request_memory_block_swap_result_[req_id] = threadpool_->Submit([=] {
    for (size_t i = 0; i < dev_block_ids.size(); ++i) {
      block_allocator_group_->SwapIn(device_idx_in_dp_group_, dev_block_ids[i], host_block_ids[i]);
    }
  });

  return Status();
}

Status ExecutorBlockManager::WaitSwapoutRequestMemoryBlockInternal(const std::vector<int64_t>& req_ids) {
  Status status = WaitSwappinessRequestMemoryBlockInternal(req_ids);
  for (int64_t req_id : req_ids) {
    request_memory_block_swap_result_.erase(req_id);
  }
  return status;
}

Status ExecutorBlockManager::WaitSwapinRequestMemoryBlockInternal(const std::vector<int64_t>& req_ids) {
  Status status = WaitSwappinessRequestMemoryBlockInternal(req_ids);
  for (int64_t req_id : req_ids) {
    request_memory_block_swap_result_.erase(req_id);
  }
  return status;
}

Status ExecutorBlockManager::WaitSwappinessRequestMemoryBlockInternal(const std::vector<int64_t>& req_ids) {
  for (int64_t req_id : req_ids) {
    auto it = request_memory_block_swap_result_.find(req_id);
    if (it == request_memory_block_swap_result_.end()) {
      KLLM_LOG_WARNING << "Wait swapout request " << req_id << " not found.";
      continue;
    }

    try {
      it->second.get();
    } catch (const std::exception& e) {
      KLLM_LOG_FATAL << "Exception in swapout, info: " << e.what();
    }
  }

  return Status();
}

}  // namespace ksana_llm
