/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "executor/executor_block_manager.h"
#include "runtime/schedule_output.h"
#include "test.h"
#include "utils/ret_code.h"

namespace ksana_llm {
namespace {

struct SwapOp {
  int index = 0;
  int host_block_id = 0;
  int device_block_id = 0;
  bool is_swapin = false;
};

class MockBlockAllocatorGroup : public BlockAllocatorGroupInterface {
 public:
  MockBlockAllocatorGroup() = default;
  ~MockBlockAllocatorGroup() override = default;

  std::vector<int> GetBlockAllocatorDevices() const override { return {0}; }
  std::shared_ptr<BlockAllocatorInterface> GetHostBlockAllocator() const override {
    struct StubHostAllocator : public BlockAllocatorInterface {
      void PreAllocateBlocks() override {}
      void Clear() override {}
      Status AllocateBlocks(size_t, std::vector<int>&) override {
        return Status(RET_RUNTIME_FAILED, "StubHostAllocator AllocateBlocks not supported");
      }
      Status FreeBlocks(const std::vector<int>&) override { return Status(); }
      Status GetBlockPtrs(const std::vector<int>&, std::vector<void*>&) override {
        return Status(RET_RUNTIME_FAILED, "StubHostAllocator GetBlockPtrs not supported");
      }
      void* GetBlocksBasePtr() override { return nullptr; }
      int GetBlocksBaseId() override { return 0; }
      size_t GetFreeBlockNumber() override { return 0; }
      size_t GetUsedBlockNumber() override { return 0; }
    };
    static auto host_allocator = std::make_shared<StubHostAllocator>();
    return host_allocator;
  }
  std::shared_ptr<BlockAllocatorInterface> GetDeviceBlockAllocator(int) const override { return nullptr; }

  Status SwapIn(int index, int device_block_id, int host_block_id) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ops_.push_back({index, host_block_id, device_block_id, true});
    }
    if (block_swapin_) {
      std::unique_lock<std::mutex> lock(blockin_mutex_);
      blockin_cv_.wait(lock, [this]() { return allow_swapin_; });
    }
    {
      std::lock_guard<std::mutex> lock(donein_mutex_);
      swapin_done_count_++;
    }
    donein_cv_.notify_all();
    return Status();
  }

  Status SwapOut(int index, int host_block_id, int device_block_id) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ops_.push_back({index, host_block_id, device_block_id, false});
    }
    if (block_swapout_) {
      std::unique_lock<std::mutex> lock(block_mutex_);
      block_cv_.wait(lock, [this]() { return allow_swapout_; });
    }
    {
      std::lock_guard<std::mutex> lock(done_mutex_);
      swapout_done_count_++;
    }
    done_cv_.notify_all();
    return Status();
  }

  Status BatchSwapOut(int, const std::vector<int>&, const std::vector<int>&) override {
    return Status(RET_RUNTIME_FAILED, "BatchSwapOut not supported in mock");
  }

  Status BatchSwapIn(int, const std::vector<int>&, const std::vector<int>&) override {
    return Status(RET_RUNTIME_FAILED, "BatchSwapIn not supported in mock");
  }

  const BlockAllocatorGroupConfig& GetBlockAllocatorGroupConfig() const override { return config_; }

  std::vector<SwapOp> GetOps() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ops_;
  }

  void SetBlockSwapOut(bool block) { block_swapout_ = block; }
  void SetBlockSwapIn(bool block) { block_swapin_ = block; }

  void ReleaseSwapOut() {
    {
      std::lock_guard<std::mutex> lock(block_mutex_);
      allow_swapout_ = true;
    }
    block_cv_.notify_all();
  }

  void ReleaseSwapIn() {
    {
      std::lock_guard<std::mutex> lock(blockin_mutex_);
      allow_swapin_ = true;
    }
    blockin_cv_.notify_all();
  }

  size_t GetSwapOutDoneCount() const { return swapout_done_count_.load(); }

  bool WaitForSwapOutDoneCount(size_t expected, int timeout_ms = 1000) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::mutex> lock(done_mutex_);
    while (swapout_done_count_.load() < expected) {
      if (done_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
        return false;
      }
    }
    return true;
  }

  bool WaitForSwapInDoneCount(size_t expected, int timeout_ms = 1000) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::mutex> lock(donein_mutex_);
    while (swapin_done_count_.load() < expected) {
      if (donein_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
        return false;
      }
    }
    return true;
  }

 private:
  BlockAllocatorGroupConfig config_;
  mutable std::mutex mutex_;
  std::vector<SwapOp> ops_;
  std::mutex block_mutex_;
  std::condition_variable block_cv_;
  bool block_swapout_ = false;
  bool allow_swapout_ = false;
  std::mutex blockin_mutex_;
  std::condition_variable blockin_cv_;
  bool block_swapin_ = false;
  bool allow_swapin_ = false;
  mutable std::mutex done_mutex_;
  std::condition_variable done_cv_;
  std::atomic<size_t> swapout_done_count_{0};
  mutable std::mutex donein_mutex_;
  std::condition_variable donein_cv_;
  std::atomic<size_t> swapin_done_count_{0};
};

class MockBlockAllocatorManager : public BlockAllocatorManagerInterface {
 public:
  void Initialize(const BlockAllocatorManagerConfig&) override {}

  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup(int group_id) override {
    auto it = groups_.find(group_id);
    if (it == groups_.end()) {
      return nullptr;
    }
    return it->second;
  }

  void SetGroup(int group_id, std::shared_ptr<BlockAllocatorGroupInterface> group) { groups_[group_id] = group; }

 private:
  std::unordered_map<int, std::shared_ptr<BlockAllocatorGroupInterface>> groups_;
};

TEST(ExecutorBlockManagerTest, SwapOutUsesHostBlockIds) {
  auto manager = std::make_shared<MockBlockAllocatorManager>();
  auto group = std::make_shared<MockBlockAllocatorGroup>();
  manager->SetGroup(0, group);

  ExecutorBlockManager block_manager(0, 0, manager, 1);

  SwapReqBlockInfo swapout_info;
  swapout_info.dev_block_ids = {1, 2};
  swapout_info.host_block_ids = {10, 20};
  std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>> swapout_req_block_ids(1);
  swapout_req_block_ids[0][101] = swapout_info;
  std::vector<std::vector<int64_t>> merged_swapout_req_ids = {{101}};

  Status status = block_manager.SwapoutRequestMemoryBlockAsync(swapout_req_block_ids);
  EXPECT_TRUE(status.OK());
  block_manager.WaitSwapoutRequestMemoryBlock(merged_swapout_req_ids);

  auto ops = group->GetOps();
  ASSERT_EQ(ops.size(), 2);
  EXPECT_FALSE(ops[0].is_swapin);
  EXPECT_EQ(ops[0].device_block_id, 1);
  EXPECT_EQ(ops[0].host_block_id, 10);
  EXPECT_FALSE(ops[1].is_swapin);
  EXPECT_EQ(ops[1].device_block_id, 2);
  EXPECT_EQ(ops[1].host_block_id, 20);
}

TEST(ExecutorBlockManagerTest, WaitSwapoutBlocksUntilSwapOutDone) {
  auto manager = std::make_shared<MockBlockAllocatorManager>();
  auto group = std::make_shared<MockBlockAllocatorGroup>();
  manager->SetGroup(0, group);
  group->SetBlockSwapOut(true);

  ExecutorBlockManager block_manager(0, 0, manager, 1);

  SwapReqBlockInfo swapout_info;
  swapout_info.dev_block_ids = {1};
  swapout_info.host_block_ids = {10};
  std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>> swapout_req_block_ids(1);
  swapout_req_block_ids[0][101] = swapout_info;
  std::vector<std::vector<int64_t>> merged_swapout_req_ids = {{101}};

  Status status = block_manager.SwapoutRequestMemoryBlockAsync(swapout_req_block_ids);
  EXPECT_TRUE(status.OK());

  std::atomic<bool> wait_done(false);
  std::atomic<size_t> swapout_done_at_return(0);
  std::thread wait_thread([&]() {
    block_manager.WaitSwapoutRequestMemoryBlock(merged_swapout_req_ids);
    swapout_done_at_return.store(group->GetSwapOutDoneCount());
    wait_done.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(wait_done.load());

  group->ReleaseSwapOut();
  EXPECT_TRUE(group->WaitForSwapOutDoneCount(1));
  wait_thread.join();

  EXPECT_TRUE(wait_done.load());
  EXPECT_GE(swapout_done_at_return.load(), 1u);
  auto ops = group->GetOps();
  ASSERT_EQ(ops.size(), 1);
  EXPECT_FALSE(ops[0].is_swapin);
  EXPECT_EQ(ops[0].device_block_id, 1);
  EXPECT_EQ(ops[0].host_block_id, 10);
}

TEST(ExecutorBlockManagerTest, SwapInUsesHostBlockIds) {
  auto manager = std::make_shared<MockBlockAllocatorManager>();
  auto group = std::make_shared<MockBlockAllocatorGroup>();
  manager->SetGroup(0, group);

  ExecutorBlockManager block_manager(0, 0, manager, 1);

  SwapReqBlockInfo swapin_info;
  swapin_info.dev_block_ids = {3, 4};
  swapin_info.host_block_ids = {30, 40};
  std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>> swapin_req_block_ids(1);
  swapin_req_block_ids[0][201] = swapin_info;
  std::vector<std::vector<int64_t>> merged_swapin_req_ids = {{201}};

  Status status = block_manager.SwapinRequestMemoryBlockAsync(swapin_req_block_ids);
  EXPECT_TRUE(status.OK());
  block_manager.WaitSwapinRequestMemoryBlock(merged_swapin_req_ids);

  auto ops = group->GetOps();
  ASSERT_EQ(ops.size(), 2);
  EXPECT_TRUE(ops[0].is_swapin);
  EXPECT_EQ(ops[0].device_block_id, 3);
  EXPECT_EQ(ops[0].host_block_id, 30);
  EXPECT_TRUE(ops[1].is_swapin);
  EXPECT_EQ(ops[1].device_block_id, 4);
  EXPECT_EQ(ops[1].host_block_id, 40);
}

TEST(ExecutorBlockManagerTest, WaitSwapinBlocksUntilSwapInDone) {
  auto manager = std::make_shared<MockBlockAllocatorManager>();
  auto group = std::make_shared<MockBlockAllocatorGroup>();
  manager->SetGroup(0, group);
  group->SetBlockSwapIn(true);

  ExecutorBlockManager block_manager(0, 0, manager, 1);

  SwapReqBlockInfo swapin_info;
  swapin_info.dev_block_ids = {5};
  swapin_info.host_block_ids = {50};
  std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>> swapin_req_block_ids(1);
  swapin_req_block_ids[0][301] = swapin_info;
  std::vector<std::vector<int64_t>> merged_swapin_req_ids = {{301}};

  Status status = block_manager.SwapinRequestMemoryBlockAsync(swapin_req_block_ids);
  EXPECT_TRUE(status.OK());

  std::atomic<bool> wait_done(false);
  std::thread wait_thread([&]() {
    block_manager.WaitSwapinRequestMemoryBlock(merged_swapin_req_ids);
    wait_done.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(wait_done.load());

  group->ReleaseSwapIn();
  EXPECT_TRUE(group->WaitForSwapInDoneCount(1));
  wait_thread.join();

  EXPECT_TRUE(wait_done.load());
  auto ops = group->GetOps();
  ASSERT_EQ(ops.size(), 1);
  EXPECT_TRUE(ops[0].is_swapin);
  EXPECT_EQ(ops[0].device_block_id, 5);
  EXPECT_EQ(ops[0].host_block_id, 50);
}

}  // namespace
}  // namespace ksana_llm
