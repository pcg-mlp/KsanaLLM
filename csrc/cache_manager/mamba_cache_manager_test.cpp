/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/
#include "cache_manager/mamba_cache_manager.h"

#include <gtest/gtest.h>

#include <memory>
#include <type_traits>

#include "cache_manager/prefix_cache_manager_test_helper.h"
#include "test.h"

using namespace ksana_llm;

static_assert(!std::is_copy_constructible_v<MambaCacheManager>);
static_assert(!std::is_copy_assignable_v<MambaCacheManager>);

class MambaCacheManagerTest : public testing::Test {
 protected:
  void Initialize(size_t mamba_block_num) {
    BlockAllocatorGroupConfig cfg;
    cfg.devices = {0};
    cfg.device_block_num = mamba_block_num;
    cfg.host_block_num = 0;
    cfg.block_size = sizeof(float);

    context_ = std::make_shared<Context>();
    memory_allocator_ = std::make_shared<FakedMemoryAllocator>();

    BlockAllocatorCreationFunc creation_fn =
        [](MemoryLocation location, size_t block_num, size_t block_size, int rank,
           std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
           std::shared_ptr<Context> context) -> std::shared_ptr<BlockAllocatorInterface> {
      return std::make_shared<FakedBlockAllocator>(location, block_num, block_size, rank, memory_allocator, context);
    };

    allocator_group_ = std::make_shared<FakedBlockAllocatorGroup>(cfg, memory_allocator_, context_, creation_fn);
    manager_ = std::make_unique<MambaCacheManager>(allocator_group_);
  }

  size_t GetFreeBlockCount() const { return allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(); }

  std::shared_ptr<Context> context_;
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_;
  std::shared_ptr<BlockAllocatorGroupInterface> allocator_group_;
  std::unique_ptr<MambaCacheManager> manager_;
};

TEST_F(MambaCacheManagerTest, AllocateRequestMambaBlock_TwoRequestsGetDistinctBlocks) {
  Initialize(/*mamba_block_num=*/2);

  int first_mamba_block_id = -1;
  int second_mamba_block_id = -1;
  Status first_status = manager_->AllocateRequestMambaBlock(/*req_id=*/1001, first_mamba_block_id);
  Status second_status = manager_->AllocateRequestMambaBlock(/*req_id=*/1002, second_mamba_block_id);

  ASSERT_TRUE(first_status.OK()) << first_status.GetMessage();
  ASSERT_TRUE(second_status.OK()) << second_status.GetMessage();
  EXPECT_GT(first_mamba_block_id, 0);
  EXPECT_GT(second_mamba_block_id, 0);
  EXPECT_NE(first_mamba_block_id, second_mamba_block_id);
  EXPECT_EQ(GetFreeBlockCount(), 0u);
}

TEST_F(MambaCacheManagerTest, AllocateRequestMambaBlock_ReservesZeroForNullSlot) {
  Initialize(/*mamba_block_num=*/1);

  int mamba_kv_cache_block_id = -1;
  Status status = manager_->AllocateRequestMambaBlock(/*req_id=*/1001, mamba_kv_cache_block_id);

  ASSERT_TRUE(status.OK()) << status.GetMessage();
  EXPECT_EQ(mamba_kv_cache_block_id, 1);
}

TEST_F(MambaCacheManagerTest, AllocateRequestMambaBlock_ExistingRequestReturnsSameBlock) {
  Initialize(/*mamba_block_num=*/2);

  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(manager_->AllocateRequestMambaBlock(/*req_id=*/1001, mamba_kv_cache_block_id).OK());
  const int first_block_id = mamba_kv_cache_block_id;

  int second_mamba_block_id = -1;
  Status second_status = manager_->AllocateRequestMambaBlock(/*req_id=*/1001, second_mamba_block_id);

  ASSERT_TRUE(second_status.OK()) << second_status.GetMessage();
  EXPECT_EQ(second_mamba_block_id, first_block_id);
  EXPECT_EQ(GetFreeBlockCount(), 1u);
}

TEST_F(MambaCacheManagerTest, AllocateRequestMambaBlock_ExhaustedPoolReturnsAllocatorError) {
  Initialize(/*mamba_block_num=*/1);

  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(manager_->AllocateRequestMambaBlock(/*req_id=*/1001, mamba_kv_cache_block_id).OK());

  int second_mamba_block_id = -1;
  Status second_status = manager_->AllocateRequestMambaBlock(/*req_id=*/1002, second_mamba_block_id);

  EXPECT_EQ(second_status.GetCode(), RET_OUT_OF_DEVICE_MEMORY);
  EXPECT_EQ(second_status.GetMessage(), "No more blocks.");
  EXPECT_EQ(second_mamba_block_id, -1);
  EXPECT_EQ(GetFreeBlockCount(), 0u);
}

TEST_F(MambaCacheManagerTest, AllocateRequestMambaBlock_FreedBlockIsNewForAnotherRequest) {
  Initialize(/*mamba_block_num=*/1);

  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(manager_->AllocateRequestMambaBlock(/*req_id=*/1001, mamba_kv_cache_block_id).OK());
  const int first_block_id = mamba_kv_cache_block_id;
  ASSERT_TRUE(manager_->FreeRequestMambaBlock(/*req_id=*/1001).OK());

  EXPECT_EQ(GetFreeBlockCount(), 1u);

  int second_mamba_block_id = -1;
  Status second_status = manager_->AllocateRequestMambaBlock(/*req_id=*/1002, second_mamba_block_id);

  ASSERT_TRUE(second_status.OK()) << second_status.GetMessage();
  EXPECT_EQ(second_mamba_block_id, first_block_id);
  EXPECT_EQ(GetFreeBlockCount(), 0u);
}

TEST_F(MambaCacheManagerTest, FreeRequestMambaBlock_UnknownRequestIsNoOp) {
  Initialize(/*mamba_block_num=*/2);

  Status status = manager_->FreeRequestMambaBlock(/*req_id=*/1001);

  ASSERT_TRUE(status.OK()) << status.GetMessage();
  EXPECT_EQ(GetFreeBlockCount(), 2u);
}

TEST_F(MambaCacheManagerTest, GetRequestMambaBlockId_ReturnsWorkingBlockOrMinusOne) {
  Initialize(/*mamba_block_num=*/2);

  EXPECT_EQ(manager_->GetRequestMambaBlockId(/*req_id=*/1001), -1);

  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(manager_->AllocateRequestMambaBlock(/*req_id=*/1001, mamba_kv_cache_block_id).OK());
  EXPECT_GT(manager_->GetRequestMambaBlockId(/*req_id=*/1001), 0);

  ASSERT_TRUE(manager_->FreeRequestMambaBlock(/*req_id=*/1001).OK());
  EXPECT_EQ(manager_->GetRequestMambaBlockId(/*req_id=*/1001), -1);
}
