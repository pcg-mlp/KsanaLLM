/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "global_cache_connector/global_cache_connector_interface.h"
#include "ipc/functions/global_cache_query.h"
#include "loguru.hpp"

namespace ksana_llm {

class MaxLogVerbosityEnv : public ::testing::Environment {
 public:
  void SetUp() override { loguru::g_stderr_verbosity = loguru::Verbosity_MAX; }
};
static auto* unused_log_env = ::testing::AddGlobalTestEnvironment(new MaxLogVerbosityEnv());

class MockConnector : public GlobalCacheConnectorInterface {
 public:
  MOCK_METHOD(Status, Initialize, (const GlobalCacheConnectorConfig&), (override));
  MOCK_METHOD(Status, QueryAndFetch,
              (const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
               std::vector<int>&, bool, int, const GlobalCacheSwaContext*, GlobalCacheSwaResult*),
              (override));
  MOCK_METHOD(Status, PutGlobalCacheBlocks,
              (const std::vector<int>&, int, int, int, int, size_t, const std::vector<int>&, const std::vector<void*>&,
               int*),
              (override));
  MOCK_METHOD(Status, PutSwaBlocks,
              (const std::vector<int>&, int, int, int, size_t, const std::vector<size_t>&, const std::vector<void*>&,
               int*),
              (override));
  MOCK_METHOD(Status, PutFirstTokens, (const std::vector<int>&, const std::vector<int>&), (override));
  MOCK_METHOD(Status, QueryFirstTokens, (const std::vector<int>&, std::vector<int>&, bool&), (override));
  MOCK_METHOD(Status, Shutdown, (), (override));
};

class MockBlockAllocator : public BlockAllocatorInterface {
 public:
  MOCK_METHOD(void, PreAllocateBlocks, (), (override));
  MOCK_METHOD(void, Clear, (), (override));
  MOCK_METHOD(Status, AllocateBlocks, (size_t, std::vector<int>&), (override));
  MOCK_METHOD(Status, FreeBlocks, (const std::vector<int>&), (override));
  MOCK_METHOD(Status, GetBlockPtrs, (const std::vector<int>&, std::vector<void*>&), (override));
  MOCK_METHOD(void*, GetBlocksBasePtr, (), (override));
  MOCK_METHOD(int, GetBlocksBaseId, (), (override));
  MOCK_METHOD(size_t, GetFreeBlockNumber, (), (override));
  MOCK_METHOD(size_t, GetUsedBlockNumber, (), (override));
};

class MockBlockAllocatorGroup : public BlockAllocatorGroupInterface {
 public:
  MOCK_METHOD(std::vector<int>, GetBlockAllocatorDevices, (), (const, override));
  MOCK_METHOD(std::shared_ptr<BlockAllocatorInterface>, GetHostBlockAllocator, (), (const, override));
  MOCK_METHOD(std::shared_ptr<BlockAllocatorInterface>, GetDeviceBlockAllocator, (int), (const, override));
  MOCK_METHOD(const BlockAllocatorGroupConfig&, GetBlockAllocatorGroupConfig, (), (const, override));
};

class MockBlockAllocatorManager : public BlockAllocatorManagerInterface {
 public:
  MOCK_METHOD(void, Initialize, (const BlockAllocatorManagerConfig&), (override));
  MOCK_METHOD(std::shared_ptr<BlockAllocatorGroupInterface>, GetBlockAllocatorGroup, (int), (override));
};

class QueryCallableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_connector_ = std::make_shared<testing::NiceMock<MockConnector>>();
    mock_bam_ = std::make_shared<testing::NiceMock<MockBlockAllocatorManager>>();
    mock_group_ = std::make_shared<testing::NiceMock<MockBlockAllocatorGroup>>();
    mock_device_alloc_ = std::make_shared<testing::NiceMock<MockBlockAllocator>>();

    ON_CALL(*mock_bam_, GetBlockAllocatorGroup(testing::_)).WillByDefault(testing::Return(mock_group_));
    ON_CALL(*mock_group_, GetDeviceBlockAllocator(testing::_)).WillByDefault(testing::Return(mock_device_alloc_));

    block_config_.block_size = 4096;
    ON_CALL(*mock_group_, GetBlockAllocatorGroupConfig()).WillByDefault(testing::ReturnRef(block_config_));

    ON_CALL(*mock_device_alloc_, GetBlockPtrs(testing::_, testing::_))
        .WillByDefault([](const std::vector<int>& blocks, std::vector<void*>& ptrs) {
          ptrs.resize(blocks.size(), reinterpret_cast<void*>(0xDEAD));
          return Status();
        });

    callable_ = std::make_unique<GlobalCacheQueryCallable>();
    callable_->SetGlobalCacheConnector(mock_connector_);
    callable_->SetBlockAllocatorManager(mock_bam_, 0);
    callable_->SetDeviceId(0);
    callable_->SetGlobalCacheRole("", 5000);
  }

  void EnableSwaMocks(size_t swa_block_byte_size = 8192) {
    mock_swa_bam_ = std::make_shared<testing::NiceMock<MockBlockAllocatorManager>>();
    mock_swa_group_ = std::make_shared<testing::NiceMock<MockBlockAllocatorGroup>>();
    mock_swa_device_alloc_ = std::make_shared<testing::NiceMock<MockBlockAllocator>>();
    ON_CALL(*mock_swa_bam_, GetBlockAllocatorGroup(testing::_)).WillByDefault(testing::Return(mock_swa_group_));
    ON_CALL(*mock_swa_group_, GetDeviceBlockAllocator(testing::_))
        .WillByDefault(testing::Return(mock_swa_device_alloc_));
    swa_block_config_.block_size = swa_block_byte_size;
    ON_CALL(*mock_swa_group_, GetBlockAllocatorGroupConfig()).WillByDefault(testing::ReturnRef(swa_block_config_));
    ON_CALL(*mock_swa_device_alloc_, GetBlockPtrs(testing::_, testing::_))
        .WillByDefault([](const std::vector<int>& blocks, std::vector<void*>& ptrs) {
          ptrs.resize(blocks.size(), reinterpret_cast<void*>(0xBEEF));
          return Status();
        });
    callable_->SetSwaBlockAllocatorManager(mock_swa_bam_);
  }

  static void FillSwaFields(GlobalCacheQueryRequest& req, size_t max_cached_len = 64) {
    req.swa_kv_cache_block_ids = {10, 11};
    req.swa_block_idx_offsets = {0, 1};
    req.swa_window_block_num = 2;
    req.max_cached_len = max_cached_len;
  }

  GlobalCacheQueryRequest MakeRequest(int64_t req_id, int token_num = 10, int local_cached = 0) {
    GlobalCacheQueryRequest req;
    req.req_id = req_id;
    req.token_ids.resize(token_num);
    for (int i = 0; i < token_num; ++i) {
      req.token_ids[i] = i + 100;
    }
    req.local_cached_len = local_cached;
    req.block_size = 16;
    req.kv_cache_block_ids = {0, 1, 2};
    return req;
  }

  void StubQueryFullHit() {
    ON_CALL(*mock_connector_,
            QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                          testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillByDefault([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                          std::vector<int>& matched_ranges, bool, int, const GlobalCacheSwaContext*,
                          GlobalCacheSwaResult*) {
          matched_ranges = {0, 9};
          return Status();
        });
  }

  std::shared_ptr<testing::NiceMock<MockConnector>> mock_connector_;
  std::shared_ptr<testing::NiceMock<MockBlockAllocatorManager>> mock_bam_;
  std::shared_ptr<testing::NiceMock<MockBlockAllocatorGroup>> mock_group_;
  std::shared_ptr<testing::NiceMock<MockBlockAllocator>> mock_device_alloc_;
  std::shared_ptr<testing::NiceMock<MockBlockAllocatorManager>> mock_swa_bam_;
  std::shared_ptr<testing::NiceMock<MockBlockAllocatorGroup>> mock_swa_group_;
  std::shared_ptr<testing::NiceMock<MockBlockAllocator>> mock_swa_device_alloc_;
  BlockAllocatorGroupConfig block_config_;
  BlockAllocatorGroupConfig swa_block_config_;
  std::unique_ptr<GlobalCacheQueryCallable> callable_;
};

// ==================== 测试用例 ====================

TEST_F(QueryCallableTest, NoConnectorReportsFailure) {
  callable_->SetGlobalCacheConnector(nullptr);

  auto req = MakeRequest(100);
  GlobalCacheQueryResponse resp;
  // 无 ipc_func_wrapper → report 不发送，但不崩溃
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, EmptyBlockPtrsReportsFailure) {
  ON_CALL(*mock_device_alloc_, GetBlockPtrs(testing::_, testing::_))
      .WillByDefault([](const std::vector<int>&, std::vector<void*>& ptrs) {
        ptrs.clear();
        return Status();
      });

  auto req = MakeRequest(200);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, SuccessfulQueryAndFetch) {
  ON_CALL(*mock_connector_,
          QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                        testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                        std::vector<int>& matched_ranges, bool, int, const GlobalCacheSwaContext*,
                        GlobalCacheSwaResult*) {
        matched_ranges = {0, 9};
        return Status();
      });

  auto req = MakeRequest(300, 10);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, QueryAndFetchFailure) {
  ON_CALL(*mock_connector_,
          QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                        testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault(testing::Return(Status(RET_RUNTIME_FAILED, "timeout")));

  auto req = MakeRequest(400, 10);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, StandaloneFullHitQueriesFirstTokens) {
  ON_CALL(*mock_connector_,
          QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                        testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                        std::vector<int>& matched_ranges, bool, int, const GlobalCacheSwaContext*,
                        GlobalCacheSwaResult*) {
        matched_ranges = {0, 9};
        return Status();
      });

  EXPECT_CALL(*mock_connector_, QueryFirstTokens(testing::_, testing::_, testing::_))
      .WillOnce([](const std::vector<int>&, std::vector<int>& tokens, bool& found) {
        tokens = {42};
        found = true;
        return Status();
      });

  auto req = MakeRequest(500, 10, 0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, SetterMethods) {
  auto callable = std::make_unique<GlobalCacheQueryCallable>();
  callable->SetGlobalCacheConnector(mock_connector_);
  callable->SetBlockAllocatorManager(mock_bam_, 1);
  callable->SetSwaBlockAllocatorManager(mock_bam_);
  callable->SetIpcFuncWrapper(nullptr);
  callable->SetEnvironment(nullptr);
  callable->SetDeviceId(3);
  callable->SetGlobalCacheRole("decode", 3000);

  EXPECT_EQ(callable->device_id_, 3);
  EXPECT_EQ(callable->role_, "decode");
  EXPECT_EQ(callable->timeout_ms_, 3000);
  EXPECT_EQ(callable->dp_group_id_, 1);
}

TEST_F(QueryCallableTest, PrefillRoleNoFirstTokenQuery) {
  callable_->SetGlobalCacheRole("prefill", 5000);
  StubQueryFullHit();
  EXPECT_CALL(*mock_connector_, QueryFirstTokens(testing::_, testing::_, testing::_)).Times(0);

  auto req = MakeRequest(600, 10, 0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST(QueryRequestSerializationTest, RoundTrip) {
  GlobalCacheQueryRequest original;
  original.req_id = 42;
  original.token_ids = {10, 20, 30, 40, 50};
  original.local_cached_len = 3;
  original.block_size = 16;
  original.kv_cache_block_ids = {0, 1, 2, 3};
  original.enqueue_wall_us = 123456789;
  original.swa_kv_cache_block_ids = {7, 8};
  original.swa_block_idx_offsets = {2, 3};
  original.swa_window_block_num = 4;
  original.max_cached_len = 99;

  std::vector<uint8_t> buffer(4096);
  size_t bytes = 0;
  ASSERT_TRUE(original.Serialize(buffer.data(), &bytes).OK());
  EXPECT_GT(bytes, 0u);

  GlobalCacheQueryRequest restored;
  ASSERT_TRUE(restored.Deserialize(buffer.data()).OK());
  EXPECT_EQ(restored.req_id, 42);
  EXPECT_EQ(restored.token_ids, original.token_ids);
  EXPECT_EQ(restored.local_cached_len, 3u);
  EXPECT_EQ(restored.block_size, 16);
  EXPECT_EQ(restored.kv_cache_block_ids, original.kv_cache_block_ids);
  EXPECT_EQ(restored.enqueue_wall_us, 123456789);
  EXPECT_EQ(restored.swa_kv_cache_block_ids, original.swa_kv_cache_block_ids);
  EXPECT_EQ(restored.swa_block_idx_offsets, original.swa_block_idx_offsets);
  EXPECT_EQ(restored.swa_window_block_num, 4u);
  EXPECT_EQ(restored.max_cached_len, 99u);
}

TEST(QueryRequestSerializationTest, EmptyVectors) {
  GlobalCacheQueryRequest original;
  original.req_id = 99;
  original.local_cached_len = 0;
  original.block_size = 0;

  std::vector<uint8_t> buffer(4096);
  size_t bytes = 0;
  ASSERT_TRUE(original.Serialize(buffer.data(), &bytes).OK());

  GlobalCacheQueryRequest restored;
  ASSERT_TRUE(restored.Deserialize(buffer.data()).OK());
  EXPECT_EQ(restored.req_id, 99);
  EXPECT_TRUE(restored.token_ids.empty());
  EXPECT_TRUE(restored.kv_cache_block_ids.empty());
  EXPECT_TRUE(restored.swa_kv_cache_block_ids.empty());
  EXPECT_TRUE(restored.swa_block_idx_offsets.empty());
}

TEST_F(QueryCallableTest, DecodeRoleFullHitTokensFound) {
  callable_->SetGlobalCacheRole("decode", 5000);
  StubQueryFullHit();
  EXPECT_CALL(*mock_connector_, QueryFirstTokens(testing::_, testing::_, testing::_))
      .WillOnce([](const std::vector<int>&, std::vector<int>& tokens, bool& found) {
        tokens = {42, 43};
        found = true;
        return Status();
      });

  auto req = MakeRequest(700, 10, 0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, DecodeRoleTokensNotFoundTimeout) {
  callable_->SetGlobalCacheRole("decode", 50);
  StubQueryFullHit();
  ON_CALL(*mock_connector_, QueryFirstTokens(testing::_, testing::_, testing::_))
      .WillByDefault([](const std::vector<int>&, std::vector<int>&, bool& found) {
        found = false;
        return Status();
      });

  auto start = std::chrono::steady_clock::now();
  auto req = MakeRequest(800, 10, 0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
  EXPECT_GE(elapsed.count(), 40);
}

TEST_F(QueryCallableTest, DecodeRolePartialHitNoTokenQuery) {
  callable_->SetGlobalCacheRole("decode", 5000);
  ON_CALL(*mock_connector_,
          QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                        testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                        std::vector<int>& matched_ranges, bool, int, const GlobalCacheSwaContext*,
                        GlobalCacheSwaResult*) {
        matched_ranges = {0, 4};
        return Status();
      });
  EXPECT_CALL(*mock_connector_, QueryFirstTokens(testing::_, testing::_, testing::_)).Times(0);

  auto req = MakeRequest(900, 10, 0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, NonDevice0NoFirstTokenQuery) {
  callable_->SetDeviceId(1);
  StubQueryFullHit();
  EXPECT_CALL(*mock_connector_, QueryFirstTokens(testing::_, testing::_, testing::_)).Times(0);

  auto req = MakeRequest(1000, 10, 0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, EnqueueWallUsComputesWait) {
  StubQueryFullHit();
  auto req = MakeRequest(1100, 10, 0);
  req.enqueue_wall_us = 1;  // >0 走 wait_us 计算分支
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, GetBlockPtrsFailureSkipsQuery) {
  ON_CALL(*mock_device_alloc_, GetBlockPtrs(testing::_, testing::_))
      .WillByDefault(testing::Return(Status(RET_RUNTIME_FAILED, "block miss")));
  EXPECT_CALL(*mock_connector_,
              QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                            testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .Times(0);

  auto req = MakeRequest(1200);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, BlockPtrSizeMismatchSkipsQuery) {
  ON_CALL(*mock_device_alloc_, GetBlockPtrs(testing::_, testing::_))
      .WillByDefault([](const std::vector<int>&, std::vector<void*>& ptrs) {
        ptrs = {reinterpret_cast<void*>(0x1)};  // 故意少于 block ids
        return Status();
      });
  EXPECT_CALL(*mock_connector_,
              QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                            testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .Times(0);

  auto req = MakeRequest(1300);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, NullDeviceAllocatorSkipsQuery) {
  ON_CALL(*mock_group_, GetDeviceBlockAllocator(testing::_)).WillByDefault(testing::Return(nullptr));
  EXPECT_CALL(*mock_connector_,
              QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                            testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .Times(0);

  auto req = MakeRequest(1400);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, NoBlockAllocatorManagerSkipsQuery) {
  auto callable = std::make_unique<GlobalCacheQueryCallable>();
  callable->SetGlobalCacheConnector(mock_connector_);
  callable->SetDeviceId(0);
  EXPECT_CALL(*mock_connector_,
              QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                            testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .Times(0);

  auto req = MakeRequest(1500);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, EmptyMatchedRangesNoTokenQuery) {
  ON_CALL(*mock_connector_,
          QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                        testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                        std::vector<int>& matched_ranges, bool, int, const GlobalCacheSwaContext*,
                        GlobalCacheSwaResult*) {
        matched_ranges.clear();
        return Status();
      });
  EXPECT_CALL(*mock_connector_, QueryFirstTokens(testing::_, testing::_, testing::_)).Times(0);

  auto req = MakeRequest(1600, 10, 0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, SwaEnabledPassesContext) {
  EnableSwaMocks();
  EXPECT_CALL(*mock_connector_,
              QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                            testing::_, testing::_, testing::_, testing::_, testing::NotNull(), testing::NotNull()))
      .WillOnce([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                   std::vector<int>& matched_ranges, bool, int, const GlobalCacheSwaContext* swa_ctx,
                   GlobalCacheSwaResult* swa_result) {
        EXPECT_NE(swa_ctx, nullptr);
        EXPECT_EQ(swa_ctx->swa_block_ptrs.size(), 2u);
        EXPECT_EQ(swa_ctx->local_swa_block_idxs.size(), 2u);
        EXPECT_GT(swa_ctx->swa_block_byte_size, 0u);
        EXPECT_EQ(swa_ctx->max_hit_token_num, 64u);
        if (swa_result != nullptr) {
          swa_result->swa_aligned_hit_len = 32;
          swa_result->filled_block_idxs = {0, 1};
        }
        matched_ranges = {0, 9};
        return Status();
      });

  auto req = MakeRequest(1700, 10, 0);
  FillSwaFields(req, /*max_cached_len=*/64);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, SwaGetBlockPtrsFailFallsBackMainKv) {
  EnableSwaMocks();
  ON_CALL(*mock_swa_device_alloc_, GetBlockPtrs(testing::_, testing::_))
      .WillByDefault(testing::Return(Status(RET_RUNTIME_FAILED, "swa miss")));
  EXPECT_CALL(*mock_connector_,
              QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                            testing::_, testing::_, testing::_, testing::_, testing::IsNull(), testing::IsNull()))
      .WillOnce([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                   std::vector<int>& matched_ranges, bool, int, const GlobalCacheSwaContext*, GlobalCacheSwaResult*) {
        matched_ranges = {0, 9};
        return Status();
      });

  auto req = MakeRequest(1800, 10, 0);
  FillSwaFields(req);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, SwaIncompleteMaxCachedZeroFallsBack) {
  EnableSwaMocks();
  EXPECT_CALL(*mock_connector_,
              QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                            testing::_, testing::_, testing::_, testing::_, testing::IsNull(), testing::IsNull()))
      .WillOnce([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                   std::vector<int>& matched_ranges, bool, int, const GlobalCacheSwaContext*, GlobalCacheSwaResult*) {
        matched_ranges = {0, 9};
        return Status();
      });

  auto req = MakeRequest(1900, 10, 0);
  FillSwaFields(req, /*max_cached_len=*/0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, SwaIncompleteZeroBlockByteFallsBack) {
  EnableSwaMocks(/*swa_block_byte_size=*/0);
  EXPECT_CALL(*mock_connector_,
              QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                            testing::_, testing::_, testing::_, testing::_, testing::IsNull(), testing::IsNull()))
      .WillOnce([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                   std::vector<int>& matched_ranges, bool, int, const GlobalCacheSwaContext*, GlobalCacheSwaResult*) {
        matched_ranges = {0, 9};
        return Status();
      });

  auto req = MakeRequest(2000, 10, 0);
  FillSwaFields(req, /*max_cached_len=*/64);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, SwaIdOffsetSizeMismatchNoSwa) {
  EnableSwaMocks();
  EXPECT_CALL(*mock_connector_,
              QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                            testing::_, testing::_, testing::_, testing::_, testing::IsNull(), testing::IsNull()))
      .WillOnce([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                   std::vector<int>& matched_ranges, bool, int, const GlobalCacheSwaContext*, GlobalCacheSwaResult*) {
        matched_ranges = {0, 9};
        return Status();
      });

  auto req = MakeRequest(2100, 10, 0);
  req.swa_kv_cache_block_ids = {10, 11};
  req.swa_block_idx_offsets = {0};  // 长度不一致
  req.max_cached_len = 64;
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, SwaNullDeviceAllocatorFallsBack) {
  EnableSwaMocks();
  ON_CALL(*mock_swa_group_, GetDeviceBlockAllocator(testing::_)).WillByDefault(testing::Return(nullptr));
  EXPECT_CALL(*mock_connector_,
              QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                            testing::_, testing::_, testing::_, testing::_, testing::IsNull(), testing::IsNull()))
      .WillOnce([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                   std::vector<int>& matched_ranges, bool, int, const GlobalCacheSwaContext*, GlobalCacheSwaResult*) {
        matched_ranges = {0, 9};
        return Status();
      });

  auto req = MakeRequest(2200, 10, 0);
  FillSwaFields(req);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, SecondCallReusesAllocatorGroup) {
  StubQueryFullHit();
  auto req = MakeRequest(2300, 10, 0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

}  // namespace ksana_llm
