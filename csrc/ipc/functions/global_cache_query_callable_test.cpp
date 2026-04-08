/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "global_cache_connector/global_cache_connector_interface.h"
#include "ipc/functions/global_cache_query.h"
#include "loguru.hpp"

namespace ksana_llm {
namespace {

class MaxLogVerbosityEnv : public ::testing::Environment {
 public:
  void SetUp() override { loguru::g_stderr_verbosity = loguru::Verbosity_MAX; }
};
static auto* unused_log_env = ::testing::AddGlobalTestEnvironment(new MaxLogVerbosityEnv());

// ==================== Mock 类 ====================

class MockConnector : public GlobalCacheConnectorInterface {
 public:
  MOCK_METHOD(Status, Initialize, (const GlobalCacheConnectorConfig&), (override));
  MOCK_METHOD(Status, QueryAndFetch,
              (const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
               std::vector<int>&, bool, int),
              (override));
  MOCK_METHOD(Status, PutGlobalCacheBlocks,
              (const std::vector<int>&, int, int, int, int, size_t, const std::vector<int>&, const std::vector<void*>&),
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
  MOCK_METHOD(Status, SwapIn, (int, int, int), (override));
  MOCK_METHOD(Status, SwapOut, (int, int, int), (override));
  MOCK_METHOD(Status, BatchSwapOut, (int, const std::vector<int>&, const std::vector<int>&), (override));
  MOCK_METHOD(Status, BatchSwapIn, (int, const std::vector<int>&, const std::vector<int>&), (override));
  MOCK_METHOD(const BlockAllocatorGroupConfig&, GetBlockAllocatorGroupConfig, (), (const, override));
};

class MockBlockAllocatorManager : public BlockAllocatorManagerInterface {
 public:
  MOCK_METHOD(void, Initialize, (const BlockAllocatorManagerConfig&), (override));
  MOCK_METHOD(std::shared_ptr<BlockAllocatorGroupInterface>, GetBlockAllocatorGroup, (int), (override));
};

// ==================== 测试夹具 ====================

class QueryCallableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_connector_ = std::make_shared<testing::NiceMock<MockConnector>>();
    mock_bam_ = std::make_shared<testing::NiceMock<MockBlockAllocatorManager>>();
    mock_group_ = std::make_shared<testing::NiceMock<MockBlockAllocatorGroup>>();
    mock_device_alloc_ = std::make_shared<testing::NiceMock<MockBlockAllocator>>();

    ON_CALL(*mock_bam_, GetBlockAllocatorGroup(testing::_))
        .WillByDefault(testing::Return(mock_group_));
    ON_CALL(*mock_group_, GetDeviceBlockAllocator(testing::_))
        .WillByDefault(testing::Return(mock_device_alloc_));

    block_config_.block_size = 4096;
    ON_CALL(*mock_group_, GetBlockAllocatorGroupConfig())
        .WillByDefault(testing::ReturnRef(block_config_));

    // 默认 GetBlockPtrs 返回有效指针
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

  GlobalCacheQueryRequest MakeRequest(int64_t req_id, int token_num = 10, int local_cached = 0) {
    GlobalCacheQueryRequest req;
    req.req_id = req_id;
    req.token_ids.resize(token_num);
    for (int i = 0; i < token_num; ++i) req.token_ids[i] = i + 100;
    req.local_cached_len = local_cached;
    req.block_size = 16;
    req.kv_cache_block_ids = {0, 1, 2};
    return req;
  }

  std::shared_ptr<testing::NiceMock<MockConnector>> mock_connector_;
  std::shared_ptr<testing::NiceMock<MockBlockAllocatorManager>> mock_bam_;
  std::shared_ptr<testing::NiceMock<MockBlockAllocatorGroup>> mock_group_;
  std::shared_ptr<testing::NiceMock<MockBlockAllocator>> mock_device_alloc_;
  BlockAllocatorGroupConfig block_config_;
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
  ON_CALL(*mock_connector_, QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                                          testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                        std::vector<int>& matched_ranges, bool, int) {
        matched_ranges = {0, 9};
        return Status();
      });

  auto req = MakeRequest(300, 10);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, QueryAndFetchFailure) {
  ON_CALL(*mock_connector_, QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                                          testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault(testing::Return(Status(RET_RUNTIME_FAILED, "timeout")));

  auto req = MakeRequest(400, 10);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, StandaloneFullHitQueriesFirstTokens) {
  ON_CALL(*mock_connector_, QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                                          testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                        std::vector<int>& matched_ranges, bool, int) {
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
  callable->SetDeviceId(3);
  callable->SetGlobalCacheRole("decode", 3000);

  // 验证 setter 不会崩溃
  EXPECT_EQ(callable->device_id_, 3);
  EXPECT_EQ(callable->role_, "decode");
  EXPECT_EQ(callable->timeout_ms_, 3000);
  EXPECT_EQ(callable->dp_group_id_, 1);
}

TEST_F(QueryCallableTest, PrefillRoleNoFirstTokenQuery) {
  callable_->SetGlobalCacheRole("prefill", 5000);

  ON_CALL(*mock_connector_, QueryAndFetch(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                                          testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault([](const std::vector<int>&, size_t, int, int, int, int, size_t, const std::vector<void*>&,
                        std::vector<int>& matched_ranges, bool, int) {
        matched_ranges = {0, 9};
        return Status();
      });

  // prefill 角色不应查询 FirstTokens
  EXPECT_CALL(*mock_connector_, QueryFirstTokens(testing::_, testing::_, testing::_)).Times(0);

  auto req = MakeRequest(600, 10, 0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

// ==================== 序列化 roundtrip 测试 ====================

TEST(QueryRequestSerializationTest, RoundTrip) {
  GlobalCacheQueryRequest original;
  original.req_id = 42;
  original.token_ids = {10, 20, 30, 40, 50};
  original.local_cached_len = 3;
  original.block_size = 16;
  original.kv_cache_block_ids = {0, 1, 2, 3};

  std::vector<uint8_t> buffer(4096);
  size_t bytes = 0;
  ASSERT_TRUE(
      original.Serialize(buffer.data(), &bytes).OK());
  EXPECT_GT(bytes, 0u);

  GlobalCacheQueryRequest restored;
  ASSERT_TRUE(restored.Deserialize(buffer.data()).OK());
  EXPECT_EQ(restored.req_id, 42);
  EXPECT_EQ(restored.token_ids, original.token_ids);
  EXPECT_EQ(restored.local_cached_len, 3u);
  EXPECT_EQ(restored.block_size, 16);
  EXPECT_EQ(restored.kv_cache_block_ids,
            original.kv_cache_block_ids);
}

TEST(QueryRequestSerializationTest, EmptyVectors) {
  GlobalCacheQueryRequest original;
  original.req_id = 99;
  original.local_cached_len = 0;
  original.block_size = 0;

  std::vector<uint8_t> buffer(4096);
  size_t bytes = 0;
  ASSERT_TRUE(
      original.Serialize(buffer.data(), &bytes).OK());

  GlobalCacheQueryRequest restored;
  ASSERT_TRUE(restored.Deserialize(buffer.data()).OK());
  EXPECT_EQ(restored.req_id, 99);
  EXPECT_TRUE(restored.token_ids.empty());
  EXPECT_TRUE(restored.kv_cache_block_ids.empty());
}

// ==================== decode 角色 retry 测试 ====================

TEST_F(QueryCallableTest, DecodeRoleFullHitTokensFound) {
  callable_->SetGlobalCacheRole("decode", 5000);

  ON_CALL(*mock_connector_,
          QueryAndFetch(
              testing::_, testing::_, testing::_,
              testing::_, testing::_, testing::_,
              testing::_, testing::_, testing::_,
              testing::_, testing::_))
      .WillByDefault(
          [](const std::vector<int>&, size_t,
             int, int, int, int, size_t,
             const std::vector<void*>&,
             std::vector<int>& matched_ranges,
             bool, int) {
            matched_ranges = {0, 9};
            return Status();
          });

  // decode 全命中后会循环重试 QueryFirstTokens
  EXPECT_CALL(*mock_connector_,
              QueryFirstTokens(
                  testing::_, testing::_, testing::_))
      .WillOnce(
          [](const std::vector<int>&,
             std::vector<int>& tokens, bool& found) {
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

  ON_CALL(*mock_connector_,
          QueryAndFetch(
              testing::_, testing::_, testing::_,
              testing::_, testing::_, testing::_,
              testing::_, testing::_, testing::_,
              testing::_, testing::_))
      .WillByDefault(
          [](const std::vector<int>&, size_t,
             int, int, int, int, size_t,
             const std::vector<void*>&,
             std::vector<int>& matched_ranges,
             bool, int) {
            matched_ranges = {0, 9};
            return Status();
          });

  // 始终找不到 token，应超时退出
  ON_CALL(*mock_connector_,
          QueryFirstTokens(
              testing::_, testing::_, testing::_))
      .WillByDefault(
          [](const std::vector<int>&,
             std::vector<int>&, bool& found) {
            found = false;
            return Status();
          });

  auto start = std::chrono::steady_clock::now();
  auto req = MakeRequest(800, 10, 0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start);
  EXPECT_GE(elapsed.count(), 40);
}

TEST_F(QueryCallableTest, DecodeRolePartialHitNoTokenQuery) {
  callable_->SetGlobalCacheRole("decode", 5000);

  ON_CALL(*mock_connector_,
          QueryAndFetch(
              testing::_, testing::_, testing::_,
              testing::_, testing::_, testing::_,
              testing::_, testing::_, testing::_,
              testing::_, testing::_))
      .WillByDefault(
          [](const std::vector<int>&, size_t,
             int, int, int, int, size_t,
             const std::vector<void*>&,
             std::vector<int>& matched_ranges,
             bool, int) {
            // 只命中部分
            matched_ranges = {0, 4};
            return Status();
          });

  // 未全命中时不应查询 token
  EXPECT_CALL(*mock_connector_,
              QueryFirstTokens(
                  testing::_, testing::_, testing::_))
      .Times(0);

  auto req = MakeRequest(900, 10, 0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryCallableTest, NonDevice0NoFirstTokenQuery) {
  callable_->SetDeviceId(1);

  ON_CALL(*mock_connector_,
          QueryAndFetch(
              testing::_, testing::_, testing::_,
              testing::_, testing::_, testing::_,
              testing::_, testing::_, testing::_,
              testing::_, testing::_))
      .WillByDefault(
          [](const std::vector<int>&, size_t,
             int, int, int, int, size_t,
             const std::vector<void*>&,
             std::vector<int>& matched_ranges,
             bool, int) {
            matched_ranges = {0, 9};
            return Status();
          });

  // 非设备 0 不查询 token
  EXPECT_CALL(*mock_connector_,
              QueryFirstTokens(
                  testing::_, testing::_, testing::_))
      .Times(0);

  auto req = MakeRequest(1000, 10, 0);
  GlobalCacheQueryResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
}

}  // namespace
}  // namespace ksana_llm
