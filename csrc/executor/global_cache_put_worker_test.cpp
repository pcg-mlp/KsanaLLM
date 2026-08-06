/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <thread>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_interface.h"
#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "executor/global_cache_put_worker.h"
#include "global_cache_connector/global_cache_connector_interface.h"
#include "loguru.hpp"
#include "runtime/executor_infer_request.h"
#include "runtime/infer_stage.h"
#include "utils/ret_code.h"

namespace ksana_llm {
namespace {

class MaxLogVerbosityEnv : public ::testing::Environment {
 public:
  void SetUp() override { loguru::g_stderr_verbosity = loguru::Verbosity_MAX; }
};
static auto* unused_log_env = ::testing::AddGlobalTestEnvironment(new MaxLogVerbosityEnv());

class MockPutConnector : public GlobalCacheConnectorInterface {
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

std::shared_ptr<ExecutorInferRequest> MakeExecReq(int64_t req_id, InferStage stage) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = req_id;
  req->infer_stage = stage;
  req->block_token_num = 16;
  req->forwarding_tokens = {1, 2, 3, 4, 5};
  req->input_tokens = {1, 2, 3, 4, 5};
  return req;
}

// ==================== 基本生命周期 ====================

TEST(PutWorkerTest, ConstructStartStop) {
  auto connector = std::make_shared<testing::NiceMock<MockPutConnector>>();
  GlobalCachePutWorker worker(connector, nullptr, 0, 0, nullptr);
  worker.Start();
  worker.Stop();
}

TEST(PutWorkerTest, DestructorCallsStop) {
  auto connector = std::make_shared<testing::NiceMock<MockPutConnector>>();
  auto worker = std::make_unique<GlobalCachePutWorker>(connector, nullptr, 0, 0, nullptr);
  worker->Start();
  // 析构时自动调用 Stop，不应崩溃
  worker.reset();
}

// ==================== SubmitBatchPut early-return 路径 ====================

TEST(PutWorkerTest, SubmitBatchPutNullConnector) {
  GlobalCachePutWorker worker(nullptr, nullptr, 0, 0, nullptr);
  worker.Start();

  auto req = MakeExecReq(100, InferStage::kContext);
  // 不应崩溃
  worker.SubmitBatchPut({req});
  worker.Stop();
}

TEST(PutWorkerTest, SubmitBatchPutAfterStop) {
  auto connector = std::make_shared<testing::NiceMock<MockPutConnector>>();
  GlobalCachePutWorker worker(connector, nullptr, 0, 0, nullptr);
  worker.Start();
  worker.Stop();

  auto req = MakeExecReq(200, InferStage::kContext);
  // terminated=true，应直接返回
  worker.SubmitBatchPut({req});
}

TEST(PutWorkerTest, SubmitBatchPutFiltersDecodeRequests) {
  auto connector = std::make_shared<testing::NiceMock<MockPutConnector>>();
  // 不应调用 PutGlobalCacheBlocks（因为 block_allocator_manager 为空，会在 PutRequestKVCache 中 early return）
  EXPECT_CALL(*connector, PutGlobalCacheBlocks(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_,
                                               testing::_, testing::_))
      .Times(0);

  GlobalCachePutWorker worker(connector, nullptr, 0, 0, nullptr);
  worker.Start();

  // 只有 kDecode 请求
  auto req = MakeExecReq(300, InferStage::kDecode);
  worker.SubmitBatchPut({req});

  // 等待线程池空闲
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  worker.Stop();
}

TEST(PutWorkerTest, SubmitBatchPutEmptyBatch) {
  auto connector = std::make_shared<testing::NiceMock<MockPutConnector>>();
  GlobalCachePutWorker worker(connector, nullptr, 0, 0, nullptr);
  worker.Start();
  worker.SubmitBatchPut({});
  worker.Stop();
}

// ==================== SubmitFirstTokensPut early-return 路径 ====================

TEST(PutWorkerTest, FirstTokensPutNullConnector) {
  GlobalCachePutWorker worker(nullptr, nullptr, 0, 0, nullptr);
  worker.Start();
  auto req = MakeExecReq(400, InferStage::kContext);
  worker.SubmitFirstTokensPut({req});
  worker.Stop();
}

TEST(PutWorkerTest, FirstTokensPutAfterStop) {
  auto connector = std::make_shared<testing::NiceMock<MockPutConnector>>();
  GlobalCachePutWorker worker(connector, nullptr, 0, 0, nullptr);
  worker.Start();
  worker.Stop();
  auto req = MakeExecReq(500, InferStage::kContext);
  worker.SubmitFirstTokensPut({req});
}

TEST(PutWorkerTest, FirstTokensPutSkippedForStandaloneRole) {
  auto connector = std::make_shared<testing::NiceMock<MockPutConnector>>();
  // standalone 角色（role=""）不保存 token
  EXPECT_CALL(*connector, PutFirstTokens(testing::_, testing::_)).Times(0);

  GlobalCachePutWorker worker(connector, nullptr, 0, 0, nullptr, "");
  worker.Start();

  auto req = MakeExecReq(600, InferStage::kContext);
  req->generated_tokens = {42};
  worker.SubmitFirstTokensPut({req});

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  worker.Stop();
}

TEST(PutWorkerTest, FirstTokensPutSkippedForDecodeRole) {
  auto connector = std::make_shared<testing::NiceMock<MockPutConnector>>();
  EXPECT_CALL(*connector, PutFirstTokens(testing::_, testing::_)).Times(0);

  GlobalCachePutWorker worker(connector, nullptr, 0, 0, nullptr, "decode");
  worker.Start();

  auto req = MakeExecReq(700, InferStage::kContext);
  req->generated_tokens = {42};
  worker.SubmitFirstTokensPut({req});

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  worker.Stop();
}

TEST(PutWorkerTest, FirstTokensPutSkippedForNonZeroDevice) {
  auto connector = std::make_shared<testing::NiceMock<MockPutConnector>>();
  EXPECT_CALL(*connector, PutFirstTokens(testing::_, testing::_)).Times(0);

  // prefill 角色但 device_id=1（仅卡0保存）
  GlobalCachePutWorker worker(connector, nullptr, 0, 1, nullptr, "prefill");
  worker.Start();

  auto req = MakeExecReq(800, InferStage::kContext);
  req->generated_tokens = {42};
  worker.SubmitFirstTokensPut({req});

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  worker.Stop();
}

TEST(PutWorkerTest, FirstTokensPutSkipsEmptyGenerated) {
  auto connector = std::make_shared<testing::NiceMock<MockPutConnector>>();
  EXPECT_CALL(*connector, PutFirstTokens(testing::_, testing::_)).Times(0);

  GlobalCachePutWorker worker(connector, nullptr, 0, 0, nullptr, "prefill");
  worker.Start();

  auto req = MakeExecReq(900, InferStage::kContext);
  req->generated_tokens.clear();
  worker.SubmitFirstTokensPut({req});

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  worker.Stop();
}

// ==================== 深层路径 mock 类 ====================

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

// ==================== 深层路径测试 fixture ====================
// 通过 BlockAllocatorGroupCreationFunc 注入 mock group，避免修改正式代码。

class PutWorkerDeepTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_connector_ = std::make_shared<testing::NiceMock<MockPutConnector>>();
    mock_group_ = std::make_shared<testing::NiceMock<MockBlockAllocatorGroup>>();
    mock_dev_alloc_ = std::make_shared<testing::NiceMock<MockBlockAllocator>>();

    group_config_.block_size = 4096;
    group_config_.devices = {0};

    ON_CALL(*mock_group_, GetBlockAllocatorGroupConfig()).WillByDefault(testing::ReturnRef(group_config_));
    ON_CALL(*mock_group_, GetDeviceBlockAllocator(0)).WillByDefault(testing::Return(mock_dev_alloc_));

    fake_ptrs_ = {&fake_mem_[0], &fake_mem_[1], &fake_mem_[2]};
    ON_CALL(*mock_dev_alloc_, GetBlockPtrs(testing::_, testing::_))
        .WillByDefault(testing::DoAll(testing::SetArgReferee<1>(fake_ptrs_), testing::Return(Status())));

    BlockAllocatorManagerConfig bam_config;
    bam_config[0] = group_config_;

    auto captured_group = mock_group_;
    bam_ = std::make_shared<BlockAllocatorManager>(
        /*is_engine=*/true, bam_config, /*memory_allocator=*/nullptr, /*context=*/nullptr,
        [](MemoryLocation, size_t, size_t, int, std::shared_ptr<MemoryAllocatorInterface>,
           std::shared_ptr<Context>) -> std::shared_ptr<BlockAllocatorInterface> { return nullptr; },
        [captured_group](
            const BlockAllocatorGroupConfig&, std::shared_ptr<MemoryAllocatorInterface>, std::shared_ptr<Context>,
            BlockAllocatorCreationFunc) -> std::shared_ptr<BlockAllocatorGroupInterface> { return captured_group; });
  }

  std::shared_ptr<ExecutorInferRequest> MakeDeepReq(int64_t req_id, InferStage stage,
                                                    const std::vector<int>& tokens = {1, 2, 3, 4, 5}) {
    auto req = std::make_shared<ExecutorInferRequest>();
    req->req_id = req_id;
    req->infer_stage = stage;
    req->block_token_num = 16;
    req->forwarding_tokens = tokens;
    req->input_tokens = tokens;
    req->kv_cache_blocks = {0, 1, 2};
    return req;
  }

  std::shared_ptr<testing::NiceMock<MockPutConnector>> mock_connector_;
  std::shared_ptr<testing::NiceMock<MockBlockAllocatorGroup>> mock_group_;
  std::shared_ptr<testing::NiceMock<MockBlockAllocator>> mock_dev_alloc_;
  std::shared_ptr<BlockAllocatorManager> bam_;
  BlockAllocatorGroupConfig group_config_;
  int fake_mem_[16] = {};
  std::vector<void*> fake_ptrs_;
};

// ---- SubmitBatchPut → PutRequestKVCache 完整路径 ----

TEST_F(PutWorkerDeepTest, BatchPutContextRequestCallsPutBlocks) {
  EXPECT_CALL(*mock_connector_, PutGlobalCacheBlocks(std::vector<int>{1, 2, 3, 4, 5}, /*tp_size=*/1, /*tp_rank=*/0,
                                                     /*layer_index=*/0, /*block_token_num=*/16,
                                                     /*block_byte_size=*/4096, std::vector<int>{0, 4}, testing::_))
      .WillOnce(testing::Return(Status()));

  GlobalCachePutWorker worker(mock_connector_, bam_, 0, 0, nullptr);
  worker.Start();
  worker.SubmitBatchPut({MakeDeepReq(100, InferStage::kContext)});
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  worker.Stop();
}

TEST_F(PutWorkerDeepTest, BatchPutConnectorFailureDoesNotCrash) {
  EXPECT_CALL(*mock_connector_, PutGlobalCacheBlocks(testing::_, testing::_, testing::_, testing::_, testing::_,
                                                     testing::_, testing::_, testing::_))
      .WillOnce(testing::Return(Status(RET_RUNTIME_FAILED, "put failed")));

  GlobalCachePutWorker worker(mock_connector_, bam_, 0, 0, nullptr);
  worker.Start();
  worker.SubmitBatchPut({MakeDeepReq(200, InferStage::kContext)});
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  worker.Stop();
}

TEST_F(PutWorkerDeepTest, BatchPutMixedStagesOnlyProcessesContext) {
  EXPECT_CALL(*mock_connector_, PutGlobalCacheBlocks(testing::_, testing::_, testing::_, testing::_, testing::_,
                                                     testing::_, testing::_, testing::_))
      .Times(1)
      .WillOnce(testing::Return(Status()));

  GlobalCachePutWorker worker(mock_connector_, bam_, 0, 0, nullptr);
  worker.Start();
  worker.SubmitBatchPut({MakeDeepReq(301, InferStage::kDecode), MakeDeepReq(302, InferStage::kContext)});
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  worker.Stop();
}

// ---- PutRequestKVCache 各 early-return 分支 ----

TEST_F(PutWorkerDeepTest, PutKVCacheNullGroupReturnsEarly) {
  EXPECT_CALL(*mock_connector_, PutGlobalCacheBlocks(testing::_, testing::_, testing::_, testing::_, testing::_,
                                                     testing::_, testing::_, testing::_))
      .Times(0);

  // dp_group_id=99 不在 bam_ 中 → GetBlockAllocatorGroup 返回 nullptr
  GlobalCachePutWorker worker(mock_connector_, bam_, /*dp_group_id=*/99, 0, nullptr);
  worker.Start();
  worker.SubmitBatchPut({MakeDeepReq(300, InferStage::kContext)});
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  worker.Stop();
}

TEST_F(PutWorkerDeepTest, PutKVCacheNullDeviceAllocatorReturnsEarly) {
  // 覆盖默认 mock：device_id=0 时返回 nullptr
  ON_CALL(*mock_group_, GetDeviceBlockAllocator(0)).WillByDefault(testing::Return(nullptr));
  EXPECT_CALL(*mock_connector_, PutGlobalCacheBlocks(testing::_, testing::_, testing::_, testing::_, testing::_,
                                                     testing::_, testing::_, testing::_))
      .Times(0);

  GlobalCachePutWorker worker(mock_connector_, bam_, 0, /*device_id=*/0, nullptr);
  worker.Start();
  worker.SubmitBatchPut({MakeDeepReq(400, InferStage::kContext)});
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  worker.Stop();
}

TEST_F(PutWorkerDeepTest, PutKVCacheEmptyBlockPtrsReturnsEarly) {
  ON_CALL(*mock_dev_alloc_, GetBlockPtrs(testing::_, testing::_))
      .WillByDefault(testing::DoAll(testing::SetArgReferee<1>(std::vector<void*>{}), testing::Return(Status())));
  EXPECT_CALL(*mock_connector_, PutGlobalCacheBlocks(testing::_, testing::_, testing::_, testing::_, testing::_,
                                                     testing::_, testing::_, testing::_))
      .Times(0);

  GlobalCachePutWorker worker(mock_connector_, bam_, 0, 0, nullptr);
  worker.Start();
  worker.SubmitBatchPut({MakeDeepReq(500, InferStage::kContext)});
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  worker.Stop();
}

TEST_F(PutWorkerDeepTest, PutKVCacheEmptyTokensReturnsEarly) {
  EXPECT_CALL(*mock_connector_, PutGlobalCacheBlocks(testing::_, testing::_, testing::_, testing::_, testing::_,
                                                     testing::_, testing::_, testing::_))
      .Times(0);

  GlobalCachePutWorker worker(mock_connector_, bam_, 0, 0, nullptr);
  worker.Start();
  worker.SubmitBatchPut({MakeDeepReq(600, InferStage::kContext, /*tokens=*/{})});
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  worker.Stop();
}

// ---- SubmitFirstTokensPut 深层路径 ----

TEST_F(PutWorkerDeepTest, FirstTokensPutPrefillCallsPutFirstTokens) {
  EXPECT_CALL(*mock_connector_, PutFirstTokens(std::vector<int>{1, 2, 3, 4, 5}, std::vector<int>{42}))
      .WillOnce(testing::Return(Status()));

  GlobalCachePutWorker worker(mock_connector_, bam_, 0, 0, nullptr, "prefill");
  worker.Start();
  auto req = MakeDeepReq(700, InferStage::kContext);
  req->generated_tokens = {42};
  worker.SubmitFirstTokensPut({req});
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  worker.Stop();
}

TEST_F(PutWorkerDeepTest, FirstTokensPutMergesGeneratedAndDraft) {
  // generated=[42] + draft.mtp=[10,11] → tokens_to_save=[42, 10, 11]
  EXPECT_CALL(*mock_connector_, PutFirstTokens(std::vector<int>{1, 2, 3, 4, 5}, std::vector<int>{42, 10, 11}))
      .WillOnce(testing::Return(Status()));

  GlobalCachePutWorker worker(mock_connector_, bam_, 0, 0, nullptr, "prefill");
  worker.Start();
  auto req = MakeDeepReq(800, InferStage::kContext);
  req->generated_tokens = {42};
  req->spec_state.draft_tokens.mtp = {10, 11};
  worker.SubmitFirstTokensPut({req});
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  worker.Stop();
}

TEST_F(PutWorkerDeepTest, FirstTokensPutConnectorFailureDoesNotCrash) {
  EXPECT_CALL(*mock_connector_, PutFirstTokens(testing::_, testing::_))
      .WillOnce(testing::Return(Status(RET_RUNTIME_FAILED, "put first tokens failed")));

  GlobalCachePutWorker worker(mock_connector_, bam_, 0, 0, nullptr, "prefill");
  worker.Start();
  auto req = MakeDeepReq(900, InferStage::kContext);
  req->generated_tokens = {42};
  worker.SubmitFirstTokensPut({req});
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  worker.Stop();
}

}  // namespace
}  // namespace ksana_llm
