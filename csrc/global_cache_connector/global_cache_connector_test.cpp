/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "global_cache_connector/global_cache_connector_factory.h"
#include "global_cache_connector/global_cache_key_utils.h"
#include "global_cache_connector/mooncake_mock_connector.h"
#include "global_cache_connector/staging_buffer_pool.h"

namespace ksana_llm {

// GPU 内存 RAII 管理，自动分配和释放
class GpuBuffer {
 public:
  explicit GpuBuffer(size_t byte_size) : byte_size_(byte_size) { cudaMalloc(&ptr_, byte_size_); }
  ~GpuBuffer() {
    if (ptr_) cudaFree(ptr_);
  }
  void* ptr() const { return ptr_; }
  size_t size() const { return byte_size_; }

  void Upload(const void* host_data) { cudaMemcpy(ptr_, host_data, byte_size_, cudaMemcpyHostToDevice); }
  void Download(void* host_data) const { cudaMemcpy(host_data, ptr_, byte_size_, cudaMemcpyDeviceToHost); }
  void Clear() { cudaMemset(ptr_, 0, byte_size_); }

 private:
  void* ptr_ = nullptr;
  size_t byte_size_;
};

// 创建临时目录用作 MockStore 的 metadata_server (endpoint)
class TempDir {
 public:
  TempDir() {
    char tmpl[] = "/tmp/gcc_test_XXXXXX";
    path_ = mkdtemp(tmpl);
  }
  ~TempDir() { std::filesystem::remove_all(path_); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// ==================== Config 测试 ====================

TEST(ConfigTest, RoleHelpers) {
  GlobalCacheConnectorConfig config;
  EXPECT_TRUE(config.IsStandalone());
  EXPECT_FALSE(config.IsDecodeRole());
  EXPECT_FALSE(config.IsPrefillRole());

  config.role = "decode";
  EXPECT_FALSE(config.IsStandalone());
  EXPECT_TRUE(config.IsDecodeRole());
  EXPECT_FALSE(config.IsPrefillRole());

  config.role = "prefill";
  EXPECT_FALSE(config.IsStandalone());
  EXPECT_FALSE(config.IsDecodeRole());
  EXPECT_TRUE(config.IsPrefillRole());
}

TEST(ConfigTest, ToStringContainsAllFields) {
  GlobalCacheConnectorConfig config;
  config.backend_type = "mooncake-mock";
  config.service_endpoint = "http://test:8080";
  config.device_id = 3;
  config.device_count = 8;
  config.timeout_ms = 3000;
  config.max_retry = 5;
  config.fetch_thread_num = 2;
  config.role = "decode";
  config.protocol = "rdma";
  config.rdma_devices = "mlx5_0,mlx5_1";
  config.master_server_addr = "10.0.0.1:50051";

  std::string s = config.ToString();
  EXPECT_NE(s.find("mooncake-mock"), std::string::npos);
  EXPECT_NE(s.find("http://test:8080"), std::string::npos);
  EXPECT_NE(s.find("3"), std::string::npos);
  EXPECT_NE(s.find("8"), std::string::npos);
  EXPECT_NE(s.find("3000"), std::string::npos);
  EXPECT_NE(s.find("5"), std::string::npos);
  EXPECT_NE(s.find("decode"), std::string::npos);
  EXPECT_NE(s.find("rdma"), std::string::npos);
  EXPECT_NE(s.find("mlx5_0,mlx5_1"), std::string::npos);
  EXPECT_NE(s.find("10.0.0.1:50051"), std::string::npos);
}

TEST(ConfigTest, ToStringStandaloneShowsPlaceholder) {
  GlobalCacheConnectorConfig config;
  config.role = "";
  std::string s = config.ToString();
  EXPECT_NE(s.find("(standalone)"), std::string::npos);
}

TEST(ConfigTest, DefaultValues) {
  GlobalCacheConnectorConfig config;
  EXPECT_EQ(config.backend_type, "mooncake-mock");
  EXPECT_EQ(config.device_id, 0);
  EXPECT_EQ(config.device_count, 1);
  EXPECT_EQ(config.timeout_ms, 5000);
  EXPECT_EQ(config.max_retry, 3);
  EXPECT_EQ(config.fetch_thread_num, 4);
  EXPECT_EQ(config.protocol, "tcp");
  EXPECT_TRUE(config.rdma_devices.empty());
  EXPECT_EQ(config.global_segment_size, 16u * 1024 * 1024);
  EXPECT_EQ(config.local_buffer_size, 16u * 1024 * 1024);
}

// ==================== Factory 测试 ====================

class FactoryTest : public ::testing::Test {
 protected:
  TempDir tmpdir_;
};

TEST_F(FactoryTest, CreateMockConnectorSucceeds) {
  GlobalCacheConnectorConfig config;
  config.backend_type = "mooncake-mock";
  config.service_endpoint = tmpdir_.path();
  auto connector = GlobalCacheConnectorFactory::CreateConnector(config);
  ASSERT_NE(connector, nullptr);
}

TEST_F(FactoryTest, CreateRealConnectorReturnsNull) {
  GlobalCacheConnectorConfig config;
  config.backend_type = "mooncake";
  config.service_endpoint = "http://127.0.0.1:8080";
  auto connector = GlobalCacheConnectorFactory::CreateConnector(config);
  EXPECT_EQ(connector, nullptr);
}

TEST_F(FactoryTest, UnknownBackendReturnsNull) {
  GlobalCacheConnectorConfig config;
  config.backend_type = "unknown-backend";
  auto connector = GlobalCacheConnectorFactory::CreateConnector(config);
  EXPECT_EQ(connector, nullptr);
}

// ==================== MooncakeMockConnector 测试 ====================

class MockConnectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    connector_ = std::make_unique<MooncakeMockConnector>();
    GlobalCacheConnectorConfig config;
    config.backend_type = "mooncake-mock";
    config.service_endpoint = tmpdir_.path();
    ASSERT_TRUE(connector_->Initialize(config).OK());
  }

  void TearDown() override {
    if (connector_) connector_->Shutdown();
  }

  TempDir tmpdir_;
  std::unique_ptr<MooncakeMockConnector> connector_;

  static constexpr int kBlockSize = 4;
  static constexpr size_t kBlockByteSize = 1024;
  static constexpr int kTpSize = 1;
  static constexpr int kTpRank = 0;
  static constexpr int kLayerIndex = 0;
};

TEST_F(MockConnectorTest, PutAndQueryRoundTrip) {
  std::vector<int> token_ids = {10, 20, 30, 40, 50, 60, 70, 80};
  int num_blocks = static_cast<int>(token_ids.size()) / kBlockSize;

  // 分配 GPU 内存并填充测试数据
  std::vector<std::unique_ptr<GpuBuffer>> gpu_blocks;
  std::vector<void*> gpu_ptrs;
  std::vector<std::vector<uint8_t>> original_data(num_blocks, std::vector<uint8_t>(kBlockByteSize));

  for (int i = 0; i < num_blocks; ++i) {
    gpu_blocks.emplace_back(std::make_unique<GpuBuffer>(kBlockByteSize));
    gpu_ptrs.push_back(gpu_blocks.back()->ptr());
    std::fill(original_data[i].begin(), original_data[i].end(), static_cast<uint8_t>(0xA0 + i));
    gpu_blocks[i]->Upload(original_data[i].data());
  }

  // Put
  std::vector<int> block_ranges = {0, static_cast<int>(token_ids.size()) - 1};
  Status status = connector_->PutGlobalCacheBlocks(token_ids, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                                   block_ranges, gpu_ptrs);
  ASSERT_TRUE(status.OK()) << status.GetMessage();

  // 清空 GPU 数据
  for (auto& buf : gpu_blocks) buf->Clear();

  // Query
  std::vector<int> matched_ranges;
  status = connector_->QueryAndFetch(token_ids, 0, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize, gpu_ptrs,
                                     matched_ranges);
  ASSERT_TRUE(status.OK()) << status.GetMessage();
  EXPECT_FALSE(matched_ranges.empty()) << "expected hits but got none";

  // 验证数据一致性
  for (int i = 0; i < num_blocks; ++i) {
    std::vector<uint8_t> readback(kBlockByteSize);
    gpu_blocks[i]->Download(readback.data());
    EXPECT_EQ(readback, original_data[i]) << "block " << i << " data mismatch";
  }
}

TEST_F(MockConnectorTest, QueryMissReturnsEmpty) {
  std::vector<int> token_ids = {1, 2, 3, 4};
  GpuBuffer buf(kBlockByteSize);
  std::vector<void*> ptrs = {buf.ptr()};
  std::vector<int> matched_ranges;

  Status status = connector_->QueryAndFetch(token_ids, 0, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                            ptrs, matched_ranges);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(matched_ranges.empty());
}

TEST_F(MockConnectorTest, PutFirstTokensAndQueryRoundTrip) {
  std::vector<int> token_ids = {100, 200, 300, 400};
  std::vector<int> tokens = {42, 99, 7, 13};

  Status status = connector_->PutFirstTokens(token_ids, tokens);
  ASSERT_TRUE(status.OK()) << status.GetMessage();

  std::vector<int> retrieved;
  bool found = false;
  status = connector_->QueryFirstTokens(token_ids, retrieved, found);
  ASSERT_TRUE(status.OK()) << status.GetMessage();
  EXPECT_TRUE(found);
  EXPECT_EQ(retrieved, tokens);
}

TEST_F(MockConnectorTest, QueryFirstTokensMissReturnsFalse) {
  std::vector<int> token_ids = {1, 2, 3};
  std::vector<int> tokens;
  bool found = true;

  Status status = connector_->QueryFirstTokens(token_ids, tokens, found);
  EXPECT_TRUE(status.OK());
  EXPECT_FALSE(found);
  EXPECT_TRUE(tokens.empty());
}

TEST_F(MockConnectorTest, PutSkipsDuplicateKeys) {
  std::vector<int> token_ids = {10, 20, 30, 40};
  GpuBuffer buf(kBlockByteSize);
  std::vector<uint8_t> data(kBlockByteSize, 0xBB);
  buf.Upload(data.data());
  std::vector<void*> ptrs = {buf.ptr()};
  std::vector<int> block_ranges = {0, static_cast<int>(token_ids.size()) - 1};

  // 第一次 Put
  ASSERT_TRUE(connector_
                  ->PutGlobalCacheBlocks(token_ids, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                         block_ranges, ptrs)
                  .OK());

  // 第二次 Put (相同 key) 应跳过但不报错
  ASSERT_TRUE(connector_
                  ->PutGlobalCacheBlocks(token_ids, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                         block_ranges, ptrs)
                  .OK());
}

TEST_F(MockConnectorTest, LocalCachedLenSkipsBlocks) {
  std::vector<int> token_ids = {10, 20, 30, 40, 50, 60, 70, 80};
  int num_blocks = 2;
  std::vector<std::unique_ptr<GpuBuffer>> gpu_blocks;
  std::vector<void*> gpu_ptrs;

  for (int i = 0; i < num_blocks; ++i) {
    gpu_blocks.emplace_back(std::make_unique<GpuBuffer>(kBlockByteSize));
    gpu_ptrs.push_back(gpu_blocks.back()->ptr());
    std::vector<uint8_t> data(kBlockByteSize, static_cast<uint8_t>(0xC0 + i));
    gpu_blocks[i]->Upload(data.data());
  }

  std::vector<int> block_ranges = {0, static_cast<int>(token_ids.size()) - 1};
  ASSERT_TRUE(connector_
                  ->PutGlobalCacheBlocks(token_ids, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                         block_ranges, gpu_ptrs)
                  .OK());

  for (auto& buf : gpu_blocks) buf->Clear();

  // local_cached_len = kBlockSize，跳过第 0 个 block，只拉取第 1 个
  std::vector<int> matched_ranges;
  Status status = connector_->QueryAndFetch(token_ids, kBlockSize, kTpSize, kTpRank, kLayerIndex, kBlockSize,
                                            kBlockByteSize, gpu_ptrs, matched_ranges);
  ASSERT_TRUE(status.OK());
  EXPECT_FALSE(matched_ranges.empty());

  // block 0 应该仍是清零状态（未拉取）
  std::vector<uint8_t> block0(kBlockByteSize, 0);
  std::vector<uint8_t> readback(kBlockByteSize);
  gpu_blocks[0]->Download(readback.data());
  EXPECT_EQ(readback, block0);

  // block 1 应该被拉取回来
  std::vector<uint8_t> expected(kBlockByteSize, 0xC1);
  gpu_blocks[1]->Download(readback.data());
  EXPECT_EQ(readback, expected);
}

TEST_F(MockConnectorTest, InvalidBlockSizeSkips) {
  std::vector<int> token_ids = {1, 2, 3, 4};
  GpuBuffer buf(kBlockByteSize);
  std::vector<void*> ptrs = {buf.ptr()};
  std::vector<int> matched_ranges;

  // block_size = 0 应直接跳过
  Status status = connector_->QueryAndFetch(token_ids, 0, kTpSize, kTpRank, kLayerIndex, /*block_size=*/0,
                                            kBlockByteSize, ptrs, matched_ranges);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(matched_ranges.empty());
}

TEST_F(MockConnectorTest, ShutdownAndReinitialize) {
  ASSERT_TRUE(connector_->Shutdown().OK());

  GlobalCacheConnectorConfig config;
  config.backend_type = "mooncake-mock";
  config.service_endpoint = tmpdir_.path();
  ASSERT_TRUE(connector_->Initialize(config).OK());
}

TEST_F(MockConnectorTest, UninitializedQuerySkips) {
  auto fresh = std::make_unique<MooncakeMockConnector>();
  std::vector<int> ids = {1, 2, 3, 4};
  GpuBuffer buf(kBlockByteSize);
  std::vector<void*> ptrs = {buf.ptr()};
  std::vector<int> matched;

  // 未初始化时 QueryAndFetch 应返回成功但 matched 为空
  Status s = fresh->QueryAndFetch(ids, 0, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize, ptrs, matched);
  EXPECT_TRUE(s.OK());
  EXPECT_TRUE(matched.empty());
}

TEST_F(MockConnectorTest, UninitializedPutReturnsError) {
  auto fresh = std::make_unique<MooncakeMockConnector>();
  std::vector<int> ids = {1, 2, 3, 4};
  GpuBuffer buf(kBlockByteSize);
  std::vector<void*> ptrs = {buf.ptr()};
  std::vector<int> ranges = {0, 3};

  Status s = fresh->PutGlobalCacheBlocks(ids, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize, ranges, ptrs);
  EXPECT_FALSE(s.OK());
}

TEST_F(MockConnectorTest, DoubleInitializeReturnsError) {
  GlobalCacheConnectorConfig config;
  config.backend_type = "mooncake-mock";
  config.service_endpoint = tmpdir_.path();
  Status s = connector_->Initialize(config);
  EXPECT_FALSE(s.OK());
}

TEST_F(MockConnectorTest, PutOddBlockRangesReturnsError) {
  std::vector<int> ids = {1, 2, 3, 4};
  GpuBuffer buf(kBlockByteSize);
  std::vector<void*> ptrs = {buf.ptr()};
  // block_ranges 大小为奇数，应返回错误
  std::vector<int> odd_ranges = {0, 3, 5};

  Status s = connector_->PutGlobalCacheBlocks(ids, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                              odd_ranges, ptrs);
  EXPECT_FALSE(s.OK());
}

TEST_F(MockConnectorTest, PutZeroBlockByteSizeReturnsError) {
  std::vector<int> ids = {1, 2, 3, 4};
  GpuBuffer buf(kBlockByteSize);
  std::vector<void*> ptrs = {buf.ptr()};
  std::vector<int> ranges = {0, 3};

  Status s = connector_->PutGlobalCacheBlocks(ids, kTpSize, kTpRank, kLayerIndex, kBlockSize,
                                              /*block_byte_size=*/0, ranges, ptrs);
  EXPECT_FALSE(s.OK());
}

TEST_F(MockConnectorTest, PutFirstTokensEmptyInputNoOp) {
  std::vector<int> empty_ids;
  std::vector<int> tokens = {42};
  EXPECT_TRUE(connector_->PutFirstTokens(empty_ids, tokens).OK());

  std::vector<int> ids = {1, 2};
  std::vector<int> empty_tokens;
  EXPECT_TRUE(connector_->PutFirstTokens(ids, empty_tokens).OK());
}

TEST_F(MockConnectorTest, MultiLayerPutAndQuery) {
  std::vector<int> token_ids = {10, 20, 30, 40};
  const int kLayer0 = 0;
  const int kLayer1 = 1;

  // 在不同 layer 写入不同数据
  GpuBuffer buf0(kBlockByteSize);
  GpuBuffer buf1(kBlockByteSize);
  std::vector<uint8_t> data0(kBlockByteSize, 0xAA);
  std::vector<uint8_t> data1(kBlockByteSize, 0xBB);
  buf0.Upload(data0.data());
  buf1.Upload(data1.data());

  std::vector<int> ranges = {0, 3};
  ASSERT_TRUE(
      connector_
          ->PutGlobalCacheBlocks(token_ids, kTpSize, kTpRank, kLayer0, kBlockSize, kBlockByteSize, ranges, {buf0.ptr()})
          .OK());
  ASSERT_TRUE(
      connector_
          ->PutGlobalCacheBlocks(token_ids, kTpSize, kTpRank, kLayer1, kBlockSize, kBlockByteSize, ranges, {buf1.ptr()})
          .OK());

  // 清空并 Query layer0
  buf0.Clear();
  buf1.Clear();
  std::vector<int> matched;
  ASSERT_TRUE(
      connector_
          ->QueryAndFetch(token_ids, 0, kTpSize, kTpRank, kLayer0, kBlockSize, kBlockByteSize, {buf0.ptr()}, matched)
          .OK());
  EXPECT_FALSE(matched.empty());
  std::vector<uint8_t> readback(kBlockByteSize);
  buf0.Download(readback.data());
  EXPECT_EQ(readback, data0);

  // Query layer1
  matched.clear();
  ASSERT_TRUE(
      connector_
          ->QueryAndFetch(token_ids, 0, kTpSize, kTpRank, kLayer1, kBlockSize, kBlockByteSize, {buf1.ptr()}, matched)
          .OK());
  EXPECT_FALSE(matched.empty());
  buf1.Download(readback.data());
  EXPECT_EQ(readback, data1);
}

TEST_F(MockConnectorTest, PutFirstTokensDuplicateSkips) {
  std::vector<int> ids = {100, 200, 300};
  std::vector<int> tokens1 = {42, 99};
  std::vector<int> tokens2 = {77, 88};

  ASSERT_TRUE(connector_->PutFirstTokens(ids, tokens1).OK());
  // 重复写入不报错
  ASSERT_TRUE(connector_->PutFirstTokens(ids, tokens2).OK());

  // 查询应该返回第一次写入的值
  std::vector<int> retrieved;
  bool found = false;
  ASSERT_TRUE(connector_->QueryFirstTokens(ids, retrieved, found).OK());
  EXPECT_TRUE(found);
  EXPECT_EQ(retrieved, tokens1);
}

// ==================== MooncakeMockConnector 扩展覆盖率测试 ====================

TEST_F(MockConnectorTest, PartialBlockMissProducesNonContiguousRanges) {
  // 写入 3 个 block（token 0-11），但只写 block 0 和 block 2（跳过 block 1）
  std::vector<int> token_ids = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
  const int num_blocks = 3;

  std::vector<std::unique_ptr<GpuBuffer>> gpu_blocks;
  std::vector<void*> gpu_ptrs;
  for (int i = 0; i < num_blocks; ++i) {
    gpu_blocks.emplace_back(std::make_unique<GpuBuffer>(kBlockByteSize));
    gpu_ptrs.push_back(gpu_blocks.back()->ptr());
    std::vector<uint8_t> data(kBlockByteSize, static_cast<uint8_t>(0xD0 + i));
    gpu_blocks[i]->Upload(data.data());
  }

  // 只 Put block 0
  std::vector<int> range0 = {0, kBlockSize - 1};
  ASSERT_TRUE(connector_
                  ->PutGlobalCacheBlocks(token_ids, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize, range0,
                                         {gpu_ptrs[0]})
                  .OK());

  // 只 Put block 2
  std::vector<int> range2 = {2 * kBlockSize, 3 * kBlockSize - 1};
  ASSERT_TRUE(connector_
                  ->PutGlobalCacheBlocks(token_ids, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize, range2,
                                         {gpu_ptrs[2]})
                  .OK());

  // 清空 GPU 并 Query 全部
  for (auto& buf : gpu_blocks) buf->Clear();
  std::vector<int> matched_ranges;
  ASSERT_TRUE(connector_
                  ->QueryAndFetch(token_ids, 0, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize, gpu_ptrs,
                                  matched_ranges)
                  .OK());

  // 应该有 2 个不连续的区间（block 0 和 block 2）
  ASSERT_EQ(matched_ranges.size(), 4u) << "expected 2 ranges (4 ints)";
}

TEST_F(MockConnectorTest, QueryFirstTokensEmptyIds) {
  std::vector<int> empty_ids;
  std::vector<int> tokens;
  bool found = true;
  EXPECT_TRUE(connector_->QueryFirstTokens(empty_ids, tokens, found).OK());
  EXPECT_FALSE(found);
}

TEST_F(MockConnectorTest, UninitializedQueryFirstTokens) {
  auto fresh = std::make_unique<MooncakeMockConnector>();
  std::vector<int> ids = {1, 2, 3};
  std::vector<int> tokens;
  bool found = true;
  EXPECT_TRUE(fresh->QueryFirstTokens(ids, tokens, found).OK());
  EXPECT_FALSE(found);
}

TEST_F(MockConnectorTest, PutFirstTokensUninitialized) {
  auto fresh = std::make_unique<MooncakeMockConnector>();
  std::vector<int> ids = {1, 2, 3};
  std::vector<int> tokens = {42};
  EXPECT_FALSE(fresh->PutFirstTokens(ids, tokens).OK());
}

TEST_F(MockConnectorTest, RetryUntilFullHitWithImmediateHit) {
  std::vector<int> token_ids = {10, 20, 30, 40};
  GpuBuffer buf(kBlockByteSize);
  std::vector<uint8_t> data(kBlockByteSize, 0xEE);
  buf.Upload(data.data());
  std::vector<void*> ptrs = {buf.ptr()};
  std::vector<int> ranges = {0, 3};

  ASSERT_TRUE(
      connector_
          ->PutGlobalCacheBlocks(token_ids, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize, ranges, ptrs)
          .OK());

  buf.Clear();
  std::vector<int> matched_ranges;
  // retry_until_full_hit=true，但数据已存在，应立即成功
  Status status = connector_->QueryAndFetch(token_ids, 0, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                            ptrs, matched_ranges, /*retry_until_full_hit=*/true, /*timeout_ms=*/100);
  EXPECT_TRUE(status.OK());
  EXPECT_FALSE(matched_ranges.empty());
}

TEST_F(MockConnectorTest, RetryUntilFullHitTimesOut) {
  std::vector<int> token_ids = {10, 20, 30, 40};
  GpuBuffer buf(kBlockByteSize);
  std::vector<void*> ptrs = {buf.ptr()};

  // 未写入数据，retry 应超时
  std::vector<int> matched_ranges;
  auto start = std::chrono::steady_clock::now();
  Status status = connector_->QueryAndFetch(token_ids, 0, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                            ptrs, matched_ranges, /*retry_until_full_hit=*/true, /*timeout_ms=*/50);
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);

  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(matched_ranges.empty());
  EXPECT_GE(elapsed.count(), 40) << "should have waited at least ~50ms";
}

// ==================== Memcpy 异常分支覆盖测试 ====================

// 测试用子类：DoMemcpy 总是抛异常，覆盖 catch 分支
class ThrowingMockConnector : public MooncakeMockConnector {
 protected:
  void DoMemcpy(void* /*dst*/, const void* /*src*/, size_t /*count*/, int /*kind*/) override {
    throw std::runtime_error("injected Memcpy failure for testing");
  }
};

class ThrowingConnectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    connector_ = std::make_unique<ThrowingMockConnector>();
    GlobalCacheConnectorConfig config;
    config.backend_type = "mooncake-mock";
    config.service_endpoint = tmpdir_.path();
    ASSERT_TRUE(connector_->Initialize(config).OK());
  }

  void TearDown() override {
    if (connector_) connector_->Shutdown();
  }

  TempDir tmpdir_;
  std::unique_ptr<ThrowingMockConnector> connector_;

  static constexpr int kBlockSize = 4;
  static constexpr size_t kBlockByteSize = 1024;
  static constexpr int kTpSize = 1;
  static constexpr int kTpRank = 0;
  static constexpr int kLayerIndex = 0;
};

// DoMemcpy 抛异常 → PutGlobalCacheBlocks 的 D2H catch 分支（mooncake_mock_connector.cpp:228-234）
TEST_F(ThrowingConnectorTest, PutWithMemcpyFailureSkipsBlock) {
  std::vector<int> token_ids = {10, 20, 30, 40};
  // 指针值不重要，DoMemcpy 会在调用前抛异常
  void* dummy_ptr = reinterpret_cast<void*>(0x1000);
  std::vector<void*> ptrs = {dummy_ptr};
  std::vector<int> block_ranges = {0, static_cast<int>(token_ids.size()) - 1};

  // 不应崩溃，D2H 失败的 block 被跳过
  Status status = connector_->PutGlobalCacheBlocks(token_ids, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                                   block_ranges, ptrs);
  EXPECT_TRUE(status.OK());
}

// DoMemcpy 抛异常 → QueryAndFetch 的 H2D catch 分支（mooncake_mock_connector.cpp:116-120）
// 需要先用正常 connector 写入数据，再用 ThrowingMockConnector 查询
TEST_F(ThrowingConnectorTest, QueryWithMemcpyFailureSkipsBlock) {
  // 使用正常的 MooncakeMockConnector 写入数据
  auto normal_connector = std::make_unique<MooncakeMockConnector>();
  GlobalCacheConnectorConfig config;
  config.backend_type = "mooncake-mock";
  config.service_endpoint = tmpdir_.path();
  ASSERT_TRUE(normal_connector->Initialize(config).OK());

  std::vector<int> token_ids = {10, 20, 30, 40};
  GpuBuffer buf(kBlockByteSize);
  std::vector<uint8_t> data(kBlockByteSize, 0xCC);
  buf.Upload(data.data());
  std::vector<void*> put_ptrs = {buf.ptr()};
  std::vector<int> block_ranges = {0, static_cast<int>(token_ids.size()) - 1};

  ASSERT_TRUE(normal_connector
                  ->PutGlobalCacheBlocks(token_ids, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                         block_ranges, put_ptrs)
                  .OK());
  normal_connector->Shutdown();

  // 用 ThrowingMockConnector 查询，H2D 会抛异常，不应崩溃
  void* dummy_ptr = reinterpret_cast<void*>(0x1000);
  std::vector<void*> query_ptrs = {dummy_ptr};
  std::vector<int> matched_ranges;
  Status status = connector_->QueryAndFetch(token_ids, 0, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                            query_ptrs, matched_ranges);
  EXPECT_TRUE(status.OK());
  // H2D 失败，block 未被标记为 fetched
  EXPECT_TRUE(matched_ranges.empty()) << "block should not be marked as fetched after Memcpy H2D failure";
}

// ==================== KeyUtils 测试 ====================

TEST(KeyUtilsTest, HashTokenPrefixReturns16HexChars) {
  std::vector<int> tokens = {1, 2, 3, 4};
  std::string hash = HashTokenPrefix(tokens.data(), tokens.size());
  EXPECT_EQ(hash.size(), 16u);
  for (char c : hash) {
    EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) << "non-hex char: " << c;
  }
}

TEST(KeyUtilsTest, HashTokenPrefixDeterministic) {
  std::vector<int> tokens = {10, 20, 30};
  std::string h1 = HashTokenPrefix(tokens.data(), tokens.size());
  std::string h2 = HashTokenPrefix(tokens.data(), tokens.size());
  EXPECT_EQ(h1, h2);
}

TEST(KeyUtilsTest, HashTokenPrefixDifferentInputsDiffer) {
  std::vector<int> a = {1, 2, 3};
  std::vector<int> b = {1, 2, 4};
  EXPECT_NE(HashTokenPrefix(a.data(), a.size()), HashTokenPrefix(b.data(), b.size()));
}

TEST(KeyUtilsTest, HashTokenPrefixDifferentLengthsDiffer) {
  std::vector<int> short_v = {1, 2};
  std::vector<int> long_v = {1, 2, 3};
  EXPECT_NE(HashTokenPrefix(short_v.data(), short_v.size()), HashTokenPrefix(long_v.data(), long_v.size()));
}

TEST(KeyUtilsTest, HashTokenPrefixSingleElement) {
  int token = 42;
  std::string hash = HashTokenPrefix(&token, 1);
  EXPECT_EQ(hash.size(), 16u);
}

TEST(KeyUtilsTest, MakeBlockKeyFormat) {
  std::string key = MakeBlockKey("abcdef0123456789", 2, 5, 10);
  EXPECT_EQ(key, "kv_abcdef0123456789_tp2_l5_b10");
}

TEST(KeyUtilsTest, MakeFirstTokenKeyFormat) {
  std::string key = MakeFirstTokenKey("abcdef0123456789");
  EXPECT_EQ(key, "ft_abcdef0123456789");
}

// ==================== StagingBufferPool 测试 ====================

TEST(StagingBufferPoolTest, ConstructorAllocatesBlocks) {
  StagingBufferPool pool(3, 1024);
  EXPECT_EQ(pool.BlockCount(), 3u);
  EXPECT_NE(pool.BasePtr(), nullptr);
  EXPECT_EQ(pool.TotalBytes(), 3u * 1024u);
}

TEST(StagingBufferPoolTest, GetAndPutRoundTrip) {
  StagingBufferPool pool(2, 512);
  auto* block = pool.Get();
  ASSERT_NE(block, nullptr);
  EXPECT_NE(block->ptr, nullptr);
  EXPECT_EQ(block->capacity, 512u);

  // 写入 pinned memory 并读回
  memset(block->ptr, 0xAB, 512);
  uint8_t readback = 0;
  memcpy(&readback, block->ptr, 1);
  EXPECT_EQ(readback, 0xAB);

  pool.Put(block);
}

TEST(StagingBufferPoolTest, GetPutGetReuse) {
  StagingBufferPool pool(1, 256);
  auto* b1 = pool.Get();
  ASSERT_NE(b1, nullptr);

  // Put 回去后应能再次 Get
  pool.Put(b1);
  auto* b2 = pool.Get();
  ASSERT_NE(b2, nullptr);
  pool.Put(b2);
}

TEST(StagingBufferPoolTest, ContiguousMemoryLayout) {
  StagingBufferPool pool(4, 128);
  // All blocks should be contiguous slices of the base allocation.
  auto* b0 = pool.TryGet();
  auto* b1 = pool.TryGet();
  auto* b2 = pool.TryGet();
  auto* b3 = pool.TryGet();
  ASSERT_NE(b0, nullptr);
  ASSERT_NE(b1, nullptr);
  ASSERT_NE(b2, nullptr);
  ASSERT_NE(b3, nullptr);

  // Each ptr should be within [BasePtr, BasePtr + TotalBytes)
  auto* base = static_cast<char*>(pool.BasePtr());
  for (auto* blk : {b0, b1, b2, b3}) {
    EXPECT_NE(blk->ptr, nullptr);
    EXPECT_EQ(blk->capacity, 128u);
    auto offset = static_cast<char*>(blk->ptr) - base;
    EXPECT_GE(offset, 0);
    EXPECT_LT(static_cast<size_t>(offset), pool.TotalBytes());
    EXPECT_EQ(offset % 128, 0);
  }

  pool.Put(b0);
  pool.Put(b1);
  pool.Put(b2);
  pool.Put(b3);
}

TEST(StagingBufferPoolTest, CudaMemcpyWithPinnedMemory) {
  StagingBufferPool pool(1, 1024);
  auto* block = pool.Get();
  ASSERT_NE(block, nullptr);

  // Host→Device→Host round-trip 验证 pinned memory 可用于 CUDA 传输
  std::vector<uint8_t> src(1024, 0xCD);
  memcpy(block->ptr, src.data(), 1024);

  void* d_ptr = nullptr;
  cudaMalloc(&d_ptr, 1024);
  ASSERT_NE(d_ptr, nullptr);
  cudaMemcpy(d_ptr, block->ptr, 1024, cudaMemcpyHostToDevice);

  std::vector<uint8_t> dst(1024, 0);
  cudaMemcpy(dst.data(), d_ptr, 1024, cudaMemcpyDeviceToHost);
  cudaFree(d_ptr);

  EXPECT_EQ(src, dst);
  pool.Put(block);
}

}  // namespace ksana_llm
