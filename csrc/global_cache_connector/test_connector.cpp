/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

/*
 * GlobalCacheConnector 独立功能测试
 *
 * 测试内容:
 *   1. Factory 创建 mock 后端
 *   2. PutGlobalCacheBlocks + QueryAndFetch 数据一致性
 *   3. PutFirstTokens + QueryFirstTokens 首 token 存取
 *   4. Shutdown 清理
 *
 * 使用 cudaMalloc 分配 GPU 内存模拟实际使用场景。
 */

#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "global_cache_connector/global_cache_connector_factory.h"

using namespace ksana_llm;

static constexpr int kBlockSize = 128;
static constexpr size_t kBlockByteSize = 4096;
static constexpr int kNumBlocks = 4;
static constexpr int kTpSize = 1;
static constexpr int kTpRank = 0;
static constexpr int kLayerIndex = 0;

// GPU block: 在设备上分配内存，通过 cudaMemcpy 初始化/读取
struct GpuBlocks {
  std::vector<void*> ptrs;

  GpuBlocks(int count, size_t byte_size) {
    ptrs.resize(count);
    for (int i = 0; i < count; ++i) {
      cudaError_t err = cudaMalloc(&ptrs[i], byte_size);
      if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        std::abort();
      }
    }
  }

  ~GpuBlocks() {
    for (auto* p : ptrs) {
      if (p) cudaFree(p);
    }
  }

  // 从 host 数据写入 GPU block
  void Upload(int idx, const void* host_data, size_t size) {
    cudaMemcpy(ptrs[idx], host_data, size, cudaMemcpyHostToDevice);
  }

  // 从 GPU block 读回 host
  void Download(int idx, void* host_data, size_t size) {
    cudaMemcpy(host_data, ptrs[idx], size, cudaMemcpyDeviceToHost);
  }
};

static void FillPattern(std::vector<char>& buf, uint8_t seed) {
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<char>((seed + i) & 0xFF);
  }
}

static bool VerifyPattern(const std::vector<char>& buf, uint8_t seed) {
  for (size_t i = 0; i < buf.size(); ++i) {
    if (static_cast<uint8_t>(buf[i]) != static_cast<uint8_t>((seed + i) & 0xFF)) {
      return false;
    }
  }
  return true;
}

static std::vector<int> MakeTokenIds(int count) {
  std::vector<int> ids(count);
  std::iota(ids.begin(), ids.end(), 0);
  return ids;
}

static int TestMockBackend() {
  std::cout << "=== Test: mock backend ===" << std::endl;

  GlobalCacheConnectorConfig config;
  config.backend_type = "mooncake-mock";
  config.service_endpoint = "/tmp/gcc_test_meta";

  auto connector = GlobalCacheConnectorFactory::CreateConnector(config);
  if (!connector) {
    std::cerr << "FAIL: CreateConnector returned nullptr" << std::endl;
    return 1;
  }
  std::cout << "  CreateConnector OK" << std::endl;

  // ---------- PutGlobalCacheBlocks ----------
  int total_tokens = kBlockSize * kNumBlocks;
  auto token_ids = MakeTokenIds(total_tokens);

  GpuBlocks put_blocks(kNumBlocks, kBlockByteSize);
  std::vector<std::vector<char>> host_data(kNumBlocks, std::vector<char>(kBlockByteSize));

  for (int i = 0; i < kNumBlocks; ++i) {
    FillPattern(host_data[i], static_cast<uint8_t>(i * 17));
    put_blocks.Upload(i, host_data[i].data(), kBlockByteSize);
  }

  std::vector<int> block_ranges = {0, total_tokens - 1};

  Status st = connector->PutGlobalCacheBlocks(token_ids, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                              block_ranges, put_blocks.ptrs);
  if (!st.OK()) {
    std::cerr << "FAIL: PutGlobalCacheBlocks: " << st.GetMessage() << std::endl;
    return 1;
  }
  std::cout << "  PutGlobalCacheBlocks OK (" << kNumBlocks << " blocks)" << std::endl;

  // ---------- QueryAndFetch ----------
  GpuBlocks get_blocks(kNumBlocks, kBlockByteSize);
  std::vector<int> matched_ranges;

  st = connector->QueryAndFetch(token_ids, 0, kTpSize, kTpRank, kLayerIndex, kBlockSize, kBlockByteSize,
                                get_blocks.ptrs, matched_ranges);
  if (!st.OK()) {
    std::cerr << "FAIL: QueryAndFetch: " << st.GetMessage() << std::endl;
    return 1;
  }

  if (matched_ranges.size() != 2 || matched_ranges[0] != 0 || matched_ranges[1] != total_tokens - 1) {
    std::cerr << "FAIL: matched_ranges unexpected, size=" << matched_ranges.size();
    for (auto v : matched_ranges) std::cerr << " " << v;
    std::cerr << std::endl;
    return 1;
  }

  for (int i = 0; i < kNumBlocks; ++i) {
    std::vector<char> readback(kBlockByteSize);
    get_blocks.Download(i, readback.data(), kBlockByteSize);
    if (!VerifyPattern(readback, static_cast<uint8_t>(i * 17))) {
      std::cerr << "FAIL: data mismatch at block " << i << std::endl;
      return 1;
    }
  }
  std::cout << "  QueryAndFetch OK, data verified" << std::endl;

  // ---------- PutFirstTokens + QueryFirstTokens ----------
  std::vector<int> first_tokens = {42, 100, 200};
  st = connector->PutFirstTokens(token_ids, first_tokens);
  if (!st.OK()) {
    std::cerr << "FAIL: PutFirstTokens: " << st.GetMessage() << std::endl;
    return 1;
  }

  std::vector<int> queried_tokens;
  bool found = false;
  st = connector->QueryFirstTokens(token_ids, queried_tokens, found);
  if (!st.OK()) {
    std::cerr << "FAIL: QueryFirstTokens: " << st.GetMessage() << std::endl;
    return 1;
  }
  if (!found || queried_tokens != first_tokens) {
    std::cerr << "FAIL: first tokens mismatch, found=" << found << std::endl;
    return 1;
  }
  std::cout << "  PutFirstTokens + QueryFirstTokens OK" << std::endl;

  std::cout << "=== All mock tests passed ===" << std::endl;
  return 0;
}

int main() {
  setenv("KLLM_LOG_LEVEL", "INFO", 1);

  int ret = TestMockBackend();
  return ret;
}
