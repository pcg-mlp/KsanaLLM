/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

/*
 * GlobalCacheConnector 性能基准测试
 *
 * 测量 mock 后端在不同参数下的 Put / Query 吞吐量和延迟。
 * 使用 cudaMalloc 分配 GPU 内存以模拟实际使用场景。
 *
 * 可选通过环境变量配置:
 *   BENCH_BLOCK_SIZE       — 每个 block 的 token 数 (默认 128)
 *   BENCH_BLOCK_BYTE_SIZE  — 每个 block 的字节大小 (默认 1MB)
 *   BENCH_NUM_BLOCKS       — block 数量 (默认 16)
 *   BENCH_NUM_LAYERS       — 模拟层数 (默认 32)
 *   BENCH_ITERATIONS       — 重复次数 (默认 10)
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "global_cache_connector/global_cache_connector_factory.h"

using namespace ksana_llm;
using Clock = std::chrono::high_resolution_clock;

static int GetEnvInt(const char* name, int default_val) {
  const char* val = std::getenv(name);
  return val ? std::atoi(val) : default_val;
}

struct BenchConfig {
  int block_size;
  size_t block_byte_size;
  int num_blocks;
  int num_layers;
  int iterations;
};

struct GpuBlockPool {
  std::vector<void*> ptrs;
  size_t byte_size;

  GpuBlockPool(int count, size_t bsize) : byte_size(bsize) {
    ptrs.resize(count);
    for (int i = 0; i < count; ++i) {
      cudaError_t err = cudaMalloc(&ptrs[i], bsize);
      if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        std::abort();
      }
      cudaMemset(ptrs[i], static_cast<int>(i & 0xFF), bsize);
    }
  }

  ~GpuBlockPool() {
    for (auto* p : ptrs) {
      if (p) cudaFree(p);
    }
  }
};

static void BenchPut(GlobalCacheConnectorInterface* connector, const BenchConfig& cfg) {
  int total_tokens = cfg.block_size * cfg.num_blocks;
  std::vector<int> token_ids(total_tokens);
  std::iota(token_ids.begin(), token_ids.end(), 0);

  GpuBlockPool blocks(cfg.num_blocks, cfg.block_byte_size);
  std::vector<int> block_ranges = {0, total_tokens - 1};

  double total_ms = 0;
  size_t total_bytes = 0;

  for (int iter = 0; iter < cfg.iterations; ++iter) {
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
      token_ids[0] = iter * cfg.num_layers + layer;

      auto t0 = Clock::now();
      connector->PutGlobalCacheBlocks(token_ids, 1, 0, layer, cfg.block_size, cfg.block_byte_size, block_ranges,
                                      blocks.ptrs);
      auto t1 = Clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
      total_bytes += cfg.num_blocks * cfg.block_byte_size;
    }
  }

  double throughput_mb = (total_bytes / (1024.0 * 1024.0)) / (total_ms / 1000.0);
  int total_ops = cfg.iterations * cfg.num_layers;
  std::cout << "  Put: " << total_ops << " calls, " << total_ms << " ms total, " << (total_ms / total_ops)
            << " ms/call, " << throughput_mb << " MB/s" << std::endl;
}

static void BenchQuery(GlobalCacheConnectorInterface* connector, const BenchConfig& cfg) {
  int total_tokens = cfg.block_size * cfg.num_blocks;
  std::vector<int> token_ids(total_tokens);
  std::iota(token_ids.begin(), token_ids.end(), 0);

  GpuBlockPool put_blocks(cfg.num_blocks, cfg.block_byte_size);
  std::vector<int> block_ranges = {0, total_tokens - 1};

  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    connector->PutGlobalCacheBlocks(token_ids, 1, 0, layer, cfg.block_size, cfg.block_byte_size, block_ranges,
                                    put_blocks.ptrs);
  }

  GpuBlockPool get_blocks(cfg.num_blocks, cfg.block_byte_size);

  double total_ms = 0;
  size_t total_bytes = 0;

  for (int iter = 0; iter < cfg.iterations; ++iter) {
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
      std::vector<int> matched;
      auto t0 = Clock::now();
      connector->QueryAndFetch(token_ids, 0, 1, 0, layer, cfg.block_size, cfg.block_byte_size, get_blocks.ptrs,
                               matched);
      auto t1 = Clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
      total_bytes += cfg.num_blocks * cfg.block_byte_size;
    }
  }

  double throughput_mb = (total_bytes / (1024.0 * 1024.0)) / (total_ms / 1000.0);
  int total_ops = cfg.iterations * cfg.num_layers;
  std::cout << "  Query: " << total_ops << " calls, " << total_ms << " ms total, " << (total_ms / total_ops)
            << " ms/call, " << throughput_mb << " MB/s" << std::endl;
}

int main() {
  BenchConfig cfg{
      .block_size = GetEnvInt("BENCH_BLOCK_SIZE", 128),
      .block_byte_size = static_cast<size_t>(GetEnvInt("BENCH_BLOCK_BYTE_SIZE", 1024 * 1024)),
      .num_blocks = GetEnvInt("BENCH_NUM_BLOCKS", 16),
      .num_layers = GetEnvInt("BENCH_NUM_LAYERS", 32),
      .iterations = GetEnvInt("BENCH_ITERATIONS", 10),
  };

  std::cout << "=== GlobalCacheConnector Benchmark ===" << std::endl;
  std::cout << "  block_size=" << cfg.block_size << ", block_byte_size=" << cfg.block_byte_size
            << ", num_blocks=" << cfg.num_blocks << ", num_layers=" << cfg.num_layers
            << ", iterations=" << cfg.iterations << std::endl;

  GlobalCacheConnectorConfig config;
  config.backend_type = "mooncake-mock";
  config.service_endpoint = "/tmp/gcc_bench_meta";

  auto connector = GlobalCacheConnectorFactory::CreateConnector(config);
  if (!connector) {
    std::cerr << "FAIL: CreateConnector returned nullptr" << std::endl;
    return 1;
  }

  BenchPut(connector.get(), cfg);
  BenchQuery(connector.get(), cfg);

  std::cout << "=== Benchmark done ===" << std::endl;
  return 0;
}
