/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

// Unit tests for BatchMemoryD2D batched device-to-device copy.
// Tests invoke LaunchBatchMemoryD2DKernel directly (raw CUDA, no framework objects)
// to isolate kernel correctness from Tensor/Stream construction overhead.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "test.h"

namespace ksana_llm {

// Defined in memory_utils.cu.
void LaunchBatchMemoryD2DKernel(const uint64_t* src_addr_list, const uint64_t* dst_addr_list, size_t src_offset,
                                size_t dst_offset, size_t block_size, int N, cudaStream_t stream);

namespace {

// Fill a host buffer with a deterministic pattern so any addressing error produces
// bytes that fail byte-for-byte comparison.
void FillPattern(uint8_t* buf, size_t size, int seed) {
  for (size_t i = 0; i < size; ++i) {
    buf[i] = static_cast<uint8_t>((seed * 31 + static_cast<int>(i)) & 0xFF);
  }
}

// Return a device copy of a host vector and record the device pointer.
template <typename T>
T* UploadVector(const std::vector<T>& host) {
  T* dev = nullptr;
  cudaMalloc(&dev, host.size() * sizeof(T));
  cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
  return dev;
}

}  // namespace

class BatchMemoryD2DTest : public testing::Test {
 protected:
  void SetUp() override { ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess); }
  void TearDown() override { ASSERT_EQ(cudaStreamDestroy(stream_), cudaSuccess); }

  cudaStream_t stream_;
};

// ---- 1. 多 block 顺序拷贝正确性 --------------------------------------------
// N 个 block 按顺序 src[i] → dst[i]，block_size = 1024。
// 每个 src block 填充与 block 索引绑定的唯一字节模式，验证 dst 完全一致。
TEST_F(BatchMemoryD2DTest, MultiBlockContiguous) {
  constexpr int N = 8;
  constexpr size_t kBlockSize = 1024;  // 1KB, 256 的倍数

  const size_t total_bytes = static_cast<size_t>(N) * kBlockSize;

  // 准备 host 源数据
  std::vector<uint8_t> host_src(total_bytes);
  for (int i = 0; i < N; ++i) {
    FillPattern(host_src.data() + i * kBlockSize, kBlockSize, i + 1);
  }

  // 上传源数据，分配目标缓冲区
  uint8_t* dev_src = nullptr;
  uint8_t* dev_dst = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_src, total_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_dst, total_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_src, host_src.data(), total_bytes, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_dst, 0, total_bytes), cudaSuccess);

  // 地址列表：src[i] / dst[i] 均按 block_size 等步长排布
  std::vector<uint64_t> host_src_list(N), host_dst_list(N);
  for (int i = 0; i < N; ++i) {
    host_src_list[i] = static_cast<uint64_t>(i) * kBlockSize;
    host_dst_list[i] = static_cast<uint64_t>(i) * kBlockSize;
  }
  uint64_t* dev_src_list = UploadVector(host_src_list);
  uint64_t* dev_dst_list = UploadVector(host_dst_list);

  LaunchBatchMemoryD2DKernel(dev_src_list, dev_dst_list, reinterpret_cast<size_t>(dev_src),
                             reinterpret_cast<size_t>(dev_dst), kBlockSize, N, stream_);

  std::vector<uint8_t> host_dst(total_bytes, 0);
  ASSERT_EQ(cudaMemcpyAsync(host_dst.data(), dev_dst, total_bytes, cudaMemcpyDeviceToHost, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(std::memcmp(host_dst.data() + i * kBlockSize, host_src.data() + i * kBlockSize, kBlockSize), 0)
        << "block " << i << " mismatch";
  }

  cudaFree(dev_src);
  cudaFree(dev_dst);
  cudaFree(dev_src_list);
  cudaFree(dev_dst_list);
}

// ---- 2. 打散地址列表（置换拷贝）-------------------------------------------
// dst[i] 接收 src[N-1-i] 的数据，通过错开地址列表实现，验证 src/dst 偏移独立解释正确。
TEST_F(BatchMemoryD2DTest, MultiBlockShuffled) {
  constexpr int N = 6;
  constexpr size_t kBlockSize = 2048;  // 2KB

  const size_t total_bytes = static_cast<size_t>(N) * kBlockSize;

  std::vector<uint8_t> host_src(total_bytes);
  for (int i = 0; i < N; ++i) {
    FillPattern(host_src.data() + i * kBlockSize, kBlockSize, (i + 1) * 7);
  }

  uint8_t* dev_src = nullptr;
  uint8_t* dev_dst = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_src, total_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_dst, total_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_src, host_src.data(), total_bytes, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_dst, 0, total_bytes), cudaSuccess);

  // dst[i] ← src[N-1-i]：src 地址逆序，dst 地址正序
  std::vector<uint64_t> host_src_list(N), host_dst_list(N);
  for (int i = 0; i < N; ++i) {
    host_src_list[i] = static_cast<uint64_t>(N - 1 - i) * kBlockSize;  // 逆序
    host_dst_list[i] = static_cast<uint64_t>(i) * kBlockSize;
  }
  uint64_t* dev_src_list = UploadVector(host_src_list);
  uint64_t* dev_dst_list = UploadVector(host_dst_list);

  LaunchBatchMemoryD2DKernel(dev_src_list, dev_dst_list, reinterpret_cast<size_t>(dev_src),
                             reinterpret_cast<size_t>(dev_dst), kBlockSize, N, stream_);

  std::vector<uint8_t> host_dst(total_bytes, 0);
  ASSERT_EQ(cudaMemcpyAsync(host_dst.data(), dev_dst, total_bytes, cudaMemcpyDeviceToHost, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (int i = 0; i < N; ++i) {
    int src_block = N - 1 - i;
    EXPECT_EQ(std::memcmp(host_dst.data() + i * kBlockSize, host_src.data() + src_block * kBlockSize, kBlockSize), 0)
        << "shuffled dst[" << i << "] should equal src[" << src_block << "]";
  }

  cudaFree(dev_src);
  cudaFree(dev_dst);
  cudaFree(dev_src_list);
  cudaFree(dev_dst_list);
}

// ---- 3. 大 block_size（多 grid.y 分块）--------------------------------------
// block_size = 65536（64KB）触发 grid.y > 1 的分块逻辑，N = 4 条 pair。
TEST_F(BatchMemoryD2DTest, LargeBlockSize) {
  constexpr int N = 4;
  constexpr size_t kBlockSize = 65536;  // 64KB

  const size_t total_bytes = static_cast<size_t>(N) * kBlockSize;

  std::vector<uint8_t> host_src(total_bytes);
  for (int i = 0; i < N; ++i) {
    FillPattern(host_src.data() + i * kBlockSize, kBlockSize, i * 13 + 3);
  }

  uint8_t* dev_src = nullptr;
  uint8_t* dev_dst = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_src, total_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_dst, total_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_src, host_src.data(), total_bytes, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_dst, 0, total_bytes), cudaSuccess);

  std::vector<uint64_t> host_src_list(N), host_dst_list(N);
  for (int i = 0; i < N; ++i) {
    host_src_list[i] = static_cast<uint64_t>(i) * kBlockSize;
    host_dst_list[i] = static_cast<uint64_t>(i) * kBlockSize;
  }
  uint64_t* dev_src_list = UploadVector(host_src_list);
  uint64_t* dev_dst_list = UploadVector(host_dst_list);

  LaunchBatchMemoryD2DKernel(dev_src_list, dev_dst_list, reinterpret_cast<size_t>(dev_src),
                             reinterpret_cast<size_t>(dev_dst), kBlockSize, N, stream_);

  std::vector<uint8_t> host_dst(total_bytes, 0);
  ASSERT_EQ(cudaMemcpyAsync(host_dst.data(), dev_dst, total_bytes, cudaMemcpyDeviceToHost, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(std::memcmp(host_dst.data() + i * kBlockSize, host_src.data() + i * kBlockSize, kBlockSize), 0)
        << "large-block block " << i << " mismatch";
  }

  cudaFree(dev_src);
  cudaFree(dev_dst);
  cudaFree(dev_src_list);
  cudaFree(dev_dst_list);
}

// ---- 4. 非 16 字节对齐（4B 对齐路径）---------------------------------------
// src/dst 指针偏移 4 字节使两者都 4B 对齐但不满足 8B/16B 对齐，
// 强制走 4B 向量化分支。验证非最优对齐路径的正确性。
TEST_F(BatchMemoryD2DTest, FourByteAlignedPath) {
  constexpr int N = 5;
  constexpr size_t kBlockSize = 512;   // 512B
  constexpr size_t kAlign4Offset = 4;  // 让指针 mod 16 == 4

  // 分配比实际需要多 16 字节，用于对齐偏移
  const size_t alloc_bytes = static_cast<size_t>(N) * kBlockSize + 16;

  std::vector<uint8_t> host_src_raw(alloc_bytes);
  // 用错开 4 字节的区域填充
  uint8_t* src_base = host_src_raw.data() + kAlign4Offset;
  for (int i = 0; i < N; ++i) {
    FillPattern(src_base + i * kBlockSize, kBlockSize, i * 19 + 5);
  }

  uint8_t* dev_src_raw = nullptr;
  uint8_t* dev_dst_raw = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_src_raw, alloc_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_dst_raw, alloc_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_src_raw, host_src_raw.data(), alloc_bytes, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_dst_raw, 0, alloc_bytes), cudaSuccess);

  // src_offset / dst_offset 让指针 mod 16 == 4（假设 cudaMalloc 返回 256B 对齐地址）
  size_t src_offset = reinterpret_cast<size_t>(dev_src_raw) + kAlign4Offset;
  size_t dst_offset = reinterpret_cast<size_t>(dev_dst_raw) + kAlign4Offset;

  std::vector<uint64_t> host_src_list(N), host_dst_list(N);
  for (int i = 0; i < N; ++i) {
    host_src_list[i] = static_cast<uint64_t>(i) * kBlockSize;
    host_dst_list[i] = static_cast<uint64_t>(i) * kBlockSize;
  }
  uint64_t* dev_src_list = UploadVector(host_src_list);
  uint64_t* dev_dst_list = UploadVector(host_dst_list);

  LaunchBatchMemoryD2DKernel(dev_src_list, dev_dst_list, src_offset, dst_offset, kBlockSize, N, stream_);

  std::vector<uint8_t> host_dst_raw(alloc_bytes, 0);
  ASSERT_EQ(cudaMemcpyAsync(host_dst_raw.data(), dev_dst_raw, alloc_bytes, cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  uint8_t* dst_base = host_dst_raw.data() + kAlign4Offset;
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(std::memcmp(dst_base + i * kBlockSize, src_base + i * kBlockSize, kBlockSize), 0)
        << "4B-aligned block " << i << " mismatch";
  }

  cudaFree(dev_src_raw);
  cudaFree(dev_dst_raw);
  cudaFree(dev_src_list);
  cudaFree(dev_dst_list);
}

// ---- 5. 空输入早退（N=0 / block_size=0）------------------------------------
// 不应触发 kernel launch；dst 保持不变。
TEST_F(BatchMemoryD2DTest, EmptyInputNoOp) {
  constexpr size_t kBlockSize = 1024;

  uint8_t* dev_dst = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_dst, kBlockSize), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_dst, 0xAB, kBlockSize), cudaSuccess);

  // N == 0
  LaunchBatchMemoryD2DKernel(nullptr, nullptr, 0, reinterpret_cast<size_t>(dev_dst), kBlockSize, 0, stream_);

  // block_size == 0
  LaunchBatchMemoryD2DKernel(nullptr, nullptr, 0, reinterpret_cast<size_t>(dev_dst), 0, 4, stream_);

  std::vector<uint8_t> host_dst(kBlockSize);
  ASSERT_EQ(cudaMemcpyAsync(host_dst.data(), dev_dst, kBlockSize, cudaMemcpyDeviceToHost, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (size_t i = 0; i < kBlockSize; ++i) {
    EXPECT_EQ(host_dst[i], 0xABu) << "empty-input guard byte " << i << " was overwritten";
  }

  cudaFree(dev_dst);
}

// ---- 6. 未被写入的 dst 块保持原值（不污染邻居块）---------------------------
// 地址列表只覆盖偶数编号的 block，验证奇数编号的 block 未被写入。
TEST_F(BatchMemoryD2DTest, NonTargetBlocksUntouched) {
  constexpr int kTotalBlocks = 8;
  constexpr int N = 4;  // 只拷贝偶数块
  constexpr size_t kBlockSize = 1024;

  const size_t total_bytes = static_cast<size_t>(kTotalBlocks) * kBlockSize;
  constexpr uint8_t kGuard = 0xCC;

  std::vector<uint8_t> host_src(total_bytes);
  for (int i = 0; i < N; ++i) {
    FillPattern(host_src.data() + (i * 2) * kBlockSize, kBlockSize, i * 11 + 2);
  }

  uint8_t* dev_src = nullptr;
  uint8_t* dev_dst = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_src, total_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_dst, total_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_src, host_src.data(), total_bytes, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_dst, kGuard, total_bytes), cudaSuccess);

  // 地址列表：src 偶数块 → dst 偶数块
  std::vector<uint64_t> host_src_list(N), host_dst_list(N);
  for (int i = 0; i < N; ++i) {
    host_src_list[i] = static_cast<uint64_t>(i * 2) * kBlockSize;
    host_dst_list[i] = static_cast<uint64_t>(i * 2) * kBlockSize;
  }
  uint64_t* dev_src_list = UploadVector(host_src_list);
  uint64_t* dev_dst_list = UploadVector(host_dst_list);

  LaunchBatchMemoryD2DKernel(dev_src_list, dev_dst_list, reinterpret_cast<size_t>(dev_src),
                             reinterpret_cast<size_t>(dev_dst), kBlockSize, N, stream_);

  std::vector<uint8_t> host_dst(total_bytes, 0);
  ASSERT_EQ(cudaMemcpyAsync(host_dst.data(), dev_dst, total_bytes, cudaMemcpyDeviceToHost, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (int i = 0; i < N; ++i) {
    // 偶数块应等于源数据
    EXPECT_EQ(std::memcmp(host_dst.data() + (i * 2) * kBlockSize, host_src.data() + (i * 2) * kBlockSize, kBlockSize),
              0)
        << "even block " << (i * 2) << " mismatch";

    // 奇数块应保持 guard byte
    for (size_t b = 0; b < kBlockSize; ++b) {
      EXPECT_EQ(host_dst[(i * 2 + 1) * kBlockSize + b], kGuard)
          << "odd block " << (i * 2 + 1) << " byte " << b << " polluted";
    }
  }

  cudaFree(dev_src);
  cudaFree(dev_dst);
  cudaFree(dev_src_list);
  cudaFree(dev_dst_list);
}

}  // namespace ksana_llm
