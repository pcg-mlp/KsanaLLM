/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

// Unit tests for DSV4 c4 prefix-cache overlap tail copy kernels:
//   - CopyTailToState  (read-copy: persistent tail region → transient state buffer)
//   - CopyKvScoreToTail (write-copy: per-token kv_score → persistent tail region)
//
// 这些 kernel 是纯 CUDA，仅依赖 cudaStream/cudaMalloc，不使用 ksana_llm 框架对象。

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "layers/nvidia/dsv4_tail_copy.h"
#include "test.h"

namespace ksana_llm {

namespace {
// 每 token 的字节数与 kernel 的 16B 向量化对齐保持一致。
constexpr int kPerTokenBytes = 16;
// 每个 KV block 末尾保留 4 token 给 tail，与 PrepareIndices 的 kTailTokens 一致。
constexpr int kTailTokens = 4;
// c4 buffer ping-pong sub_slot 数；与 CompressorMeta::GetC4Slot() 保持一致。
constexpr int kC4Slot = 2;

// 把一个 (block_id, slot) 编码到 16 字节里，用于单元测试的可比对模式。
void FillTokenPattern(uint8_t *dst, int block_id, int slot) {
  for (int i = 0; i < kPerTokenBytes; ++i) {
    dst[i] = static_cast<uint8_t>((block_id * 31 + slot * 7 + i) & 0xFF);
  }
}
}  // namespace

class Dsv4TailCopyTest : public testing::Test {
 protected:
  void SetUp() override { ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess); }
  void TearDown() override { ASSERT_EQ(cudaStreamDestroy(stream_), cudaSuccess); }

  cudaStream_t stream_;
};

// ---- CopyTailToState ------------------------------------------------------

// 基本路径：2 个请求各自从一个 block 读 4 token tail 到 state buffer 不同 page。
TEST_F(Dsv4TailCopyTest, CopyTailToStateBasic) {
  // 4 个物理 block；block_cache_size = 64B 即整个 block 当作 tail 区使用。
  constexpr int kNumBlocks = 4;
  constexpr int64_t kBlockCacheSize = kTailTokens * kPerTokenBytes;
  constexpr int kMaxBatch = 4;
  constexpr int64_t kStatePerSlot = kTailTokens * kPerTokenBytes;

  // host: 给每个 block 填唯一可识别的字节模式
  std::vector<uint8_t> host_tail(kNumBlocks * kBlockCacheSize, 0xCC);
  for (int b = 0; b < kNumBlocks; ++b) {
    for (int s = 0; s < kTailTokens; ++s) {
      FillTokenPattern(host_tail.data() + b * kBlockCacheSize + s * kPerTokenBytes, b, s);
    }
  }

  void *dev_tail = nullptr;
  void *dev_state = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_tail, host_tail.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_state, kMaxBatch * kC4Slot * kStatePerSlot), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_tail, host_tail.data(), host_tail.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_state, 0, kMaxBatch * kC4Slot * kStatePerSlot), cudaSuccess);

  // 2 个请求：req 0 → block 1, page_index = batch_slot(0)*kC4Slot + sub_slot(1) = 1
  //           req 1 → block 3, page_index = batch_slot(2)*kC4Slot + sub_slot(0) = 4
  const int num_req = 2;
  std::vector<int32_t> host_read_block = {1, 3};
  std::vector<int32_t> host_target_indices = {0 * kC4Slot + 1, 2 * kC4Slot + 0};  // page indices

  int32_t *dev_read_block = nullptr, *dev_target_indices = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_read_block, num_req * sizeof(int32_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_target_indices, num_req * sizeof(int32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_read_block, host_read_block.data(), num_req * sizeof(int32_t), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(dev_target_indices, host_target_indices.data(), num_req * sizeof(int32_t), cudaMemcpyHostToDevice),
      cudaSuccess);

  CopyTailToState(dev_state, dev_tail, dev_read_block, dev_target_indices, num_req, kBlockCacheSize, kPerTokenBytes,
                  kTailTokens, stream_);

  std::vector<uint8_t> host_state(kMaxBatch * kC4Slot * kStatePerSlot, 0);
  ASSERT_EQ(cudaMemcpyAsync(host_state.data(), dev_state, host_state.size(), cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  // 校验拷贝目标位置内容与源 block tail 一致。
  for (int r = 0; r < num_req; ++r) {
    const int64_t dst_offset = host_target_indices[r] * kStatePerSlot;
    for (int s = 0; s < kTailTokens; ++s) {
      uint8_t expected[kPerTokenBytes];
      FillTokenPattern(expected, host_read_block[r], s);
      EXPECT_EQ(std::memcmp(host_state.data() + dst_offset + s * kPerTokenBytes, expected, kPerTokenBytes), 0)
          << "req " << r << " slot " << s << " mismatch";
    }
  }

  // 校验未被任一请求写入的 page 仍为 0。
  for (int page = 0; page < kMaxBatch * kC4Slot; ++page) {
    bool touched = false;
    for (int r = 0; r < num_req; ++r) {
      if (host_target_indices[r] == page) {
        touched = true;
        break;
      }
    }
    if (touched) continue;
    const int64_t off = page * kStatePerSlot;
    for (int i = 0; i < kStatePerSlot; ++i) {
      EXPECT_EQ(host_state[off + i], 0u) << "untouched page " << page << " polluted";
    }
  }

  cudaFree(dev_tail);
  cudaFree(dev_state);
  cudaFree(dev_read_block);
  cudaFree(dev_target_indices);
}

// 早退路径：tail_read_block < 0 (sentinel) 应跳过，不写入 state buffer。
TEST_F(Dsv4TailCopyTest, CopyTailToStateEarlyExit) {
  constexpr int kNumBlocks = 2;
  constexpr int64_t kBlockCacheSize = kTailTokens * kPerTokenBytes;
  constexpr int kMaxBatch = 4;
  constexpr int64_t kStatePerSlot = kTailTokens * kPerTokenBytes;

  std::vector<uint8_t> host_tail(kNumBlocks * kBlockCacheSize, 0xAB);  // 源用非零值，方便检测被错误拷入
  void *dev_tail = nullptr;
  void *dev_state = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_tail, host_tail.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_state, kMaxBatch * kC4Slot * kStatePerSlot), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_tail, host_tail.data(), host_tail.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_state, 0, kMaxBatch * kC4Slot * kStatePerSlot), cudaSuccess);

  // 3 个请求：全部 block sentinel = -1，应被早退。
  const int num_req = 3;
  std::vector<int32_t> host_read_block = {-1, -1, -1};
  std::vector<int32_t> host_target_indices = {1, 3, 5};  // 不影响，因为 block_id < 0 直接 return

  int32_t *dev_read_block = nullptr, *dev_target_indices = nullptr;
  cudaMalloc(&dev_read_block, num_req * sizeof(int32_t));
  cudaMalloc(&dev_target_indices, num_req * sizeof(int32_t));
  cudaMemcpy(dev_read_block, host_read_block.data(), num_req * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_target_indices, host_target_indices.data(), num_req * sizeof(int32_t), cudaMemcpyHostToDevice);

  CopyTailToState(dev_state, dev_tail, dev_read_block, dev_target_indices, num_req, kBlockCacheSize, kPerTokenBytes,
                  kTailTokens, stream_);

  std::vector<uint8_t> host_state(kMaxBatch * kC4Slot * kStatePerSlot, 0xFF);
  ASSERT_EQ(cudaMemcpyAsync(host_state.data(), dev_state, host_state.size(), cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  // state buffer 应保持 0（未被写入）。
  for (size_t i = 0; i < host_state.size(); ++i) {
    EXPECT_EQ(host_state[i], 0u) << "early-exit byte " << i << " was written";
  }

  cudaFree(dev_tail);
  cudaFree(dev_state);
  cudaFree(dev_read_block);
  cudaFree(dev_target_indices);
}

// ---- CopyKvScoreToTail ----------------------------------------------------

// 基本路径：连续 kv_score 8 token 中第 4..7 token 落在某 block 的 tail 4 槽位。
TEST_F(Dsv4TailCopyTest, CopyKvScoreToTailBasic) {
  constexpr int kNumBlocks = 2;
  constexpr int64_t kBlockCacheSize = kTailTokens * kPerTokenBytes;
  constexpr int kTotalTokens = 8;

  // 源 kv_score：每 token 16 字节，按 token id 编码可识别模式。
  std::vector<uint8_t> host_kv_score(kTotalTokens * kPerTokenBytes);
  for (int t = 0; t < kTotalTokens; ++t) {
    for (int i = 0; i < kPerTokenBytes; ++i) {
      host_kv_score[t * kPerTokenBytes + i] = static_cast<uint8_t>((t * 13 + i) & 0xFF);
    }
  }

  void *dev_tail = nullptr, *dev_kv_score = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_tail, kNumBlocks * kBlockCacheSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_kv_score, host_kv_score.size()), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_tail, 0, kNumBlocks * kBlockCacheSize), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_kv_score, host_kv_score.data(), host_kv_score.size(), cudaMemcpyHostToDevice), cudaSuccess);

  // 模拟 PrepareIndices 输出：本 step 8 个 token 中，token 4..7 落 block 1 tail 0..3 槽。
  // Packed encoding: block_id * 4 | slot
  std::vector<int32_t> host_write_indices(kTotalTokens, -1);
  for (int t = 4; t < 8; ++t) {
    host_write_indices[t] = 1 * kTailTokens | (t - 4);
  }

  int32_t *dev_write_indices = nullptr;
  cudaMalloc(&dev_write_indices, kTotalTokens * sizeof(int32_t));
  cudaMemcpy(dev_write_indices, host_write_indices.data(), kTotalTokens * sizeof(int32_t), cudaMemcpyHostToDevice);

  CopyKvScoreToTail(dev_tail, dev_kv_score, dev_write_indices, kTotalTokens, kBlockCacheSize, kPerTokenBytes, stream_);

  std::vector<uint8_t> host_tail(kNumBlocks * kBlockCacheSize, 0);
  ASSERT_EQ(cudaMemcpyAsync(host_tail.data(), dev_tail, host_tail.size(), cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  // block 0 完全没被写入（仍为 0）。
  for (int i = 0; i < kBlockCacheSize; ++i) {
    EXPECT_EQ(host_tail[i], 0u) << "block 0 polluted at byte " << i;
  }
  // block 1 的 4 个 tail slot 应分别等于 host_kv_score[4..7]。
  for (int s = 0; s < kTailTokens; ++s) {
    const int t = 4 + s;
    EXPECT_EQ(std::memcmp(host_tail.data() + kBlockCacheSize + s * kPerTokenBytes,
                          host_kv_score.data() + t * kPerTokenBytes, kPerTokenBytes),
              0)
        << "block 1 slot " << s << " mismatch";
  }

  cudaFree(dev_tail);
  cudaFree(dev_kv_score);
  cudaFree(dev_write_indices);
}

// 早退路径：所有 token 的 tail_write_indices = -1 时不应有任何字节被写入。
TEST_F(Dsv4TailCopyTest, CopyKvScoreToTailAllSentinel) {
  constexpr int kNumBlocks = 2;
  constexpr int64_t kBlockCacheSize = kTailTokens * kPerTokenBytes;
  constexpr int kTotalTokens = 6;

  std::vector<uint8_t> host_kv_score(kTotalTokens * kPerTokenBytes, 0xEE);

  void *dev_tail = nullptr, *dev_kv_score = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_tail, kNumBlocks * kBlockCacheSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_kv_score, host_kv_score.size()), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_tail, 0, kNumBlocks * kBlockCacheSize), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_kv_score, host_kv_score.data(), host_kv_score.size(), cudaMemcpyHostToDevice), cudaSuccess);

  std::vector<int32_t> host_write_indices(kTotalTokens, -1);

  int32_t *dev_write_indices = nullptr;
  cudaMalloc(&dev_write_indices, kTotalTokens * sizeof(int32_t));
  cudaMemcpy(dev_write_indices, host_write_indices.data(), kTotalTokens * sizeof(int32_t), cudaMemcpyHostToDevice);

  CopyKvScoreToTail(dev_tail, dev_kv_score, dev_write_indices, kTotalTokens, kBlockCacheSize, kPerTokenBytes, stream_);

  std::vector<uint8_t> host_tail(kNumBlocks * kBlockCacheSize, 0xFF);
  ASSERT_EQ(cudaMemcpyAsync(host_tail.data(), dev_tail, host_tail.size(), cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (size_t i = 0; i < host_tail.size(); ++i) {
    EXPECT_EQ(host_tail[i], 0u) << "all-sentinel write polluted byte " << i;
  }

  cudaFree(dev_tail);
  cudaFree(dev_kv_score);
  cudaFree(dev_write_indices);
}

// 验证当 SWA pool 与主 KV pool 的 block_size 不同时, CopyTailToState / CopyKvScoreToTail
// 用 SWA pool 的 stride 仍然能正确寻址 tail 区 (即 caller 必须传 SWA stride, 不能传 main KV stride)。
// 模拟场景: SWA pool 单 block = 64B (4 tail tokens * 16B), 主 KV pool 单 block = 256B。
// 若 kernel 误用主 KV stride 进行 byte 计算, 会因 4 倍偏移落到 block 4 位置, 目标 block 1 写入失败。
TEST_F(Dsv4TailCopyTest, BlockStrideMismatchSafe) {
  constexpr int kSwaPoolBlocks = 4;
  constexpr int64_t kSwaBlockSize = kTailTokens * kPerTokenBytes;  // 64B (SWA pool stride)
  constexpr int64_t kMainBlockSize = kSwaBlockSize * 4;            // 256B (主 KV pool stride, 不应被使用)
  constexpr int kTotalTokens = 4;
  static_assert(kSwaBlockSize != kMainBlockSize, "test requires distinct strides");

  std::vector<uint8_t> host_kv_score(kTotalTokens * kPerTokenBytes);
  for (int t = 0; t < kTotalTokens; ++t) {
    for (int i = 0; i < kPerTokenBytes; ++i) {
      host_kv_score[t * kPerTokenBytes + i] = static_cast<uint8_t>((t * 17 + i) & 0xFF);
    }
  }

  void *dev_tail = nullptr, *dev_kv_score = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_tail, kSwaPoolBlocks * kSwaBlockSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_kv_score, host_kv_score.size()), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_tail, 0, kSwaPoolBlocks * kSwaBlockSize), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_kv_score, host_kv_score.data(), host_kv_score.size(), cudaMemcpyHostToDevice), cudaSuccess);

  // Packed encoding: block_id * 4 | slot. All 4 tokens go to block 1 slots 0..3.
  std::vector<int32_t> host_write_indices = {1 * kTailTokens | 0, 1 * kTailTokens | 1, 1 * kTailTokens | 2,
                                             1 * kTailTokens | 3};

  int32_t *dev_write_indices = nullptr;
  cudaMalloc(&dev_write_indices, kTotalTokens * sizeof(int32_t));
  cudaMemcpy(dev_write_indices, host_write_indices.data(), kTotalTokens * sizeof(int32_t), cudaMemcpyHostToDevice);

  // 用 SWA pool 自身的 stride 调用 kernel (caller 必须传 SWA stride 而不是主 KV stride)。
  CopyKvScoreToTail(dev_tail, dev_kv_score, dev_write_indices, kTotalTokens, kSwaBlockSize, kPerTokenBytes, stream_);

  std::vector<uint8_t> host_tail(kSwaPoolBlocks * kSwaBlockSize, 0);
  ASSERT_EQ(cudaMemcpyAsync(host_tail.data(), dev_tail, host_tail.size(), cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  // block 0 / 2 / 3 都应保持为 0; block 1 的 4 个 tail slot 应等于 host_kv_score。
  for (int b = 0; b < kSwaPoolBlocks; ++b) {
    if (b == 1) continue;
    for (int i = 0; i < kSwaBlockSize; ++i) {
      EXPECT_EQ(host_tail[b * kSwaBlockSize + i], 0u) << "block " << b << " polluted at byte " << i;
    }
  }
  for (int s = 0; s < kTailTokens; ++s) {
    EXPECT_EQ(std::memcmp(host_tail.data() + 1 * kSwaBlockSize + s * kPerTokenBytes,
                          host_kv_score.data() + s * kPerTokenBytes, kPerTokenBytes),
              0)
        << "block 1 slot " << s << " mismatch (SWA pool stride 错配会写到 block 4 之外)";
  }

  cudaFree(dev_tail);
  cudaFree(dev_kv_score);
  cudaFree(dev_write_indices);
}

}  // namespace ksana_llm
