/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/

// 单元测试: req_id 完全解耦的 (cur_pairs + prev_ring) 协议.
//   - LaunchReplaceLastTokenKernel: 每 thread 处理 1 个本步 req, 自检 input_ids[dst]==-1 后,
//     在 prev_ring 中按 req_id 暴力扫描查找上一步 sampled token 回填.
//   - LaunchWriteRingKernel: 把本步 sampling 输出 + cur_pairs(req_id 部分) 写入 ring slot, thread 0 同时写 count 头部.
// 测试纯 CUDA, 不依赖 ksana_llm 框架对象.

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "layers/nvidia/replace_last_token.h"
#include "test.h"

namespace ksana_llm {

class ReplaceLastTokenTest : public testing::Test {
 protected:
  void SetUp() override { ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess); }
  void TearDown() override { ASSERT_EQ(cudaStreamDestroy(stream_), cudaSuccess); }

  cudaStream_t stream_;

  // Helper: 把 host 数据 H2D 到一块新分配的 device 内存, 返回 device 指针 (out 引用).
  // 调用方负责 cudaFree.
  template <typename T>
  void H2DAlloc(T** dev_ptr, const std::vector<T>& host) {
    ASSERT_EQ(cudaMalloc(dev_ptr, host.size() * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(*dev_ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
  }

  // Helper: 构造 prev_ring host 端布局 [count, req_id_0, token_0, ...].
  std::vector<int64_t> BuildPrevRing(const std::vector<std::pair<int64_t, int64_t>>& pairs) {
    std::vector<int64_t> ring;
    ring.reserve(1 + 2 * pairs.size());
    ring.push_back(static_cast<int64_t>(pairs.size()));
    for (const auto& p : pairs) {
      ring.push_back(p.first);   // req_id
      ring.push_back(p.second);  // token
    }
    return ring;
  }

  // Helper: 构造 cur_pairs host 端布局 [(req_id_0, off_0), ...].
  std::vector<int64_t> BuildCurPairs(const std::vector<std::pair<int64_t, int64_t>>& pairs) {
    std::vector<int64_t> v;
    v.reserve(2 * pairs.size());
    for (const auto& p : pairs) {
      v.push_back(p.first);   // req_id
      v.push_back(p.second);  // last_offset
    }
    return v;
  }
};

// 基本路径: 3 个 decode req, 全部需要替换 (input_ids 末位都是 -1), prev_ring 顺序与 cur_pairs 一致.
TEST_F(ReplaceLastTokenTest, BasicReplaceAll) {
  // input_ids: 每 req 1 个 token, 全部为 -1 (placeholder)
  std::vector<int32_t> host_input = {-1, -1, -1};
  // cur_pairs: req_id=10/20/30, last_offset=0/1/2
  auto cur_pairs_host = BuildCurPairs({{10, 0}, {20, 1}, {30, 2}});
  // prev_ring: count=3, (10,99) (20,88) (30,77)
  auto prev_ring_host = BuildPrevRing({{10, 99}, {20, 88}, {30, 77}});

  int32_t* dev_input = nullptr;
  int64_t* dev_cur_pairs = nullptr;
  int64_t* dev_prev_ring = nullptr;
  H2DAlloc(&dev_input, host_input);
  H2DAlloc(&dev_cur_pairs, cur_pairs_host);
  H2DAlloc(&dev_prev_ring, prev_ring_host);

  LaunchReplaceLastTokenKernel(dev_input, dev_cur_pairs, dev_prev_ring, /*batch_size=*/3, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  std::vector<int32_t> host_result(host_input.size());
  ASSERT_EQ(cudaMemcpy(host_result.data(), dev_input, host_result.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            cudaSuccess);

  EXPECT_EQ(host_result[0], 99);
  EXPECT_EQ(host_result[1], 88);
  EXPECT_EQ(host_result[2], 77);

  cudaFree(dev_input);
  cudaFree(dev_cur_pairs);
  cudaFree(dev_prev_ring);
}

// 解耦路径: 本步 batch 顺序与上一步不同, 验证按 req_id 命中 (而非按 batch 下标).
TEST_F(ReplaceLastTokenTest, ReorderByReqId) {
  // 本步顺序 30, 10, 20; 上一步 ring 顺序 10, 20, 30 → 各自按 req_id 找到自己上一步的 token.
  std::vector<int32_t> host_input = {-1, -1, -1};
  auto cur_pairs_host = BuildCurPairs({{30, 0}, {10, 1}, {20, 2}});
  auto prev_ring_host = BuildPrevRing({{10, 99}, {20, 88}, {30, 77}});

  int32_t* dev_input = nullptr;
  int64_t* dev_cur_pairs = nullptr;
  int64_t* dev_prev_ring = nullptr;
  H2DAlloc(&dev_input, host_input);
  H2DAlloc(&dev_cur_pairs, cur_pairs_host);
  H2DAlloc(&dev_prev_ring, prev_ring_host);

  LaunchReplaceLastTokenKernel(dev_input, dev_cur_pairs, dev_prev_ring, /*batch_size=*/3, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  std::vector<int32_t> host_result(host_input.size());
  ASSERT_EQ(cudaMemcpy(host_result.data(), dev_input, host_result.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            cudaSuccess);

  EXPECT_EQ(host_result[0], 77);  // req=30 → 77
  EXPECT_EQ(host_result[1], 99);  // req=10 → 99
  EXPECT_EQ(host_result[2], 88);  // req=20 → 88

  cudaFree(dev_input);
  cudaFree(dev_cur_pairs);
  cudaFree(dev_prev_ring);
}

// 缺席路径: 本步出现的 req 上一步不在 prev_ring (例如刚 swap-in 的新 req).
// 正常路径下新 swap-in 的 req 不应有 placeholder；若末位被外部写成 -1，kernel 扫不到时保持 -1（保护性兜底）。
TEST_F(ReplaceLastTokenTest, MissingReqKeepPlaceholder) {
  std::vector<int32_t> host_input = {-1, -1};
  // 本步 req_id=10 老 req (能命中), req_id=99 新 req (不在 prev_ring).
  auto cur_pairs_host = BuildCurPairs({{10, 0}, {99, 1}});
  auto prev_ring_host = BuildPrevRing({{10, 7}});

  int32_t* dev_input = nullptr;
  int64_t* dev_cur_pairs = nullptr;
  int64_t* dev_prev_ring = nullptr;
  H2DAlloc(&dev_input, host_input);
  H2DAlloc(&dev_cur_pairs, cur_pairs_host);
  H2DAlloc(&dev_prev_ring, prev_ring_host);

  LaunchReplaceLastTokenKernel(dev_input, dev_cur_pairs, dev_prev_ring, /*batch_size=*/2, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  std::vector<int32_t> host_result(host_input.size());
  ASSERT_EQ(cudaMemcpy(host_result.data(), dev_input, host_result.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            cudaSuccess);

  EXPECT_EQ(host_result[0], 7);   // req=10 命中 prev_ring
  EXPECT_EQ(host_result[1], -1);  // req=99 未命中, 保持 -1

  cudaFree(dev_input);
  cudaFree(dev_cur_pairs);
  cudaFree(dev_prev_ring);
}

// 混合 prefill + decode batch: prefill req 末位不是 -1, kernel 自检后跳过, 不动 prefill 内容.
// 同时 decode req 正确替换.
TEST_F(ReplaceLastTokenTest, MixedPrefillDecodeKeepNonPlaceholder) {
  // input_ids 模拟: req_A (prefill) 占 0..3, last_offset=3 是真实 token 5; req_B (decode) 占 4, last_offset=4 是 -1.
  std::vector<int32_t> host_input = {1, 2, 3, 5, -1};
  auto cur_pairs_host = BuildCurPairs({{100, 3}, {200, 4}});
  auto prev_ring_host = BuildPrevRing({{100, 999}, {200, 42}});  // 上一步 100/200 都有采样

  int32_t* dev_input = nullptr;
  int64_t* dev_cur_pairs = nullptr;
  int64_t* dev_prev_ring = nullptr;
  H2DAlloc(&dev_input, host_input);
  H2DAlloc(&dev_cur_pairs, cur_pairs_host);
  H2DAlloc(&dev_prev_ring, prev_ring_host);

  LaunchReplaceLastTokenKernel(dev_input, dev_cur_pairs, dev_prev_ring, /*batch_size=*/2, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  std::vector<int32_t> host_result(host_input.size());
  ASSERT_EQ(cudaMemcpy(host_result.data(), dev_input, host_result.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            cudaSuccess);

  // prefill req_A: 末位是真实 token 5, 自检不替换 (即使 prev_ring 里有 999 也不动).
  EXPECT_EQ(host_result[0], 1);
  EXPECT_EQ(host_result[1], 2);
  EXPECT_EQ(host_result[2], 3);
  EXPECT_EQ(host_result[3], 5);
  // decode req_B: 末位是 -1, 命中 prev_ring 替换为 42.
  EXPECT_EQ(host_result[4], 42);

  cudaFree(dev_input);
  cudaFree(dev_cur_pairs);
  cudaFree(dev_prev_ring);
}

// WriteRing 空 batch：batch_size<=0 时 LaunchWriteRingKernel 直接 return，ring 内容不变（与 Replace 空 batch 一致）。
TEST_F(ReplaceLastTokenTest, WriteRingEmptyBatchNoOp) {
  std::vector<int64_t> ring_init(9, -1);
  int64_t* dev_ring = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_ring, ring_init.size() * sizeof(int64_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_ring, ring_init.data(), ring_init.size() * sizeof(int64_t), cudaMemcpyHostToDevice),
            cudaSuccess);

  LaunchWriteRingKernel(dev_ring, /*cur_pairs=*/nullptr, /*sampled=*/nullptr, /*batch_size=*/0, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  int64_t count = 0;
  ASSERT_EQ(cudaMemcpy(&count, dev_ring, sizeof(int64_t), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(count, -1);
  cudaFree(dev_ring);
}

// prev_ring count=0 时 Replace 应保持 placeholder 不变。
TEST_F(ReplaceLastTokenTest, ReplaceWithZeroCountRingKeepsPlaceholder) {
  std::vector<int32_t> host_input = {-1, -1};
  auto cur_pairs_host = BuildCurPairs({{10, 0}, {20, 1}});
  std::vector<int64_t> prev_ring_host = {0};  // count=0

  int32_t* dev_input = nullptr;
  int64_t* dev_cur_pairs = nullptr;
  int64_t* dev_prev_ring = nullptr;
  H2DAlloc(&dev_input, host_input);
  H2DAlloc(&dev_cur_pairs, cur_pairs_host);
  H2DAlloc(&dev_prev_ring, prev_ring_host);

  LaunchReplaceLastTokenKernel(dev_input, dev_cur_pairs, dev_prev_ring, /*batch_size=*/2, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  std::vector<int32_t> host_result(host_input.size());
  ASSERT_EQ(cudaMemcpy(host_result.data(), dev_input, host_result.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(host_result[0], -1);
  EXPECT_EQ(host_result[1], -1);

  cudaFree(dev_input);
  cudaFree(dev_cur_pairs);
  cudaFree(dev_prev_ring);
}

// 边界: batch_size = 0 时 kernel 应当 no-op, 不调用 grid launch.
TEST_F(ReplaceLastTokenTest, EmptyBatchNoOp) {
  int32_t* dev_input = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_input, sizeof(int32_t)), cudaSuccess);
  int32_t fill = -1;
  ASSERT_EQ(cudaMemcpy(dev_input, &fill, sizeof(int32_t), cudaMemcpyHostToDevice), cudaSuccess);

  LaunchReplaceLastTokenKernel(dev_input, /*cur_pairs=*/nullptr, /*prev_ring=*/nullptr, /*batch_size=*/0, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  int32_t result = 0;
  ASSERT_EQ(cudaMemcpy(&result, dev_input, sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(result, -1);
  cudaFree(dev_input);
}

// 大批量: 256 个 req, 验证多 block 路径与边界 thread 正确性.
// cur_pairs 中 req_id 与 prev_ring 顺序相同 (恒等), 简单验证 kernel 在大 B_prev 下扫描正确.
TEST_F(ReplaceLastTokenTest, LargeBatch) {
  constexpr int kBatch = 256;
  // 每 req 占 input_ids 中 2 个槽 (一个上下文 + 一个 -1 末位); 模拟混合分布的安全场景.
  constexpr int kStride = 2;
  constexpr int kTotal = kBatch * kStride;

  std::vector<int32_t> host_input(kTotal, -1);
  // 让上下文位置非 -1 以验证不被误改.
  for (int i = 0; i < kBatch; ++i) {
    host_input[i * kStride] = 7;  // 任意非 -1 值
  }

  std::vector<int64_t> cur_pairs_host(2 * kBatch);
  std::vector<int64_t> prev_ring_host(1 + 2 * kBatch);
  prev_ring_host[0] = kBatch;
  for (int i = 0; i < kBatch; ++i) {
    const int64_t req_id = 1000 + i;
    const int64_t last_off = i * kStride + (kStride - 1);
    cur_pairs_host[2 * i] = req_id;
    cur_pairs_host[2 * i + 1] = last_off;
    prev_ring_host[1 + 2 * i] = req_id;
    prev_ring_host[1 + 2 * i + 1] = static_cast<int64_t>(i + 50000);  // 期望回填值
  }

  int32_t* dev_input = nullptr;
  int64_t* dev_cur_pairs = nullptr;
  int64_t* dev_prev_ring = nullptr;
  H2DAlloc(&dev_input, host_input);
  H2DAlloc(&dev_cur_pairs, cur_pairs_host);
  H2DAlloc(&dev_prev_ring, prev_ring_host);

  LaunchReplaceLastTokenKernel(dev_input, dev_cur_pairs, dev_prev_ring, kBatch, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  std::vector<int32_t> host_result(kTotal);
  ASSERT_EQ(cudaMemcpy(host_result.data(), dev_input, kTotal * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);

  for (int i = 0; i < kBatch; ++i) {
    EXPECT_EQ(host_result[i * kStride], 7) << "context position polluted at req " << i;
    EXPECT_EQ(host_result[i * kStride + (kStride - 1)], i + 50000) << "req " << i << " mismatch";
  }

  cudaFree(dev_input);
  cudaFree(dev_cur_pairs);
  cudaFree(dev_prev_ring);
}

// WriteRing 单测: 验证 ring 头部 count + (req_id, token) 配对正确写入.
TEST_F(ReplaceLastTokenTest, WriteRingBasic) {
  constexpr int kBatch = 4;
  std::vector<int64_t> cur_pairs_host = {7, 11, 8, 22, 9, 33, 10, 44};  // (7,_) (8,_) (9,_) (10,_) (offset 不读)
  std::vector<uint32_t> sampled_host = {100u, 200u, 300u, 400u};

  // ring buffer: 1 + 2*kBatch int64; 预填脏数据验证被覆盖.
  std::vector<int64_t> ring_init(1 + 2 * kBatch, -777);
  int64_t* dev_ring = nullptr;
  int64_t* dev_cur_pairs = nullptr;
  uint32_t* dev_sampled = nullptr;
  H2DAlloc(&dev_ring, ring_init);
  H2DAlloc(&dev_cur_pairs, cur_pairs_host);
  H2DAlloc(&dev_sampled, sampled_host);

  LaunchWriteRingKernel(dev_ring, dev_cur_pairs, dev_sampled, kBatch, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  std::vector<int64_t> ring_result(ring_init.size());
  ASSERT_EQ(cudaMemcpy(ring_result.data(), dev_ring, ring_result.size() * sizeof(int64_t), cudaMemcpyDeviceToHost),
            cudaSuccess);

  EXPECT_EQ(ring_result[0], kBatch);  // count 头部
  EXPECT_EQ(ring_result[1], 7);
  EXPECT_EQ(ring_result[2], 100);
  EXPECT_EQ(ring_result[3], 8);
  EXPECT_EQ(ring_result[4], 200);
  EXPECT_EQ(ring_result[5], 9);
  EXPECT_EQ(ring_result[6], 300);
  EXPECT_EQ(ring_result[7], 10);
  EXPECT_EQ(ring_result[8], 400);

  cudaFree(dev_ring);
  cudaFree(dev_cur_pairs);
  cudaFree(dev_sampled);
}

// 闭环测试：WriteRing → ReplaceLastToken 跨步衔接，验证 build-ahead 端到端契约。
//   step N : sampling 输出 sampled_tokens, WriteRing 写入 slot 0
//   step N+1: cur_pairs 用同样 (req_id), placeholder=-1, ReplaceLastToken 从 slot 0 中按 req_id 命中替换
TEST_F(ReplaceLastTokenTest, WriteRingThenReplaceClosedLoop) {
  // step N: 3 个 req
  std::vector<int64_t> stepN_pairs = {10, 0, 20, 0, 30, 0};  // offset 占位 (WriteRing 不读)
  std::vector<uint32_t> stepN_sampled = {111u, 222u, 333u};

  int64_t* dev_ring = nullptr;
  int64_t* dev_stepN_pairs = nullptr;
  uint32_t* dev_stepN_sampled = nullptr;
  std::vector<int64_t> ring_init(1 + 2 * 8, 0);  // 预留更大容量
  H2DAlloc(&dev_ring, ring_init);
  H2DAlloc(&dev_stepN_pairs, stepN_pairs);
  H2DAlloc(&dev_stepN_sampled, stepN_sampled);

  LaunchWriteRingKernel(dev_ring, dev_stepN_pairs, dev_stepN_sampled, /*batch_size=*/3, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  // step N+1: batch 重排为 30, 10, 20, 全部 placeholder; 期望命中 ring 各取自己上一步采样.
  std::vector<int32_t> stepN1_input = {-1, -1, -1};
  std::vector<int64_t> stepN1_pairs = {30, 0, 10, 1, 20, 2};
  int32_t* dev_stepN1_input = nullptr;
  int64_t* dev_stepN1_pairs = nullptr;
  H2DAlloc(&dev_stepN1_input, stepN1_input);
  H2DAlloc(&dev_stepN1_pairs, stepN1_pairs);

  LaunchReplaceLastTokenKernel(dev_stepN1_input, dev_stepN1_pairs, dev_ring, /*batch_size=*/3, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  std::vector<int32_t> result(stepN1_input.size());
  ASSERT_EQ(cudaMemcpy(result.data(), dev_stepN1_input, result.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            cudaSuccess);

  EXPECT_EQ(result[0], 333);  // req=30 → 333
  EXPECT_EQ(result[1], 111);  // req=10 → 111
  EXPECT_EQ(result[2], 222);  // req=20 → 222

  cudaFree(dev_ring);
  cudaFree(dev_stepN_pairs);
  cudaFree(dev_stepN_sampled);
  cudaFree(dev_stepN1_input);
  cudaFree(dev_stepN1_pairs);
}

}  // namespace ksana_llm
