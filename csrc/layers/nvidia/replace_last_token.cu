/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/

#include "layers/nvidia/replace_last_token.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace ksana_llm {

namespace {

// 替换 placeholder kernel.
// 每个 thread 处理本步一个 req: 取 (req_id, dst_off) 自检 placeholder
// (input_ids[dst_off] == -1), 命中则在 prev_ring 中按 req_id 线性扫描查找上一步
// 采样写入的 token，找到后回填；未命中保持 -1（正常路径下不应发生，kernel 作保护性兜底）。
__global__ void ReplaceLastTokenKernel(int32_t* __restrict__ input_ids, const int64_t* __restrict__ cur_pairs,
                                       const int64_t* __restrict__ prev_ring, int batch_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size) {
    return;
  }
  const int64_t target = cur_pairs[2 * idx];
  const int64_t dst_off = cur_pairs[2 * idx + 1];

  // 末位非 placeholder (例如 prefill / chunked prefill 中间步): 不动.
  if (input_ids[dst_off] != -1) {
    return;
  }

  // ring 头部自描述 B_prev, 单线程读后线性扫描.
  const int B_prev = static_cast<int>(prev_ring[0]);
  for (int j = 0; j < B_prev; ++j) {
    if (prev_ring[1 + 2 * j] == target) {
      input_ids[dst_off] = static_cast<int32_t>(prev_ring[1 + 2 * j + 1]);
      return;
    }
  }
  // 未命中: input_ids[dst_off] 保持 -1.
}

// 写 ring kernel.
// thread 0 同时负责写 count 头部 (ring[0] = B_cur), 所有 thread 写自己的 (req_id, token) 对.
__global__ void WriteRingKernel(int64_t* __restrict__ ring, const int64_t* __restrict__ cur_pairs,
                                const uint32_t* __restrict__ sampled_tokens, int batch_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size) {
    return;
  }
  if (idx == 0) {
    ring[0] = static_cast<int64_t>(batch_size);
  }
  ring[1 + 2 * idx] = cur_pairs[2 * idx];
  ring[1 + 2 * idx + 1] = static_cast<int64_t>(sampled_tokens[idx]);
}

}  // namespace

void LaunchReplaceLastTokenKernel(int32_t* input_ids, const int64_t* cur_pairs, const int64_t* prev_ring,
                                  int batch_size, cudaStream_t stream) {
  if (batch_size <= 0) {
    return;
  }
  constexpr int kThreadsPerBlock = 128;
  const int num_blocks = (batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
  ReplaceLastTokenKernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(input_ids, cur_pairs, prev_ring, batch_size);
}

void LaunchWriteRingKernel(int64_t* ring, const int64_t* cur_pairs, const uint32_t* sampled_tokens, int batch_size,
                           cudaStream_t stream) {
  if (batch_size <= 0) {
    return;
  }
  constexpr int kThreadsPerBlock = 128;
  const int num_blocks = (batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
  WriteRingKernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(ring, cur_pairs, sampled_tokens, batch_size);
}

}  // namespace ksana_llm
