/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/nvidia/dsv4_tail_copy.h"

#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace ksana_llm {

namespace {

// 单线程一次拷贝 16B (uint4)。per_token_bytes 随 compressor_dtype 变化:
//   FP32: per_token_bytes = 2*coff*head_dim*4; c4→16*head_dim, c128→8*head_dim
//   BF16: per_token_bytes = 2*coff*head_dim*2; c4→ 8*head_dim, c128→4*head_dim
// head_dim ∈ {128, 512} 保证始终 16 字节对齐，由调用方保证。
constexpr int kVecBytes = 16;

constexpr int kC4Ratio = 4;

__global__ void CopyTailToStateKernel(uint8_t* __restrict__ state_layer_base,
                                      const uint8_t* __restrict__ tail_layer_base,
                                      const int32_t* __restrict__ tail_read_block,
                                      const int32_t* __restrict__ tail_target_indices, int num_req,
                                      int64_t block_cache_size, int per_token_bytes, int tail_tokens) {
  // grid = num_req * tail_tokens, 每个 block 处理 1 个 (req, slot) pair。
  const int bid = blockIdx.x;
  const int req_idx = bid / tail_tokens;
  const int slot_idx = bid % tail_tokens;
  if (req_idx >= num_req) return;

  const int32_t block_id = tail_read_block[req_idx];
  if (block_id < 0) return;  // block id sentinel, skip request

  const int32_t target_page = tail_target_indices[req_idx];
  const int64_t state_per_batch_per_slot_bytes = static_cast<int64_t>(per_token_bytes) * kC4Ratio;

  const uint8_t* src = tail_layer_base + static_cast<int64_t>(block_id) * block_cache_size + slot_idx * per_token_bytes;
  uint8_t* dst = state_layer_base + static_cast<int64_t>(target_page) * state_per_batch_per_slot_bytes +
                 slot_idx * per_token_bytes;

  const int num_vecs = per_token_bytes / kVecBytes;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int v = threadIdx.x; v < num_vecs; v += blockDim.x) {
    reinterpret_cast<uint4*>(dst)[v] = reinterpret_cast<const uint4*>(src)[v];
  }
}

__global__ void CopyKvScoreToTailKernel(uint8_t* __restrict__ tail_layer_base, const uint8_t* __restrict__ kv_score,
                                        const int32_t* __restrict__ tail_write_indices, int total_tokens,
                                        int64_t block_cache_size, int per_token_bytes) {
  const int tid = blockIdx.x;
  if (tid >= total_tokens) return;

  const int32_t packed = tail_write_indices[tid];
  if (packed < 0) return;  // token not in block tail slots
  const int32_t block_id = packed >> 2;
  const int32_t slot_idx = packed & 3;

  const uint8_t* src = kv_score + static_cast<int64_t>(tid) * per_token_bytes;
  uint8_t* dst = tail_layer_base + static_cast<int64_t>(block_id) * block_cache_size + slot_idx * per_token_bytes;

  const int num_vecs = per_token_bytes / kVecBytes;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int v = threadIdx.x; v < num_vecs; v += blockDim.x) {
    reinterpret_cast<uint4*>(dst)[v] = reinterpret_cast<const uint4*>(src)[v];
  }
}

}  // namespace

void CopyTailToState(void* state_layer_base, const void* tail_layer_base, const int32_t* tail_read_block,
                     const int32_t* tail_target_indices, int num_req, int64_t block_cache_size, int per_token_bytes,
                     int tail_tokens, cudaStream_t stream) {
  if (num_req <= 0 || tail_tokens <= 0 || per_token_bytes <= 0) return;
  // per_token_bytes 须是 16 字节对齐 (caller-enforced)。
  const int num_blocks = num_req * tail_tokens;
  // 64 threads/block: num_vecs = per_token_bytes/16, 每线程循环 1~8 次。
  constexpr int kThreadsPerBlock = 64;
  constexpr auto kernel = CopyTailToStateKernel;
  LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kThreadsPerBlock, 0, stream, reinterpret_cast<uint8_t*>(state_layer_base),
                         reinterpret_cast<const uint8_t*>(tail_layer_base), tail_read_block, tail_target_indices,
                         num_req, block_cache_size, per_token_bytes, tail_tokens);
}

void CopyKvScoreToTail(void* tail_layer_base, const void* kv_score, const int32_t* tail_write_indices, int total_tokens,
                       int64_t block_cache_size, int per_token_bytes, cudaStream_t stream) {
  if (total_tokens <= 0 || per_token_bytes <= 0) return;
  constexpr int kThreadsPerBlock = 64;
  constexpr auto kernel = CopyKvScoreToTailKernel;
  LAUNCH_KERNEL_WITH_PDL(kernel, total_tokens, kThreadsPerBlock, 0, stream, reinterpret_cast<uint8_t*>(tail_layer_base),
                         reinterpret_cast<const uint8_t*>(kv_score), tail_write_indices, total_tokens, block_cache_size,
                         per_token_bytes);
}

}  // namespace ksana_llm
