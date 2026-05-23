/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ksana_llm {

// DeepSeek V4 prefix cache 的 c4 compressor overlap tail 持久化拷贝。
//
// tail 区是每个 KV block 末尾的两段固定大小 region (main / indexer)，存放该 block
// 最后 4 个 token 的 FP32 kv_score。一次 prefill 的数据流:
//
//   KV pool tail  ──CopyTailToState──►  transient c4_state_buffer  ──►  compressor
//                                                                       │
//                                                                       ▼
//   KV pool tail  ◄──CopyKvScoreToTail──  新 token kv_score (仅 block 末 4 槽)
//
// compressor kernel 不直接访问 tail 区: read-copy 把 prefix 末 4 token 的 kv_score
// 加载到 transient C4-state / indexer-state buffer 作为 overlap；write-copy 在
// compressor 之后把当前 step 的 kv_score 持久化到 tail 区，供下个请求复用。

// Copy 持久 tail 区到 transient state buffer (read-copy)。
// 拷贝 [num_req, tail_tokens] 个 token，每个 token per_token_bytes 字节。
//
// Params:
//   state_layer_base       当前 layer 在 state buffer 中的基址 (已加 layer 偏移)。
//   tail_layer_base        当前 layer 在 KV pool tail 区的基址 (已加 layer 偏移)。
//   tail_read_block[r]     第 r 个请求 prefix 命中 block 的物理 id；<0 表示跳过。
//   tail_target_indices[r] 第 r 个请求在 state buffer 中的目标 page index
//                          (= batch_slot * kc4_slot + sub_slot)；与 tail_read_block 配合使用。
//
// 字节寻址:
//   src = tail_layer_base + tail_read_block[r] * block_cache_size + slot * per_token_bytes
//   dst = state_layer_base + tail_target_indices[r] * (per_token_bytes * kC4Ratio) + slot * per_token_bytes
void CopyTailToState(void* state_layer_base, const void* tail_layer_base, const int32_t* tail_read_block,
                     const int32_t* tail_target_indices, int num_req, int64_t block_cache_size, int per_token_bytes,
                     int tail_tokens, cudaStream_t stream);

// Copy 当前 step 的 kv_score 到持久 tail 区 (write-copy)。
// 遍历 total_tokens 个 token，仅在 tail_write_indices[i] >= 0 时拷贝。
//
// Params:
//   tail_layer_base          当前 layer 的 tail 区基址 (已加 layer 偏移)。
//   kv_score                 FP32 [total_tokens, per_token_bytes/sizeof(float)]，连续。
//   tail_write_indices[i]    packed (block_id, slot) for the i-th token.
//                            Encoding: block_id * kTailTokens | slot  (slot uses lower 2 bits).
//                            Value < 0 means skip (token not in a tail position).
//
// 字节寻址:
//   block_id = tail_write_indices[i] >> 2
//   slot_idx = tail_write_indices[i] & 3
//   src = kv_score + i * per_token_bytes
//   dst = tail_layer_base + block_id * block_cache_size + slot_idx * per_token_bytes
void CopyKvScoreToTail(void* tail_layer_base, const void* kv_score, const int32_t* tail_write_indices, int total_tokens,
                       int64_t block_cache_size, int per_token_bytes, cudaStream_t stream);

}  // namespace ksana_llm
