/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ksana_llm {

// =============================================================================
// build-ahead 采样 ring 协议（Executor 端）—— req_id 解耦，双 slot 防覆盖
// -----------------------------------------------------------------------------
// 设计要点:
//   1. ModelInput 在每步 PrepareInputIds 时, 将 (req_id, last_token_offset)
//      成对写入一个 int64 tensor: cur_pairs[2 * B_cur]
//      其中 last_token_offset = 本 req 在 flat input_ids 里的"末位 token"下标.
//        - decode req:  该 req 在 input_ids 中唯一 token 的下标
//        - prefill req: 该 req 最后一个输入 token 下标 (sampling 也按它取)
//
//   2. Sampler 持有两个独立 device ring slot, 每个布局:
//        ring_slot_dev : int64[1 + 2 * B_max]
//        layout: [count, req_id_0, token_0, req_id_1, token_1, ...]
//      其中 count 为"本步 WriteRingKernel 写入时的 B_cur".
//      slot 0 / slot 1 物理隔离, 由 ModelRunner 每步翻转, 避免 N 与 N+1 步互相覆盖.
//
//   3. 一个 step 的两次 launch:
//      (a) ReplaceLastTokenKernel: 替换 input_ids 中的 placeholder (-1)
//          - 输入: cur_pairs (本步), prev_ring (sampler.GetRingSlot(cur_slot ^ 1))
//          - 每个 thread 处理 1 个本步 req: 取 (req_id, dst_off);
//            若 input_ids[dst_off] == -1 则在 prev_ring 中按 req_id 线性扫描定位
//            上一步采样的 token, 命中后写入 input_ids[dst_off].
//      (b) WriteRingKernel: 把"本步 sampling 输出"写入当前 slot
//          - 输入: cur_pairs (本步, 仅取 req_id 部分), sampled_tokens (本步)
//          - thread 0 写 ring[0] = B_cur (count 头部);
//            其余 thread 写 (req_id, token) 对.
//
//   4. 首步 (cold start): Sampler 初始化时把两个 slot 全置 0,
//      ring[0] = 0, ReplaceKernel 自然循环零次, placeholder 保持 -1
//      （首步 ring count=0，ReplaceKernel 自然 no-op，不会残留 -1）。
//
// 调用方负责通过 cudaEvent / cudaStreamWaitEvent 保证:
//   - ReplaceLastTokenKernel: prev_ring 所在 slot 的上一步 WriteRing 已完成 (ring_write_event);
//                             input_ids 已被 PrepareInputIds 写好.
//   - WriteRingKernel        : sampled_tokens 已被本步 sampling 写好;
//                             目标 slot 在本步 forward 入口未被读 (slot 翻转保证).
// =============================================================================

// 替换 placeholder (-1) 为上一步 sampling 写入 ring 的 token.
//
// Params:
//   input_ids   int32, 本步 ModelInput 的输入 token id (含若干 -1 placeholder).
//   cur_pairs   int64[2 * batch_size], 本步 (req_id, last_token_offset) 交错布局.
//   prev_ring   int64[1 + 2 * B_prev], 上一步 sampling 写入的 ring slot,
//               布局 [count, req_id_0, token_0, req_id_1, token_1, ...].
//   batch_size  本步 forward_reqs 个数 (B_cur).
//   stream      CUDA stream, 调用方保证已对 input_ids / prev_ring 做必要 wait.
void LaunchReplaceLastTokenKernel(int32_t* input_ids, const int64_t* cur_pairs, const int64_t* prev_ring,
                                  int batch_size, cudaStream_t stream);

// 把本步 sampling 输出写入指定 ring slot (含 count 头部).
//
// Params:
//   ring            int64[1 + 2 * batch_size], 待写入的 ring slot 起始指针.
//   cur_pairs       int64[2 * batch_size], 本步 (req_id, last_offset) 交错布局,
//                   仅 req_id 部分被读取, offset 部分忽略.
//   sampled_tokens  uint32[batch_size], 本步 sampling 输出 (vocab id, 必然非负).
//   batch_size      本步 forward_reqs 个数 (B_cur).
//   stream          CUDA stream, 调用方保证已对 sampled_tokens 做必要 wait.
void LaunchWriteRingKernel(int64_t* ring, const int64_t* cur_pairs, const uint32_t* sampled_tokens, int batch_size,
                           cudaStream_t stream);

}  // namespace ksana_llm
