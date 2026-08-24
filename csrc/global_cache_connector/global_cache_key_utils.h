/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "global_cache_connector/sha256_lite.h"

namespace ksana_llm {

/**
 * @brief SHA256 hash of raw token prefix bytes, returned as 16-char hex string.
 *
 * Uses SHA256 for collision safety (aligned with vLLM v0.11 default).
 * Takes the first 8 bytes (64-bit equivalent) of the digest to keep keys compact.
 */
inline std::string DigestToHex16(const uint8_t digest[32]) {
  char hex[17];
  for (int i = 0; i < 8; ++i) {
    snprintf(hex + i * 2, 3, "%02x", digest[i]);
  }
  return std::string(hex, 16);
}

inline std::string HashTokenPrefix(const int* data, size_t len) {
  uint8_t digest[32];
  sha256_detail::Sha256(reinterpret_cast<const uint8_t*>(data), len * sizeof(int), digest);
  return DigestToHex16(digest);
}

/**
 * @brief 按 block 增量 SHA256，产出 [start_block, end_block) 每个 block 的 16-hex prefix hash。
 *
 * out_hashes[i] 对应 block (start_block + i)，语义等同
 * HashTokenPrefix(data, min((bi+1)*block_size, total_tokens))。
 * 复杂度 O(total_tokens)，避免对每个 prefix 全量重算。
 */
inline void HashTokenPrefixesIncremental(const int* data, int total_tokens, int block_size, int start_block,
                                         int end_block, std::vector<std::string>& out_hashes) {
  out_hashes.clear();
  if (data == nullptr || total_tokens <= 0 || block_size <= 0 || start_block >= end_block) {
    return;
  }
  const int needed = end_block - start_block;
  out_hashes.reserve(static_cast<size_t>(needed));

  const int token_blocks = (total_tokens + block_size - 1) / block_size;
  // 全部落在 token 末尾之后：与 HashTokenPrefix(data, total_tokens) 一致，复用同一 digest
  if (start_block >= token_blocks) {
    const std::string h = HashTokenPrefix(data, static_cast<size_t>(total_tokens));
    out_hashes.assign(static_cast<size_t>(needed), h);
    return;
  }

  const int clamped_end = std::min(end_block, token_blocks);
  sha256_detail::Sha256Context ctx;
  const size_t seed_tok = static_cast<size_t>(start_block) * static_cast<size_t>(block_size);
  if (seed_tok > 0) {
    ctx.Update(reinterpret_cast<const uint8_t*>(data), seed_tok * sizeof(int));
  }

  for (int bi = start_block; bi < clamped_end; ++bi) {
    const size_t tok_start = static_cast<size_t>(bi) * static_cast<size_t>(block_size);
    const size_t tok_end =
        std::min(static_cast<size_t>(bi + 1) * static_cast<size_t>(block_size), static_cast<size_t>(total_tokens));
    if (tok_start >= tok_end) {
      break;
    }
    ctx.Update(reinterpret_cast<const uint8_t*>(data + tok_start), (tok_end - tok_start) * sizeof(int));
    uint8_t digest[32];
    ctx.FinalizeTo(digest);
    out_hashes.push_back(DigestToHex16(digest));
  }

  // 超出有效 token block 的尾部：旧逻辑仍用 prefix_len=total_tokens，digest 与最后一档相同
  while (static_cast<int>(out_hashes.size()) < needed) {
    out_hashes.push_back(out_hashes.back());
  }
}

inline std::string MakeBlockKey(const std::string& prefix_hash, int tp_rank, int layer_index, int block_idx) {
  return "kv_" + prefix_hash + "_tp" + std::to_string(tp_rank) + "_l" + std::to_string(layer_index) + "_b" +
         std::to_string(block_idx);
}

/**
 * @brief SWA block 的 key，与主 KV 共用同一个前缀哈希，只靠 "skv_" 前缀区分命名空间。
 *
 * 共用前缀哈希使得「主 KV 命中」与「SWA 命中」可以按同一个 block index 直接对齐比较，
 * 无需额外映射；主 KV 的 key 格式保持不变，存量 cache 不作废。
 */
inline std::string MakeSwaBlockKey(const std::string& prefix_hash, int tp_rank, int layer_index, int block_idx) {
  return "skv_" + prefix_hash + "_tp" + std::to_string(tp_rank) + "_l" + std::to_string(layer_index) + "_b" +
         std::to_string(block_idx);
}

inline std::string MakeFirstTokenKey(const std::string& prefix_hash) { return "ft_" + prefix_hash; }

}  // namespace ksana_llm
