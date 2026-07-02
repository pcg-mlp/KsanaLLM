/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Model perf 专用确定性 token 生成工具。
 * ==============================================================================*/
#pragma once

#include <cstdint>
#include <vector>

namespace ksana_llm {

// 使用 xorshift32 生成确定性 token 序列，token 取值范围为 [0, vocab)。
// 相同 (token_num, seed, vocab) 在多次执行、多轮 warmup/stat 中产出相同内容，保证 MoE 路由等结果可复现。
inline void GenerateTokens(size_t token_num, uint32_t seed, uint32_t vocab, std::vector<int>& tokens) {
  tokens.resize(token_num);
  const uint32_t vocab_size = vocab > 0 ? vocab : 1;
  uint32_t state = seed != 0 ? seed : 0x6d2b79f5u;
  for (size_t i = 0; i < token_num; ++i) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    tokens[i] = static_cast<int>(state % vocab_size);
  }
}

}  // namespace ksana_llm
