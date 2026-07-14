/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ksana_llm {

struct GrammarOutput {
  // 需要结构化输出的请求ID列表
  std::vector<int64_t> structured_output_request_ids;

  // Bitmask 数据: [batch_size, bitmask_elements]
  // bitmask_elements = ceil(vocab_size / 32)
  std::vector<int32_t> grammar_bitmask;

  // 辅助信息
  size_t batch_size = 0;
  size_t bitmask_elements = 0;
  size_t vocab_size = 0;

  // 每个请求对应的 logits offset
  std::vector<size_t> logits_offsets;

  bool IsEmpty() const { return structured_output_request_ids.empty(); }

  // 用于 IPC 序列化
  void Serialize(std::vector<uint8_t>& buffer) const;
  void Deserialize(const std::vector<uint8_t>& buffer);
};

}  // namespace ksana_llm