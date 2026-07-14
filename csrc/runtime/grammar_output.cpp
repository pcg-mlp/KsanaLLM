/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/grammar_output.h"

#include <cstring>

namespace ksana_llm {

void GrammarOutput::Serialize(std::vector<uint8_t>& buffer) const {
  buffer.clear();

  // 计算所需大小：4个 size_t + 3个动态数组
  const size_t req_count = structured_output_request_ids.size();
  const size_t bitmask_size = grammar_bitmask.size();

  const size_t total_size = sizeof(size_t) * 4 +             // 4个元数据字段
                            req_count * sizeof(int64_t) +    // request_ids
                            req_count * sizeof(size_t) +     // logits_offsets
                            bitmask_size * sizeof(int32_t);  // grammar_bitmask

  buffer.resize(total_size);
  uint8_t* ptr = buffer.data();

  // 写入元数据
  std::memcpy(ptr, &batch_size, sizeof(size_t));
  ptr += sizeof(size_t);
  std::memcpy(ptr, &bitmask_elements, sizeof(size_t));
  ptr += sizeof(size_t);
  std::memcpy(ptr, &vocab_size, sizeof(size_t));
  ptr += sizeof(size_t);
  std::memcpy(ptr, &req_count, sizeof(size_t));
  ptr += sizeof(size_t);

  // 写入动态数组
  if (req_count > 0) {
    std::memcpy(ptr, structured_output_request_ids.data(), req_count * sizeof(int64_t));
    ptr += req_count * sizeof(int64_t);

    std::memcpy(ptr, logits_offsets.data(), req_count * sizeof(size_t));
    ptr += req_count * sizeof(size_t);
  }

  if (bitmask_size > 0) {
    std::memcpy(ptr, grammar_bitmask.data(), bitmask_size * sizeof(int32_t));
  }
}

void GrammarOutput::Deserialize(const std::vector<uint8_t>& buffer) {
  if (buffer.empty()) {
    return;
  }

  const uint8_t* ptr = buffer.data();

  // 读取元数据
  std::memcpy(&batch_size, ptr, sizeof(size_t));
  ptr += sizeof(size_t);
  std::memcpy(&bitmask_elements, ptr, sizeof(size_t));
  ptr += sizeof(size_t);
  std::memcpy(&vocab_size, ptr, sizeof(size_t));
  ptr += sizeof(size_t);

  size_t req_count;
  std::memcpy(&req_count, ptr, sizeof(size_t));
  ptr += sizeof(size_t);

  // 读取动态数组（批量复制）
  structured_output_request_ids.resize(req_count);
  logits_offsets.resize(req_count);

  if (req_count > 0) {
    std::memcpy(structured_output_request_ids.data(), ptr, req_count * sizeof(int64_t));
    ptr += req_count * sizeof(int64_t);

    std::memcpy(logits_offsets.data(), ptr, req_count * sizeof(size_t));
    ptr += req_count * sizeof(size_t);
  }

  const size_t bitmask_size = batch_size * bitmask_elements;
  grammar_bitmask.resize(bitmask_size);

  if (bitmask_size > 0) {
    std::memcpy(grammar_bitmask.data(), ptr, bitmask_size * sizeof(int32_t));
  }
}

}  // namespace ksana_llm