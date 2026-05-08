/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>

namespace ksana_llm {

class IdGenerator {
 public:
  // Reserve the lowest kSubIdBits bits for output_group branch index.
  // When output_group > 1 (beam search / multi-return), each branch gets
  // a unique req_id = base_id | branch_index.
  static constexpr int kSubIdBits = 8;

  int64_t Gen() {
    // Keep the highest sign bit to 0, shift left to reserve low bits for sub-id.
    return static_cast<int64_t>((id_.fetch_add(1, std::memory_order_relaxed) & ((1ul << (63 - kSubIdBits)) - 1))
                                << kSubIdBits);
  }

 private:
  // default to zero.
  std::atomic_size_t id_ = 0;
};

}  // namespace ksana_llm
