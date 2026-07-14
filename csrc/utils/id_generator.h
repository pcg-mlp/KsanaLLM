/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>

namespace ksana_llm {

class IdGenerator {
 public:
  int64_t Gen() {
    // Keep the highest sign bit to 0.
    return static_cast<int64_t>(id_.fetch_add(1, std::memory_order_relaxed) & ((1ul << 63) - 1));
  }

 private:
  // default to zero.
  std::atomic_size_t id_ = 0;
};

}  // namespace ksana_llm
