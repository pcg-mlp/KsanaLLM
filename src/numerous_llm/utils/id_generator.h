/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>

namespace numerous_llm {

class IdGenerator {
 public:
  int64_t Gen() {
    ++id_;
    if (id_ == max_) {
      id_ = 0;
    }

    return id_;
  }

 private:
  // default to zero.
  std::atomic_int64_t id_ = 0;

  // max id value.
  int64_t max_ = std::numeric_limits<int64_t>::max();
};

}  // namespace numerous_llm
