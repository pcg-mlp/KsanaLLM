/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cmath>

#include "3rdparty/half/include/half.hpp"

namespace llm_kernels {
namespace ascend {
namespace test {

template <typename T>
void ArgmaxRef(const T* input, uint32_t* output, const uint32_t batch_size, const uint32_t vocab_size) {
  for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    uint32_t max_index = 0;
    T max_value;
    if (std::is_same<T, half_float::half>::value) {
      max_value = std::numeric_limits<half_float::half>::min();
    } else if (std::is_same<T, float>::value) {
      max_value = std::numeric_limits<float>::min();
    } else if (std::is_same<T, aclFloat16>::value) {
      max_value = aclFloatToFloat16(std::numeric_limits<float>::min());
    } else {
      throw std::invalid_argument("Invalid embedding lookup type, only support float16 or float32.");
    }
    for (size_t vocab_idx = 0; vocab_idx < vocab_size; ++vocab_idx) {
      if (max_value < input[batch_idx * vocab_size + vocab_idx]) {
        max_value = input[batch_idx * vocab_size + vocab_idx];
        max_index = vocab_idx;
      }
    }
    output[batch_idx] = max_index;
  }
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels