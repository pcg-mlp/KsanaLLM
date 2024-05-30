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
void CalcMatmulRef(T *a_ptr, T *b_ptr, T *c_ptr, size_t m, size_t n, size_t k) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      T sum = (T)(0.0);
      for (size_t s = 0; s < k; ++s) {
        sum += (a_ptr[i * k + s] * b_ptr[s * n + j]);
      }
      c_ptr[i * n + j] = sum;
    }
  }
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels