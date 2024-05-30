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
T ClampInfForHalf(const float input) {
  return input;
}

template <>
half_float::half ClampInfForHalf(const float input) {
  // clamp inf values to enable fp16 training
  return input > 0.0f ? (half_float::half)std::min(input, HALF_FLT_MAX - 1000)
                      : (half_float::half)std::max(input, -HALF_FLT_MAX + 1000);
}

template <typename T>
void RmsNormRef(const T *input, const T *gamma, const float eps, const size_t m, const size_t n, T *output) {
  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    float var_sum = 0.0f;
    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
      var_sum += float(input[m_idx * n + n_idx] * input[m_idx * n + n_idx]);
    }
    float s_variance = 1.0f / std::sqrt(var_sum / (float)n + eps);
    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
      float value = (input[m_idx * n + n_idx] * s_variance) * float(gamma[n_idx]);
      if (std::is_same<T, half_float::half>::value) {
        output[m_idx * n + n_idx] = ClampInfForHalf<T>(value);
      } else if (std::is_same<T, float>::value) {
        output[m_idx * n + n_idx] = value;
      } else if (std::is_same<T, aclFloat16>::value) {
        output[m_idx * n + n_idx] = aclFloatToFloat16(value);
      } else {
        throw std::invalid_argument("Invalid rms norm compute type, only support float16 or float32.");
      }
    }
  }
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels