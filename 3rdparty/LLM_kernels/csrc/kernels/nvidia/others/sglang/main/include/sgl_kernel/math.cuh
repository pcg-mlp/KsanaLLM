/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/include/sgl_kernel/math.cuh
 *
 * Copyright 2023-2024 SGLang Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "sgl_kernel/type.cuh"

#include <cmath>

namespace device::math {

inline constexpr float log2e = 1.44269504088896340736f;
inline constexpr float loge2 = 0.693147180559945309417f;
inline constexpr float FP8_E4M3_MAX = 448.0f;
static_assert(log2e * loge2 == 1.0f, "log2e * loge2 must be 1");

template <typename T>
SGL_DEVICE T max(T a, T b) {
  return dtype_trait<T>::max(a, b);
}

template <typename T>
SGL_DEVICE T min(T a, T b) {
  return dtype_trait<T>::min(a, b);
}

template <typename T>
SGL_DEVICE T abs(T a) {
  return dtype_trait<T>::abs(a);
}

template <typename T>
SGL_DEVICE T sqrt(T a) {
  return dtype_trait<T>::sqrt(a);
}

template <typename T>
SGL_DEVICE T rsqrt(T a) {
  return dtype_trait<T>::rsqrt(a);
}

SGL_DEVICE float exp(float a) { return ::expf(a); }

SGL_DEVICE float sin(float a) { return ::sinf(a); }

SGL_DEVICE float cos(float a) { return ::cosf(a); }

}  // namespace device::math
