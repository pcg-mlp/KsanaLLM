/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/include/sgl_kernel/type.cuh
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
#include "sgl_kernel/utils.cuh"

template <typename T>
struct dtype_trait {};

#define SGL_REGISTER_DTYPE_TRAIT(TYPE, PACK2, ...)  \
  template <>                                       \
  struct dtype_trait<TYPE> {                        \
    using self_t = TYPE;                            \
    using packed_t = PACK2;                         \
    template <typename S>                           \
    SGL_DEVICE static self_t from(const S& value) { \
      return static_cast<TYPE>(value);              \
    }                                               \
    __VA_ARGS__                                     \
  }

#define SGL_REGISTER_TYPE_END static_assert(true)

#define SGL_REGISTER_FROM_FUNCTION(FROM, FN)                     \
  SGL_DEVICE static self_t from(const FROM& x) { return FN(x); } \
  static_assert(true)

#define SGL_REGISTER_UNARY_FUNCTION(NAME, FN)                      \
  SGL_DEVICE static self_t NAME(const self_t& x) { return FN(x); } \
  static_assert(true)

#define SGL_REGISTER_BINARY_FUNCTION(NAME, FN)                                         \
  SGL_DEVICE static self_t NAME(const self_t& x, const self_t& y) { return FN(x, y); } \
  static_assert(true)

SGL_REGISTER_DTYPE_TRAIT(fp32_t, fp32x2_t, SGL_REGISTER_TYPE_END;  //
                         SGL_REGISTER_FROM_FUNCTION(fp16_t, __half2float);
                         SGL_REGISTER_FROM_FUNCTION(bf16_t, __bfloat162float); SGL_REGISTER_UNARY_FUNCTION(abs, fabsf);
                         SGL_REGISTER_UNARY_FUNCTION(sqrt, sqrtf); SGL_REGISTER_UNARY_FUNCTION(rsqrt, rsqrtf);
                         SGL_REGISTER_BINARY_FUNCTION(max, fmaxf); SGL_REGISTER_BINARY_FUNCTION(min, fminf););
SGL_REGISTER_DTYPE_TRAIT(fp16_t, fp16x2_t);
SGL_REGISTER_DTYPE_TRAIT(bf16_t, bf16x2_t);

/// TODO: Add ROCM implementation
SGL_REGISTER_DTYPE_TRAIT(fp32x2_t, fp32x4_t, SGL_REGISTER_TYPE_END;
                         SGL_REGISTER_FROM_FUNCTION(fp16x2_t, __half22float2);
                         SGL_REGISTER_FROM_FUNCTION(bf16x2_t, __bfloat1622float2););

SGL_REGISTER_DTYPE_TRAIT(fp16x2_t, void, SGL_REGISTER_TYPE_END;
                         SGL_REGISTER_FROM_FUNCTION(fp32x2_t, __float22half2_rn););

SGL_REGISTER_DTYPE_TRAIT(bf16x2_t, void, SGL_REGISTER_TYPE_END;
                         SGL_REGISTER_FROM_FUNCTION(fp32x2_t, __float22bfloat162_rn););

#undef SGL_REGISTER_DTYPE_TRAIT
#undef SGL_REGISTER_FROM_FUNCTION

template <typename T>
using packed_t = typename dtype_trait<T>::packed_t;

namespace device {

template <typename To, typename From>
SGL_DEVICE To cast(const From& value) {
  return dtype_trait<To>::from(value);
}

}  // namespace device
