/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/include/sgl_kernel/vec.cuh
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

#include <cstddef>
#include <cstdint>

namespace device {

namespace details {

template <std::size_t N>
struct uint_trait {};

template <>
struct uint_trait<1> {
  using type = uint8_t;
};

template <>
struct uint_trait<2> {
  using type = uint16_t;
};

template <>
struct uint_trait<4> {
  using type = uint32_t;
};

template <>
struct uint_trait<8> {
  using type = uint64_t;
};

template <typename T>
using sized_int = typename uint_trait<sizeof(T)>::type;

}  // namespace details

template <typename T, std::size_t N>
struct alignas(sizeof(T) * N) AlignedStorage {
  T data[N];
};

template <typename T, std::size_t N>
struct AlignedVector {
 private:
  /// NOTE: 1. must be pow of two 2. 16 * 8 = 128 byte, which is the max vector size supported by most devices
  static_assert((N > 0 && (N & (N - 1)) == 0) && sizeof(T) * N <= 16, "CUDA only support at most 128B vector op");
  using element_t = typename details::sized_int<T>;
  using storage_t = AlignedStorage<element_t, N>;

 public:
  SGL_DEVICE void load(const void* ptr, std::size_t offset = 0) {
    m_storage = reinterpret_cast<const storage_t*>(ptr)[offset];
  }
  SGL_DEVICE void store(void* ptr, std::size_t offset = 0) const {
    reinterpret_cast<storage_t*>(ptr)[offset] = m_storage;
  }
  SGL_DEVICE void fill(T value) {
    const auto store_value = *reinterpret_cast<element_t*>(&value);
#pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      m_storage.data[i] = store_value;
    }
  }

  SGL_DEVICE auto operator[](std::size_t idx) -> T& { return reinterpret_cast<T*>(&m_storage)[idx]; }
  SGL_DEVICE auto operator[](std::size_t idx) const -> T { return reinterpret_cast<const T*>(&m_storage)[idx]; }
  SGL_DEVICE auto data() -> T* { return reinterpret_cast<T*>(&m_storage); }
  SGL_DEVICE auto data() const -> const T* { return reinterpret_cast<const T*>(&m_storage); }

 private:
  storage_t m_storage;
};

}  // namespace device
