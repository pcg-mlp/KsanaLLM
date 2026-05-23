#pragma once
#include <cstdint>
#include <cuda.h>


#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CEIL_DIV(a, b) ((a + b - 1) / (b))

#define STR(x) #x
#define PRAGMA_UNROLL _Pragma(STR(unroll))
#define PRAGMA_UNROLL_COUNT(n) _Pragma(STR(unroll n))
#define CUDA_INLINE __device__ __forceinline__


template <typename T>
CUDA_INLINE uint32_t cast_smem_ptr_to_uint(T *smem_ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
};

constexpr uint32_t static_next_power_of_2(uint32_t v) {
  if (v <= 1) return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

constexpr uint32_t get_max_load_bytes(uint32_t bytes) {
  if (bytes % 16 == 0) return 16;
  if (bytes % 8 == 0) return 8;
  if (bytes % 4 == 0) return 4;
  if (bytes % 2 == 0) return 2;
  return 1;
}

template <int bytes>
struct LoadTypeChooser {
  using Type = typename LoadTypeChooser<get_max_load_bytes(bytes)>::Type;
};
template <>
struct LoadTypeChooser<1> {
  using Type = uint8_t;
};
template <>
struct LoadTypeChooser<2> {
  using Type = uint16_t;
};
template <>
struct LoadTypeChooser<4> {
  using Type = uint32_t;
};
template <>
struct LoadTypeChooser<8> {
  using Type = uint2;
};
template <>
struct LoadTypeChooser<16> {
  using Type = uint4;
};

template <uint32_t M_, uint32_t N_, uint32_t K_>
struct Shape {
  static constexpr uint32_t M = M_;
  static constexpr uint32_t N = N_;
  static constexpr uint32_t K = K_;
};
