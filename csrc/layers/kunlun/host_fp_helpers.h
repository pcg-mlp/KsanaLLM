/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "device/device_types.h"  // ksana_llm::float16 / bfloat16 typedefs

// Host-side fp16/bf16 <-> fp32 conversion helpers used by Kunlun attention layers
// for the host-side RoPE workaround. Templated on KsanaLLM's float16/bfloat16
// (both 2-byte typedefs); we operate on the underlying uint16_t bits.
//
// fp16 round behaviour matches the prior in-place implementation: truncation of
// the 13 low mantissa bits (no round-to-nearest-even). bf16 also uses truncation.

namespace ksana_llm {
namespace kunlun_host {

template <typename T>
inline float HalfToFp32(uint16_t bits);

template <>
inline float HalfToFp32<float16>(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1u;
  uint32_t exp16 = (h >> 10) & 0x1Fu;
  uint32_t frac16 = h & 0x3FFu;
  if (exp16 == 0) {
    if (frac16 == 0) return sign ? -0.0f : 0.0f;
    float val = std::ldexp(static_cast<float>(frac16), -24);
    return sign ? -val : val;
  } else if (exp16 == 0x1F) {
    return sign ? -INFINITY : INFINITY;
  }
  int32_t exp32 = static_cast<int32_t>(exp16) - 15 + 127;
  uint32_t u32 = (sign << 31) | (static_cast<uint32_t>(exp32) << 23) | (frac16 << 13);
  float f;
  std::memcpy(&f, &u32, sizeof(float));
  return f;
}

template <>
inline float HalfToFp32<bfloat16>(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(float));
  return f;
}

template <typename T>
inline uint16_t Fp32ToHalf(float f);

template <>
inline uint16_t Fp32ToHalf<float16>(float f) {
  uint32_t u32;
  std::memcpy(&u32, &f, sizeof(uint32_t));
  uint32_t sign = (u32 >> 31) & 0x1u;
  int32_t exp32 = static_cast<int32_t>((u32 >> 23) & 0xFFu) - 127;
  uint32_t frac32 = (u32 >> 13) & 0x3FFu;
  if (exp32 <= -15) return static_cast<uint16_t>(sign << 15);
  if (exp32 >= 16) return static_cast<uint16_t>((sign << 15) | 0x7C00u);
  uint16_t exp16 = static_cast<uint16_t>((exp32 + 15) & 0x1F);
  return static_cast<uint16_t>((sign << 15) | (exp16 << 10) | frac32);
}

template <>
inline uint16_t Fp32ToHalf<bfloat16>(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(uint32_t));
  return static_cast<uint16_t>(u >> 16);
}

}  // namespace kunlun_host
}  // namespace ksana_llm
