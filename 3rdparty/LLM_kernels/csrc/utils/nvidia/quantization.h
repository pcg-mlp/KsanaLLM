/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <stdlib.h>
#include <cstdint>

namespace llm_kernels {
namespace utils {

class QuantMode {
  // [WARNING] KEEP BELOW DEFINITION IN SYNC WITH tensorrt_llm/quantization/mode.py
 public:
  using BaseType = std::uint32_t;

  explicit constexpr QuantMode(BaseType value) noexcept : value_{value} {}

  QuantMode() noexcept = default;

  constexpr QuantMode(QuantMode const&) noexcept = default;

  constexpr QuantMode& operator=(const QuantMode& other) noexcept = default;

  static constexpr QuantMode None() noexcept { return QuantMode(BaseType(0)); }

  static constexpr QuantMode Int4Weights() noexcept { return QuantMode(BaseType(1u) << 0); }

  static constexpr QuantMode Int8Weights() noexcept { return QuantMode(BaseType(1u) << 1); }

  static constexpr QuantMode Activations() noexcept { return QuantMode(BaseType(1u) << 2); }

  static constexpr QuantMode PerChannelScaling() noexcept { return QuantMode(BaseType(1u) << 3); }

  static constexpr QuantMode PerTokenScaling() noexcept { return QuantMode(BaseType(1u) << 4); }

  static constexpr QuantMode PerGroupScaling() noexcept { return QuantMode(BaseType(1u) << 5); }

  static constexpr QuantMode Int8KvCache() noexcept { return QuantMode(BaseType(1u) << 6); }

  static constexpr QuantMode Fp8KvCache() noexcept { return QuantMode(BaseType(1u) << 7); }

  static constexpr QuantMode Fp8Qdq() noexcept { return QuantMode(BaseType(1u) << 8); }

  constexpr BaseType Value() const noexcept { return value_; }

  constexpr bool IsSet(QuantMode const& mode) const noexcept { return (value_ & mode.Value()) == mode.Value(); }

  constexpr bool HasInt4Weights() const noexcept { return IsSet(Int4Weights()); }

  constexpr bool HasInt8Weights() const noexcept { return IsSet(Int8Weights()); }

  constexpr bool HasActivations() const noexcept { return IsSet(Activations()); }

  constexpr bool HasPerChannelScaling() const noexcept { return IsSet(PerChannelScaling()); }

  constexpr bool HasPerTokenScaling() const noexcept { return IsSet(PerTokenScaling()); }

  constexpr bool HasPerGroupScaling() const noexcept { return IsSet(PerGroupScaling()); }

  constexpr bool HasStaticActivationScaling() const noexcept { return !HasPerTokenScaling(); }

  constexpr bool HasInt8KvCache() const noexcept { return IsSet(Int8KvCache()); }

  constexpr bool HasFp8KvCache() const noexcept { return IsSet(Fp8KvCache()); }

  constexpr bool HasFp8Qdq() const noexcept { return IsSet(Fp8Qdq()); }

  constexpr bool HasKvCacheQuant() const noexcept { return HasInt8KvCache() || HasFp8KvCache(); }

  static constexpr QuantMode FromDescription(bool quantize_weights = false, bool quantize_activations = false,
                                             bool per_token = false, bool per_channel = false,
                                             bool use_int4_weights = false, bool use_int8_kvcache = false,
                                             bool use_fp8_kvcache = false, bool use_fp8_qdq = false) {
    QuantMode quant_mode{};
    if (quantize_weights) {
      if (use_int4_weights)
        quant_mode += Int4Weights();
      else
        quant_mode += Int8Weights();
    }

    if (quantize_activations) {
      quant_mode += Activations();
    }

    if (per_channel) {
      quant_mode += QuantMode::PerChannelScaling();
    }
    if (per_token) {
      quant_mode += QuantMode::PerTokenScaling();
    }

    if (use_int8_kvcache) {
      quant_mode += Int8KvCache();
    }

    if (use_fp8_kvcache) {
      quant_mode += Fp8KvCache();
    }

    if (use_fp8_qdq) {
      quant_mode += Fp8Qdq();
    }

    return quant_mode;
  }

  constexpr QuantMode operator+(const QuantMode& other) const noexcept { return QuantMode(value_ | other.value_); }

  constexpr QuantMode& operator+=(const QuantMode& other) noexcept { return *this = *this + other; }

  constexpr QuantMode operator-(const QuantMode& other) const noexcept { return QuantMode(value_ & ~other.value_); }

  constexpr QuantMode& operator-=(const QuantMode& other) noexcept { return *this = *this - other; }

  constexpr bool operator==(const QuantMode& other) const noexcept { return value_ == other.value_; }

  constexpr bool operator!=(const QuantMode& other) const noexcept { return !(*this == other); }

 private:
  BaseType value_{0};
};

}  // namespace utils
}  // namespace llm_kernels
