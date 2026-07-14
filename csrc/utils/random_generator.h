/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <random>
#include <type_traits>
#include <limits>

namespace ksana_llm {

// Random generator for arithmetic types
template <typename T>
class RandomGenerator {
  static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");

 public:
  // Default seed from non-deterministic source
  RandomGenerator() : rng_(std::random_device{}()) {}

  // User-supplied seed for reproducibility
  explicit RandomGenerator(uint64_t seed) : rng_(seed) {}

  // Uniform integer in [min_val, max_val]
  template <typename U = T>
  std::enable_if_t<std::is_integral_v<U>, U> uniform(U min_val = std::numeric_limits<U>::min(),
                                                     U max_val = std::numeric_limits<U>::max()) {
    std::uniform_int_distribution<U> dist(min_val, max_val);
    return dist(rng_);
  }

  // Uniform floating-point in [min_val, max_val]
  template <typename U = T>
  std::enable_if_t<std::is_floating_point_v<U>, U> uniform(U min_val = U(0), U max_val = U(1)) {
    std::uniform_real_distribution<U> dist(min_val, max_val);
    return dist(rng_);
  }

  // Auto dispatch
  T operator()(T min_val = std::is_integral_v<T> ? std::numeric_limits<T>::min() : T(0),
               T max_val = std::is_integral_v<T> ? std::numeric_limits<T>::max() : T(1)) {
    return uniform(min_val, max_val);
  }

 private:
  std::mt19937_64 rng_;
};

}  // namespace ksana_llm
