/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <array>
#include <cstddef>

namespace ksana_llm {
namespace pd_v2 {

// Fixed-dimension linear-algebra primitives for the LinUCB router. The LinUCB
// arm dimension is pinned at 4 (bias + idle + tpot_headroom + remaining_small),
// so these avoid any matrix library: a 4x4 matrix is a row-major
// std::array<double,16>, a vector is std::array<double,4>. All functions are
// pure (no allocation, no state) so the router can keep its critical section
// trivially short.
inline constexpr std::size_t kLinUCBDim = 4;

using LinUCBVec = std::array<double, kLinUCBDim>;
using LinUCBMat = std::array<double, kLinUCBDim * kLinUCBDim>;  // row-major

// Identity matrix scaled by `lambda` (the regularized A0 = lambda*I, whose
// inverse is (1/lambda)*I).
inline LinUCBMat ScaledIdentity(double value) {
  LinUCBMat m{};  // zero-initialized
  for (std::size_t i = 0; i < kLinUCBDim; ++i) {
    m[i * kLinUCBDim + i] = value;
  }
  return m;
}

// Dot product of two 4-vectors.
inline double Dot(const LinUCBVec& a, const LinUCBVec& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < kLinUCBDim; ++i) s += a[i] * b[i];
  return s;
}

// Matrix-vector product: out = A * x.
inline LinUCBVec MatVec(const LinUCBMat& a, const LinUCBVec& x) {
  LinUCBVec out{};
  for (std::size_t i = 0; i < kLinUCBDim; ++i) {
    double s = 0.0;
    for (std::size_t j = 0; j < kLinUCBDim; ++j) {
      s += a[i * kLinUCBDim + j] * x[j];
    }
    out[i] = s;
  }
  return out;
}

// Quadratic form x^T A x. Used for the LinUCB exploration bonus
// sqrt(x^T Ainv x).
inline double QuadForm(const LinUCBMat& a, const LinUCBVec& x) { return Dot(x, MatVec(a, x)); }

// Sherman-Morrison rank-1 update of an inverse. Given Ainv = A^{-1}, return
// (A + x x^T)^{-1} = Ainv - (Ainv x)(Ainv x)^T / (1 + x^T Ainv x). This is the
// per-reward A-matrix update for a LinUCB arm, done directly on the stored
// inverse so no per-step matrix inversion is needed. `Ainv` is symmetric and
// stays symmetric under this update.
inline void ShermanMorrisonUpdate(LinUCBMat& ainv, const LinUCBVec& x) {
  const LinUCBVec u = MatVec(ainv, x);  // Ainv * x (== x^T * Ainv since symmetric)
  const double denom = 1.0 + Dot(x, u);
  // denom >= 1 for any PSD Ainv and real x, so this is numerically safe.
  for (std::size_t i = 0; i < kLinUCBDim; ++i) {
    for (std::size_t j = 0; j < kLinUCBDim; ++j) {
      ainv[i * kLinUCBDim + j] -= (u[i] * u[j]) / denom;
    }
  }
}

}  // namespace pd_v2
}  // namespace ksana_llm
