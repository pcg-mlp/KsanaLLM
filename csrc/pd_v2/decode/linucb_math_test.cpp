/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/decode/linucb_math.h"

#include <gtest/gtest.h>

#include <cmath>

namespace ksana_llm {
namespace pd_v2 {
namespace {

// Brute-force 4x4 inverse via Gauss-Jordan, used only as a test oracle.
LinUCBMat BruteInverse(const LinUCBMat& a) {
  constexpr std::size_t n = kLinUCBDim;
  double m[n][2 * n];
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) m[i][j] = a[i * n + j];
    for (std::size_t j = 0; j < n; ++j) m[i][n + j] = (i == j) ? 1.0 : 0.0;
  }
  for (std::size_t col = 0; col < n; ++col) {
    // Partial pivot.
    std::size_t pivot = col;
    for (std::size_t r = col + 1; r < n; ++r) {
      if (std::fabs(m[r][col]) > std::fabs(m[pivot][col])) pivot = r;
    }
    for (std::size_t j = 0; j < 2 * n; ++j) std::swap(m[col][j], m[pivot][j]);
    const double diag = m[col][col];
    for (std::size_t j = 0; j < 2 * n; ++j) m[col][j] /= diag;
    for (std::size_t r = 0; r < n; ++r) {
      if (r == col) continue;
      const double factor = m[r][col];
      for (std::size_t j = 0; j < 2 * n; ++j) m[r][j] -= factor * m[col][j];
    }
  }
  LinUCBMat inv{};
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) inv[i * n + j] = m[i][n + j];
  }
  return inv;
}

// Reconstruct A from its Sherman-Morrison-maintained inverse by accumulating
// the rank-1 updates onto lambda*I, then compare A^{-1} against the
// incrementally-maintained Ainv.
TEST(LinUCBMathTest, ShermanMorrisonMatchesBruteInverse) {
  const double lambda = 1.0;
  LinUCBMat ainv = ScaledIdentity(1.0 / lambda);
  LinUCBMat a = ScaledIdentity(lambda);  // ground-truth A, updated in lockstep

  const std::array<LinUCBVec, 5> xs = {{
      {1.0, 0.5, 0.2, 0.9},
      {1.0, 0.1, 0.8, 0.3},
      {1.0, 0.7, 0.7, 0.7},
      {1.0, 0.0, 1.0, 0.5},
      {1.0, 0.9, 0.1, 0.4},
  }};

  for (const auto& x : xs) {
    // A += x x^T
    for (std::size_t i = 0; i < kLinUCBDim; ++i) {
      for (std::size_t j = 0; j < kLinUCBDim; ++j) {
        a[i * kLinUCBDim + j] += x[i] * x[j];
      }
    }
    ShermanMorrisonUpdate(ainv, x);

    const LinUCBMat oracle = BruteInverse(a);
    for (std::size_t k = 0; k < kLinUCBDim * kLinUCBDim; ++k) {
      EXPECT_NEAR(ainv[k], oracle[k], 1e-9) << "mismatch at index " << k;
    }
  }
}

TEST(LinUCBMathTest, MatVecAndQuadForm) {
  LinUCBMat a = ScaledIdentity(2.0);
  LinUCBVec x = {1.0, 2.0, 3.0, 4.0};
  LinUCBVec av = MatVec(a, x);
  EXPECT_DOUBLE_EQ(av[0], 2.0);
  EXPECT_DOUBLE_EQ(av[3], 8.0);
  // x^T (2I) x = 2 * |x|^2 = 2*(1+4+9+16) = 60
  EXPECT_DOUBLE_EQ(QuadForm(a, x), 60.0);
}

TEST(LinUCBMathTest, QuadFormStaysNonNegative) {
  // Ainv starts PSD and Sherman-Morrison keeps it PSD, so the exploration
  // bonus argument x^T Ainv x must never go negative (would NaN the sqrt).
  LinUCBMat ainv = ScaledIdentity(1.0);
  const std::array<LinUCBVec, 3> xs = {{
      {1.0, 0.3, 0.6, 0.2},
      {1.0, 0.9, 0.1, 0.5},
      {1.0, 0.4, 0.4, 0.8},
  }};
  for (const auto& x : xs) {
    EXPECT_GE(QuadForm(ainv, x), 0.0);
    ShermanMorrisonUpdate(ainv, x);
  }
}

}  // namespace
}  // namespace pd_v2
}  // namespace ksana_llm
