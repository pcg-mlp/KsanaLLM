/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <gtest/gtest.h>

#include "random_generator.h"

namespace ksana_llm {

// ---------- 整数 ----------
TEST(RandomGeneratorInt, FullRange) {
  RandomGenerator<int> rng;
  int v = rng();  // 默认全范围
  EXPECT_GE(v, std::numeric_limits<int>::min());
  EXPECT_LE(v, std::numeric_limits<int>::max());
}

TEST(RandomGeneratorInt, NarrowRange) {
  RandomGenerator<int> rng(123);
  for (int i = 0; i < 1000; ++i) {
    int v = rng.uniform(10, 20);
    EXPECT_GE(v, 10);
    EXPECT_LE(v, 20);
  }
}

TEST(RandomGeneratorInt, ReproducibleWithSeed) {
  RandomGenerator<int> rng1(42);
  RandomGenerator<int> rng2(42);
  EXPECT_EQ(rng1.uniform(0, 1000), rng2.uniform(0, 1000));
}

// ---------- 浮点 ----------
TEST(RandomGeneratorFloat, DefaultZeroOne) {
  RandomGenerator<float> rng;
  for (int i = 0; i < 1000; ++i) {
    float v = rng();  // 默认 [0,1]
    EXPECT_GE(v, 0.0f);
    EXPECT_LE(v, 1.0f);
  }
}

TEST(RandomGeneratorFloat, NegativeRange) {
  RandomGenerator<double> rng;
  for (int i = 0; i < 1000; ++i) {
    double v = rng.uniform(-1.0, 1.0);
    EXPECT_GE(v, -1.0);
    EXPECT_LE(v, 1.0);
  }
}

// ---------- 边界值 ----------
TEST(RandomGeneratorInt, SingleValue) {
  RandomGenerator<int> rng;
  EXPECT_EQ(rng.uniform(7, 7), 7);
}

TEST(RandomGeneratorFloat, SingleValue) {
  RandomGenerator<float> rng;
  EXPECT_FLOAT_EQ(rng.uniform(3.14f, 3.14f), 3.14f);
}

// ---------- 64 位 ----------
TEST(RandomGeneratorInt64, FullRange) {
  RandomGenerator<int64_t> rng;
  int64_t v = rng();
  EXPECT_GE(v, std::numeric_limits<int64_t>::min());
  EXPECT_LE(v, std::numeric_limits<int64_t>::max());
}

}  // namespace ksana_llm
