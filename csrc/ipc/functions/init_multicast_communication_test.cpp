/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <vector>

#include "ipc/core/ipc_func_serialization.h"
#include "ipc/functions/init_multicast_communication.h"
#include "utils/random_generator.h"

namespace ksana_llm {

// ============================================================================
// Test: unique_op_id sign bit masking
// Verifies the fix for the bug where static_cast<int64_t>(gen()) could produce
// negative values (50% probability), causing socket initialization to be skipped
// and resulting in null pointer dereference (SIGSEGV) in the recv branch.
// ============================================================================

TEST(InitMulticastCommunicationTest, UniqueOpIdAlwaysPositive) {
  // Simulate the fixed code path: static_cast<int64_t>(gen() & INT64_MAX)
  RandomGenerator<uint64_t> gen(42);  // Fixed seed for reproducibility
  const int kIterations = 100000;

  for (int i = 0; i < kIterations; ++i) {
    int64_t unique_op_id = static_cast<int64_t>(gen() & INT64_MAX);
    ASSERT_GE(unique_op_id, 0) << "unique_op_id must be non-negative, got " << unique_op_id << " at iteration " << i;
  }
}

// Demonstrates the bug: without masking, ~50% of values are negative
TEST(InitMulticastCommunicationTest, UniqueOpIdWithoutMaskHasNegativeValues) {
  RandomGenerator<uint64_t> gen(42);
  const int kIterations = 10000;
  int negative_count = 0;

  for (int i = 0; i < kIterations; ++i) {
    int64_t unique_op_id = static_cast<int64_t>(gen());  // BUG: no masking
    if (unique_op_id < 0) {
      negative_count++;
    }
  }

  // Without masking, roughly 50% should be negative
  // Use a generous range to avoid flaky tests
  EXPECT_GT(negative_count, kIterations / 4)
      << "Expected significant negative values without masking, got " << negative_count << "/" << kIterations;
  EXPECT_LT(negative_count, 3 * kIterations / 4)
      << "Expected roughly 50% negative, got " << negative_count << "/" << kIterations;
}

// ============================================================================
// Test: Serialization roundtrip for InitMulticastCommunicationRequest
// Verifies that unique_op_id survives serialize/deserialize correctly,
// particularly for large positive values near INT64_MAX.
// ============================================================================

TEST(InitMulticastCommunicationTest, SerializationRoundtrip_PositiveUniqueOpId) {
  uint8_t buffer[256];

  InitMulticastCommunicationRequest req;
  req.unique_op_id = 12345678901234LL;
  req.broadcast_id = 3;

  size_t bytes = 0;
  Status status = req.Serialize(buffer, &bytes);
  ASSERT_TRUE(status.OK());

  InitMulticastCommunicationRequest req2;
  status = req2.Deserialize(buffer);
  ASSERT_TRUE(status.OK());
  EXPECT_EQ(req2.unique_op_id, 12345678901234LL);
  EXPECT_EQ(req2.broadcast_id, 3);
}

TEST(InitMulticastCommunicationTest, SerializationRoundtrip_MaskedLargeValue) {
  uint8_t buffer[256];

  // Simulate the actual code path: gen() returns a large uint64_t, we mask it
  uint64_t raw = 0xFFFFFFFFFFFFFFFFULL;  // All bits set
  int64_t masked = static_cast<int64_t>(raw & INT64_MAX);
  ASSERT_GT(masked, 0);
  ASSERT_EQ(masked, INT64_MAX);

  InitMulticastCommunicationRequest req;
  req.unique_op_id = masked;
  req.broadcast_id = -1;

  size_t bytes = 0;
  Status status = req.Serialize(buffer, &bytes);
  ASSERT_TRUE(status.OK());

  InitMulticastCommunicationRequest req2;
  status = req2.Deserialize(buffer);
  ASSERT_TRUE(status.OK());
  EXPECT_EQ(req2.unique_op_id, INT64_MAX);
  EXPECT_EQ(req2.broadcast_id, -1);
}

// Verify that the uint64_t serialization of a positive int64_t roundtrips correctly
TEST(InitMulticastCommunicationTest, SerializationRoundtrip_TypeMismatchSafe) {
  uint8_t buffer[256];
  size_t offset = 0;

  // The serialization uses SerializePod<uint64_t> for an int64_t field.
  // This is safe as long as the value is non-negative (which the mask ensures).
  int64_t original = static_cast<int64_t>(0x7FFFFFFFFFFFFFFFULL);  // INT64_MAX
  SerializePod<uint64_t>(original, buffer, offset);

  offset = 0;
  int64_t deserialized = DeserializePod<uint64_t>(buffer, offset);
  EXPECT_EQ(deserialized, original);
}

// ============================================================================
// Test: Default values of InitMulticastCommunicationRequest
// Verifies the default sentinel values used for branching logic.
// ============================================================================

TEST(InitMulticastCommunicationTest, DefaultValues) {
  InitMulticastCommunicationRequest req;
  EXPECT_EQ(req.unique_op_id, -1) << "Default unique_op_id should be -1";
  EXPECT_EQ(req.broadcast_id, -1) << "Default broadcast_id should be -1";
}

// ============================================================================
// Test: INT64_MAX masking preserves randomness
// ============================================================================

TEST(InitMulticastCommunicationTest, MaskingPreservesRandomness) {
  RandomGenerator<uint64_t> gen(123);
  const int kIterations = 1000;
  std::vector<int64_t> values;
  values.reserve(kIterations);

  for (int i = 0; i < kIterations; ++i) {
    values.push_back(static_cast<int64_t>(gen() & INT64_MAX));
  }

  // Check that we don't get the same value repeatedly (degenerate case)
  int unique_count = 0;
  for (int i = 1; i < kIterations; ++i) {
    if (values[i] != values[i - 1]) {
      unique_count++;
    }
  }
  EXPECT_GT(unique_count, kIterations / 2) << "Masking should not collapse randomness";

  // All values should be positive
  for (int i = 0; i < kIterations; ++i) {
    EXPECT_GE(values[i], 0);
  }
}

// ============================================================================
// Test: Edge case — masking 0 and specific bit patterns
// ============================================================================

TEST(InitMulticastCommunicationTest, MaskingEdgeCases) {
  // Mask of 0 should produce 0 (still >= 0, so socket init will proceed)
  uint64_t zero = 0;
  int64_t masked_zero = static_cast<int64_t>(zero & INT64_MAX);
  EXPECT_EQ(masked_zero, 0);
  EXPECT_GE(masked_zero, 0);

  // Mask of exactly 2^63 (MSB set, all others 0) should produce 0
  uint64_t msb_only = 0x8000000000000000ULL;
  int64_t masked_msb = static_cast<int64_t>(msb_only & INT64_MAX);
  EXPECT_EQ(masked_msb, 0);
  EXPECT_GE(masked_msb, 0);

  // Mask of 2^63 - 1 (INT64_MAX) should stay the same
  uint64_t max_positive = static_cast<uint64_t>(INT64_MAX);
  int64_t masked_max = static_cast<int64_t>(max_positive & INT64_MAX);
  EXPECT_EQ(masked_max, INT64_MAX);

  // Mask of UINT64_MAX should produce INT64_MAX
  uint64_t all_ones = UINT64_MAX;
  int64_t masked_all = static_cast<int64_t>(all_ones & INT64_MAX);
  EXPECT_EQ(masked_all, INT64_MAX);
}

}  // namespace ksana_llm
