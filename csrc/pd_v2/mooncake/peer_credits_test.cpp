/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/mooncake/peer_credits.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

namespace {

constexpr uint32_t kBase = 1024;  // arbitrary partition base offset
constexpr uint32_t kSize = 32;    // partition span

TEST(PeerCreditsTest, InitFailsOnZeroSize) {
  PeerCredits c;
  EXPECT_FALSE(c.Init(0, 0).OK());
}

TEST(PeerCreditsTest, AcquireAndReleaseRoundTrip) {
  PeerCredits c;
  ASSERT_TRUE(c.Init(kBase, kSize).OK());
  EXPECT_EQ(c.free_slots(), kSize);

  uint32_t s0 = 0;
  ASSERT_TRUE(c.AcquireSlots(4, /*timeout_ms=*/0, &s0).OK());
  EXPECT_GE(s0, kBase);
  EXPECT_LE(s0 + 4u, kBase + kSize);
  EXPECT_EQ(c.free_slots(), kSize - 4);

  ASSERT_TRUE(c.ReleaseSlots(s0, 4).OK());
  EXPECT_EQ(c.free_slots(), kSize);
}

TEST(PeerCreditsTest, AcquireRejectsOverPartitionRequest) {
  PeerCredits c;
  ASSERT_TRUE(c.Init(kBase, kSize).OK());
  uint32_t s = 0;
  EXPECT_FALSE(c.AcquireSlots(kSize + 1, /*timeout_ms=*/0, &s).OK());
  EXPECT_FALSE(c.AcquireSlots(0, /*timeout_ms=*/0, &s).OK());
}

TEST(PeerCreditsTest, AllocationsAreNonOverlapping) {
  PeerCredits c;
  ASSERT_TRUE(c.Init(kBase, kSize).OK());
  std::vector<std::pair<uint32_t, uint32_t>> ranges;
  uint32_t total = 0;
  while (total < kSize) {
    uint32_t s = 0;
    ASSERT_TRUE(c.AcquireSlots(4, /*timeout_ms=*/0, &s).OK());
    ranges.emplace_back(s, 4);
    total += 4;
  }
  // Next acquire of 1 must fail (full).
  uint32_t s = 0;
  EXPECT_FALSE(c.AcquireSlots(1, /*timeout_ms=*/0, &s).OK());
  // Verify ranges don't overlap.
  std::vector<bool> claimed(kSize, false);
  for (auto& r : ranges) {
    for (uint32_t i = 0; i < r.second; ++i) {
      uint32_t rel = r.first - kBase + i;
      ASSERT_LT(rel, kSize);
      EXPECT_FALSE(claimed[rel]) << "slot " << rel << " double-allocated";
      claimed[rel] = true;
    }
  }
  // Free everything; pool should be empty again.
  for (auto& r : ranges) ASSERT_TRUE(c.ReleaseSlots(r.first, r.second).OK());
  EXPECT_EQ(c.free_slots(), kSize);
}

TEST(PeerCreditsTest, OutOfOrderReleasePermitsReallocation) {
  PeerCredits c;
  ASSERT_TRUE(c.Init(kBase, kSize).OK());
  uint32_t a = 0, b = 0, x = 0;
  ASSERT_TRUE(c.AcquireSlots(8, /*timeout_ms=*/0, &a).OK());
  ASSERT_TRUE(c.AcquireSlots(8, /*timeout_ms=*/0, &b).OK());
  // Release the first range out-of-order.
  ASSERT_TRUE(c.ReleaseSlots(a, 8).OK());
  // Should be able to fit an 8-slot request again (in the released gap).
  ASSERT_TRUE(c.AcquireSlots(8, /*timeout_ms=*/0, &x).OK());
  EXPECT_EQ(x, a) << "next-fit should refill the released range";
  ASSERT_TRUE(c.ReleaseSlots(b, 8).OK());
  ASSERT_TRUE(c.ReleaseSlots(x, 8).OK());
  EXPECT_EQ(c.free_slots(), kSize);
}

TEST(PeerCreditsTest, AcquireBlocksUntilRelease) {
  PeerCredits c;
  ASSERT_TRUE(c.Init(kBase, kSize).OK());
  // Fill the partition so the next acquire must block.
  uint32_t held = 0;
  ASSERT_TRUE(c.AcquireSlots(kSize, /*timeout_ms=*/0, &held).OK());

  std::atomic<bool> got_it{false};
  uint32_t got_slot = 0;
  std::thread t([&] {
    ASSERT_TRUE(c.AcquireSlots(4, /*timeout_ms=*/-1, &got_slot).OK());
    got_it = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(got_it.load());
  ASSERT_TRUE(c.ReleaseSlots(held, kSize).OK());
  t.join();
  EXPECT_TRUE(got_it.load());
  EXPECT_GE(got_slot, kBase);
}

TEST(PeerCreditsTest, AcquireTimesOut) {
  PeerCredits c;
  ASSERT_TRUE(c.Init(kBase, kSize).OK());
  uint32_t held = 0;
  ASSERT_TRUE(c.AcquireSlots(kSize, /*timeout_ms=*/0, &held).OK());
  uint32_t s = 0;
  auto t0 = std::chrono::steady_clock::now();
  Status st = c.AcquireSlots(4, /*timeout_ms=*/30, &s);
  auto t1 = std::chrono::steady_clock::now();
  EXPECT_FALSE(st.OK());
  EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(), 25);
  ASSERT_TRUE(c.ReleaseSlots(held, kSize).OK());
}

TEST(PeerCreditsTest, ShutdownUnblocksWaiters) {
  PeerCredits c;
  ASSERT_TRUE(c.Init(kBase, kSize).OK());
  uint32_t held = 0;
  ASSERT_TRUE(c.AcquireSlots(kSize, /*timeout_ms=*/0, &held).OK());

  std::atomic<bool> done{false};
  std::thread t([&] {
    uint32_t s = 0;
    Status st = c.AcquireSlots(4, /*timeout_ms=*/-1, &s);
    EXPECT_FALSE(st.OK());
    done = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  c.Shutdown();
  t.join();
  EXPECT_TRUE(done.load());
}

TEST(PeerCreditsTest, ReleaseRejectsOutOfRange) {
  PeerCredits c;
  ASSERT_TRUE(c.Init(kBase, kSize).OK());
  EXPECT_FALSE(c.ReleaseSlots(kBase - 1, 1).OK());
  EXPECT_FALSE(c.ReleaseSlots(kBase + kSize, 1).OK());
  EXPECT_FALSE(c.ReleaseSlots(kBase, kSize + 1).OK());
  // Zero-length is a no-op, not an error.
  EXPECT_TRUE(c.ReleaseSlots(kBase, 0).OK());
}

TEST(PeerCreditsTest, ReinitAfterShutdown) {
  PeerCredits c;
  ASSERT_TRUE(c.Init(kBase, kSize).OK());
  c.Shutdown();
  uint32_t s = 0;
  EXPECT_FALSE(c.AcquireSlots(1, /*timeout_ms=*/0, &s).OK());
  ASSERT_TRUE(c.Init(kBase, kSize).OK());
  EXPECT_TRUE(c.AcquireSlots(1, /*timeout_ms=*/0, &s).OK());
}

}  // namespace
}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
