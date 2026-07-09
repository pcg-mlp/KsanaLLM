/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/mooncake/host_stage_pool.h"

#include <gtest/gtest.h>

#include <cstring>
#include <set>
#include <string>

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

namespace {

constexpr size_t kSlot = HostStagePool::kSlotSize;

// 16 MiB / 8 peers / 64 KB → 256 slots, 32 per peer. Small enough to
// always succeed under default ulimit -l, big enough to exercise the
// partition arithmetic.
constexpr size_t kSmallPool = 16ull * 1024 * 1024;
constexpr size_t kSmallPeers = 8;

TEST(HostStagePoolTest, InitDerivedSizes) {
  HostStagePool p;
  ASSERT_TRUE(p.Init(kSmallPool, kSmallPeers).OK());
  EXPECT_EQ(p.total_bytes(), kSmallPool);
  EXPECT_EQ(p.total_slots(), kSmallPool / kSlot);
  EXPECT_EQ(p.max_peers(), kSmallPeers);
  EXPECT_EQ(p.slots_per_peer(), p.total_slots() / kSmallPeers);
  EXPECT_NE(p.base(), nullptr);
}

TEST(HostStagePoolTest, RegisterPeerStaticPartition) {
  HostStagePool p;
  ASSERT_TRUE(p.Init(kSmallPool, kSmallPeers).OK());
  HostStagePool::PeerPartition a{}, b{}, c{};
  ASSERT_TRUE(p.RegisterPeer("decode-A", &a).OK());
  ASSERT_TRUE(p.RegisterPeer("decode-B", &b).OK());
  ASSERT_TRUE(p.RegisterPeer("decode-C", &c).OK());
  EXPECT_EQ(a.base_slot, 0u);
  EXPECT_EQ(b.base_slot, p.slots_per_peer());
  EXPECT_EQ(c.base_slot, 2u * p.slots_per_peer());
  EXPECT_EQ(a.n_slots, p.slots_per_peer());
  EXPECT_EQ(b.n_slots, p.slots_per_peer());
  EXPECT_EQ(c.n_slots, p.slots_per_peer());
}

TEST(HostStagePoolTest, RegisterPeerIdempotent) {
  HostStagePool p;
  ASSERT_TRUE(p.Init(kSmallPool, kSmallPeers).OK());
  HostStagePool::PeerPartition first{}, again{};
  ASSERT_TRUE(p.RegisterPeer("decode-A", &first).OK());
  ASSERT_TRUE(p.RegisterPeer("decode-A", &again).OK());
  EXPECT_EQ(first.base_slot, again.base_slot);
  EXPECT_EQ(first.n_slots, again.n_slots);
}

TEST(HostStagePoolTest, RegisterPeerExhaustsMaxPeers) {
  HostStagePool p;
  ASSERT_TRUE(p.Init(kSmallPool, /*max_peers=*/2).OK());
  HostStagePool::PeerPartition x{};
  ASSERT_TRUE(p.RegisterPeer("decode-A", &x).OK());
  ASSERT_TRUE(p.RegisterPeer("decode-B", &x).OK());
  // Third *new* peer fails — partitions exhausted.
  EXPECT_FALSE(p.RegisterPeer("decode-C", &x).OK());
  // Re-register an existing peer still works (idempotent path is
  // independent of the new-partition counter).
  ASSERT_TRUE(p.RegisterPeer("decode-A", &x).OK());
}

TEST(HostStagePoolTest, PartitionsAreNonOverlapping) {
  HostStagePool p;
  ASSERT_TRUE(p.Init(kSmallPool, kSmallPeers).OK());
  std::set<size_t> claimed;
  for (size_t i = 0; i < kSmallPeers; ++i) {
    HostStagePool::PeerPartition part{};
    ASSERT_TRUE(p.RegisterPeer("decode-" + std::to_string(i), &part).OK());
    for (size_t s = part.base_slot; s < part.base_slot + part.n_slots; ++s) {
      auto inserted = claimed.insert(s);
      EXPECT_TRUE(inserted.second) << "slot " << s << " double-claimed";
    }
  }
  EXPECT_EQ(claimed.size(), kSmallPeers * p.slots_per_peer());
}

TEST(HostStagePoolTest, SlotPtrAddressing) {
  HostStagePool p;
  ASSERT_TRUE(p.Init(kSmallPool, kSmallPeers).OK());
  HostStagePool::PeerPartition part{};
  ASSERT_TRUE(p.RegisterPeer("decode-A", &part).OK());
  char* slot0 = p.SlotPtr(part.base_slot);
  char* slot1 = p.SlotPtr(part.base_slot + 1);
  EXPECT_EQ(static_cast<size_t>(slot1 - slot0), kSlot);
  // Sanity: write+read across slot boundary, mimicking a multi-slot WRITE.
  std::memset(slot0, 0xAB, kSlot * 2);
  EXPECT_EQ(static_cast<unsigned char>(slot0[0]), 0xABu);
  EXPECT_EQ(static_cast<unsigned char>(slot1[kSlot - 1]), 0xABu);
}

TEST(HostStagePoolTest, RejectsBadInit) {
  {
    HostStagePool p;
    EXPECT_FALSE(p.Init(0, kSmallPeers).OK());
  }
  {
    HostStagePool p;
    EXPECT_FALSE(p.Init(kSlot + 1, kSmallPeers).OK());
  }
  {
    HostStagePool p;
    EXPECT_FALSE(p.Init(kSmallPool, 0).OK());
  }
  {
    // total_slots < max_peers
    HostStagePool p;
    EXPECT_FALSE(p.Init(kSlot, /*max_peers=*/4).OK());
  }
}

TEST(HostStagePoolTest, RejectsDoubleInit) {
  HostStagePool p;
  ASSERT_TRUE(p.Init(kSmallPool, kSmallPeers).OK());
  EXPECT_FALSE(p.Init(kSmallPool, kSmallPeers).OK());
}

TEST(HostStagePoolTest, PeerPartitionForUnknownPeer) {
  HostStagePool p;
  ASSERT_TRUE(p.Init(kSmallPool, kSmallPeers).OK());
  HostStagePool::PeerPartition part{};
  EXPECT_FALSE(p.PeerPartitionFor("never-registered", &part).OK());
  ASSERT_TRUE(p.RegisterPeer("decode-A", &part).OK());
  HostStagePool::PeerPartition lookup{};
  ASSERT_TRUE(p.PeerPartitionFor("decode-A", &lookup).OK());
  EXPECT_EQ(lookup.base_slot, part.base_slot);
  EXPECT_EQ(lookup.n_slots, part.n_slots);
}

TEST(HostStagePoolTest, ShutdownReleases) {
  HostStagePool p;
  ASSERT_TRUE(p.Init(kSmallPool, kSmallPeers).OK());
  ASSERT_TRUE(p.Shutdown().OK());
  // Re-init after shutdown works.
  ASSERT_TRUE(p.Init(kSmallPool, kSmallPeers).OK());
}

}  // namespace
}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
