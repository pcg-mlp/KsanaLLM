/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/decode/peer_selector.h"

#include <gtest/gtest.h>

#include <unordered_set>

namespace ksana_llm {
namespace pd_v2 {

namespace {

PeerHealth MakeHealth(uint64_t seg_id, std::string name, uint32_t free_kv, bool healthy = true) {
  PeerHealth h;
  h.segment_id = seg_id;
  h.server_name = std::move(name);
  h.free_kv_blocks = free_kv;
  h.last_snapshot_ms = 1000;
  h.healthy = healthy;
  return h;
}

}  // namespace

TEST(PeerSelectorTest, NoPeersRegisteredYieldsNullopt) {
  PeerSelector sel;
  EXPECT_FALSE(sel.Select().has_value());
}

TEST(PeerSelectorTest, RegisteredButNoSnapshotYieldsNullopt) {
  PeerSelector sel;
  sel.RegisterPeer(0x1234, "peer-a");
  // No UpdateSnapshot yet -> peer is registered as unhealthy by default.
  EXPECT_FALSE(sel.Select().has_value());
  EXPECT_EQ(sel.RegisteredPeerCount(), 1u);
  EXPECT_EQ(sel.HealthyPeerCount(), 0u);
}

TEST(PeerSelectorTest, UnhealthyPeersAreSkipped) {
  PeerSelector sel;
  sel.RegisterPeer(1, "a");
  sel.RegisterPeer(2, "b");
  sel.UpdateSnapshot(1, MakeHealth(1, "a", 1000, /*healthy=*/false));
  sel.UpdateSnapshot(2, MakeHealth(2, "b", 10));
  EXPECT_EQ(*sel.Select(), 2u);
  EXPECT_EQ(sel.HealthyPeerCount(), 1u);
}

TEST(PeerSelectorTest, DefaultRoundRobinCyclesStableOrder) {
  PeerSelector sel;
  sel.RegisterPeer(100, "a");
  sel.RegisterPeer(50, "b");
  sel.RegisterPeer(200, "c");
  sel.UpdateSnapshot(100, MakeHealth(100, "a", 1));
  sel.UpdateSnapshot(50, MakeHealth(50, "b", 1));
  sel.UpdateSnapshot(200, MakeHealth(200, "c", 1));

  // Sorted: 50, 100, 200. Cycle should be 50, 100, 200, 50, 100, 200, ...
  std::vector<uint64_t> seen;
  for (int i = 0; i < 6; ++i) seen.push_back(*sel.Select());
  EXPECT_EQ(seen, std::vector<uint64_t>({50, 100, 200, 50, 100, 200}));
}

TEST(PeerSelectorTest, LinUCBUsesRoundRobinForRemotePeers) {
  PeerSelector sel(SelectorPolicy::kLinUCB);
  sel.RegisterPeer(100, "a");
  sel.RegisterPeer(50, "b");
  sel.UpdateSnapshot(100, MakeHealth(100, "a", 1));
  sel.UpdateSnapshot(50, MakeHealth(50, "b", 1000));

  EXPECT_EQ(*sel.Select(), 50u);
  EXPECT_EQ(*sel.Select(), 100u);
}

TEST(PeerSelectorTest, UnregisterRemovesPeer) {
  PeerSelector sel;
  sel.RegisterPeer(1, "a");
  sel.RegisterPeer(2, "b");
  sel.UpdateSnapshot(1, MakeHealth(1, "a", 100));
  sel.UpdateSnapshot(2, MakeHealth(2, "b", 200));

  sel.UnregisterPeer(2);
  EXPECT_EQ(sel.RegisteredPeerCount(), 1u);
  EXPECT_EQ(*sel.Select(), 1u);
}

TEST(PeerSelectorTest, UpdateSnapshotForUnknownPeerUpserts) {
  // S1 behavior: UpdateSnapshot is now an upsert. A snapshot for a
  // never-before-seen peer (segment_id and server_name both unknown)
  // installs the peer in the table. This is the discovery path that
  // OnPrefillSnapshot relies on when a Decode receives the first
  // PrefillSnapshot from a never-contacted Prefill.
  PeerSelector sel;
  sel.UpdateSnapshot(999, MakeHealth(999, "ghost", 1000));
  EXPECT_EQ(sel.RegisteredPeerCount(), 1u);
  // healthy=true bit was set by MakeHealth, so it's selectable now.
  ASSERT_TRUE(sel.Select().has_value());
  EXPECT_EQ(*sel.Select(), 999u);
}

TEST(PeerSelectorTest, UpdateSnapshotEpochRotatesSegmentIdUnderSameName) {
  // Decode boots, sees Prefill via Snapshot at seg_id=42; Prefill restarts,
  // closes its old segment and reopens — same server_name but new local
  // SegmentID (53). UpdateSnapshot should replace the old key in place.
  PeerSelector sel;
  sel.RegisterPeer(42, "p1");
  sel.UpdateSnapshot(42, MakeHealth(42, "p1", 100));
  ASSERT_EQ(sel.RegisteredPeerCount(), 1u);
  sel.UpdateSnapshot(53, MakeHealth(53, "p1", 200));
  EXPECT_EQ(sel.RegisteredPeerCount(), 1u);  // old seg evicted, new inserted
  ASSERT_TRUE(sel.GetSnapshot(53).has_value());
  EXPECT_FALSE(sel.GetSnapshot(42).has_value());
  EXPECT_EQ(sel.GetSnapshot(53)->free_kv_blocks, 200u);
}

TEST(PeerSelectorTest, SegmentIdByNameReturnsCachedKey) {
  PeerSelector sel;
  sel.RegisterPeer(7, "n7");
  ASSERT_TRUE(sel.SegmentIdByName("n7").has_value());
  EXPECT_EQ(*sel.SegmentIdByName("n7"), 7u);
  EXPECT_FALSE(sel.SegmentIdByName("missing").has_value());
}

TEST(PeerSelectorTest, UpdateSnapshotPreservesServerNameOnEmpty) {
  PeerSelector sel;
  sel.RegisterPeer(7, "originally-named");
  // Snapshot leaves server_name empty (periodic push semantics).
  PeerHealth h = MakeHealth(7, "", 50);
  sel.UpdateSnapshot(7, h);

  auto out = sel.GetSnapshot(7);
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->server_name, "originally-named");
  EXPECT_EQ(out->free_kv_blocks, 50u);
}

TEST(PeerSelectorTest, HealthyPeersListsOnlyHealthySortedAscending) {
  PeerSelector sel;
  sel.RegisterPeer(300, "c");
  sel.RegisterPeer(100, "a");
  sel.RegisterPeer(200, "b");
  // 300 stays unhealthy (no snapshot); 100 + 200 become healthy.
  sel.UpdateSnapshot(200, MakeHealth(200, "b", 10));
  sel.UpdateSnapshot(100, MakeHealth(100, "a", 20));

  // Only the healthy peers, sorted by segment_id ascending — the order the
  // connector's submit-fallback iterates when rerouting off a full peer.
  EXPECT_EQ(sel.HealthyPeers(), std::vector<uint64_t>({100, 200}));

  // Marking 200 unhealthy drops it; 300 becoming healthy appends at the end.
  sel.UpdateSnapshot(200, MakeHealth(200, "b", 10, /*healthy=*/false));
  sel.UpdateSnapshot(300, MakeHealth(300, "c", 5));
  EXPECT_EQ(sel.HealthyPeers(), std::vector<uint64_t>({100, 300}));
}

TEST(PeerSelectorTest, HealthyPeersEmptyWhenNoneHealthy) {
  PeerSelector sel;
  EXPECT_TRUE(sel.HealthyPeers().empty());
  sel.RegisterPeer(1, "a");  // registered but no snapshot -> unhealthy
  EXPECT_TRUE(sel.HealthyPeers().empty());
}

TEST(PeerSelectorTest, GetSnapshotReturnsCurrentView) {
  PeerSelector sel;
  sel.RegisterPeer(11, "n11");
  sel.UpdateSnapshot(11, MakeHealth(11, "n11", 333));
  auto out = sel.GetSnapshot(11);
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->free_kv_blocks, 333u);
  EXPECT_FALSE(sel.GetSnapshot(0xDEAD).has_value());
}

}  // namespace pd_v2
}  // namespace ksana_llm
