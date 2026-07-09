/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pd_v2/public/pd_v2_config.h"
#include "pd_v2/public/pd_v2_types.h"
#include "utils/status.h"

namespace ksana_llm {
namespace pd_v2 {

// Decode-engine-local peer routing table. Snapshot-driven: PeerHealth is
// pushed in via PrefillRequestAck / PrefillRequestComplete / periodic
// PrefillSnapshot, and Select() returns a peer_id (== segment_id) chosen
// purely from those snapshots.
//
// Design (§4.3):
//   - PeerSelector is *stateless about per-request bookkeeping*. Select()
//     does not record (req_id, peer_id) mappings, and there is no
//     reservation counter to keep in sync. The connector engine layer that
//     owns CanAccept is responsible for any req_id->peer_id memory.
//   - Snapshots are the single source of truth. Whoever calls Select()
//     accepts that the chosen peer's view may be slightly stale; recovery
//     is by retry against the next snapshot.
//   - Policies:
//       kRoundRobin - cycle through the healthy peers in segment_id order.
//       kLinUCB     - use the same round-robin remote-peer selection; LinUCB
//                     only decides whether to stay local before remote routing.
//
// Thread-safety: every method takes the internal mutex. Select() is
// expected to be called from the engine event loop; UpdateSnapshot from
// the connector's network thread.
class PeerSelector {
 public:
  explicit PeerSelector(SelectorPolicy policy = SelectorPolicy::kRoundRobin);

  // Picks a peer based on the current snapshot table. Returns nullopt if
  // no peer is currently healthy — the caller should surface this as
  // backpressure to the engine hook.
  std::optional<uint64_t> Select();

  // Add a peer to the routing table. Idempotent on repeat call.
  // server_name is purely informational (e.g. routed to logs); peers are
  // keyed by segment_id end-to-end. If `healthy` defaults to false; a
  // freshly registered peer is not selectable until the first snapshot
  // arrives.
  void RegisterPeer(uint64_t segment_id, std::string server_name);

  // Drop a peer from the routing table. In-flight req_ids whose
  // connector-side bookkeeping points at this segment_id should be
  // failed by the connector layer (this class doesn't see req_ids).
  void UnregisterPeer(uint64_t segment_id);

  // Refresh a peer's health snapshot. Called on Ack / Complete / periodic
  // Snapshot receipt. If `segment_id` is unknown but `health.server_name`
  // matches an already-registered peer's name, the existing entry is
  // updated (server_name stays the same key, segment_id may have rotated
  // — happens after explicit closeSegment + openSegment from epoch
  // change). If neither segment_id nor server_name is found, the call is
  // a no-op — caller should RegisterPeer first (Snapshot-driven discovery
  // path: do openSegment + RegisterPeer **before** UpdateSnapshot).
  void UpdateSnapshot(uint64_t segment_id, const PeerHealth& health);

  // Lookup helpers used by snapshot-driven discovery: when a Snapshot
  // arrives from a Prefill we haven't seen, the connector needs to
  // (1) check whether the server_name is already known, (2) if not,
  // openSegment + RegisterPeer to learn the local seg_id, then call
  // UpdateSnapshot. Returns nullopt when the name isn't registered.
  std::optional<uint64_t> SegmentIdByName(const std::string& server_name) const;

  // Inspect the current snapshot for a peer; returns nullopt if not in
  // the table. Used by tests and by the Decode connector's TCP send path
  // to short-circuit "peer marked unhealthy after Select()" races.
  std::optional<PeerHealth> GetSnapshot(uint64_t segment_id) const;

  size_t HealthyPeerCount() const;
  size_t RegisteredPeerCount() const;

  // Return every currently-healthy peer's segment_id, sorted ascending.
  // Used by the Decode connector's async submit fallback: when the
  // primary peer's stage pool is momentarily full, the worker iterates
  // these to try RDMA-staging to another Prefill before blocking. Order
  // is stable (segment_id ascending) so callers can round-robin over it.
  std::vector<uint64_t> HealthyPeers() const;

 private:
  SelectorPolicy policy_;

  mutable std::mutex mu_;
  std::unordered_map<uint64_t, PeerHealth> peers_;

  // Cursor used by kRoundRobin. Iterates over a stable ordering derived
  // from segment_id (sorted ascending) every Select() call.
  std::atomic<uint64_t> rr_cursor_{0};
};

}  // namespace pd_v2
}  // namespace ksana_llm
