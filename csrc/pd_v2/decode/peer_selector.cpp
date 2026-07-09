/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/decode/peer_selector.h"

#include <algorithm>
#include <vector>

namespace ksana_llm {
namespace pd_v2 {

PeerSelector::PeerSelector(SelectorPolicy policy) : policy_(policy) {}

void PeerSelector::RegisterPeer(uint64_t segment_id, std::string server_name) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peers_.find(segment_id);
  if (it == peers_.end()) {
    PeerHealth h;
    h.segment_id = segment_id;
    h.server_name = std::move(server_name);
    h.healthy = false;  // not selectable until first snapshot arrives
    peers_.emplace(segment_id, std::move(h));
  } else {
    it->second.server_name = std::move(server_name);
  }
}

void PeerSelector::UnregisterPeer(uint64_t segment_id) {
  std::lock_guard<std::mutex> lk(mu_);
  peers_.erase(segment_id);
}

void PeerSelector::UpdateSnapshot(uint64_t segment_id, const PeerHealth& health) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peers_.find(segment_id);
  if (it == peers_.end()) {
    // Fallback: maybe the local seg_id rotated under the same server_name
    // (S3 path: epoch change → closeSegment + openSegment → new seg_id).
    // Look up by server_name to update in place under the new key.
    if (!health.server_name.empty()) {
      for (auto sit = peers_.begin(); sit != peers_.end(); ++sit) {
        if (sit->second.server_name == health.server_name) {
          peers_.erase(sit);
          break;
        }
      }
    }
    PeerHealth h = health;
    h.segment_id = segment_id;
    peers_.emplace(segment_id, std::move(h));
    return;
  }
  // server_name is informational and may not be set on the snapshot
  // (e.g. periodic pushes are keyed by segment_id only); preserve the
  // existing name if the snapshot leaves it empty.
  std::string preserved_name = it->second.server_name;
  it->second = health;
  it->second.segment_id = segment_id;  // wire-supplied segment_id wins on collision
  if (it->second.server_name.empty()) {
    it->second.server_name = std::move(preserved_name);
  }
}

std::optional<uint64_t> PeerSelector::SegmentIdByName(const std::string& server_name) const {
  std::lock_guard<std::mutex> lk(mu_);
  for (const auto& kv : peers_) {
    if (kv.second.server_name == server_name) return kv.first;
  }
  return std::nullopt;
}

std::optional<PeerHealth> PeerSelector::GetSnapshot(uint64_t segment_id) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peers_.find(segment_id);
  if (it == peers_.end()) return std::nullopt;
  return it->second;
}

size_t PeerSelector::HealthyPeerCount() const {
  std::lock_guard<std::mutex> lk(mu_);
  size_t n = 0;
  for (const auto& kv : peers_) {
    if (kv.second.healthy) ++n;
  }
  return n;
}

size_t PeerSelector::RegisteredPeerCount() const {
  std::lock_guard<std::mutex> lk(mu_);
  return peers_.size();
}

std::vector<uint64_t> PeerSelector::HealthyPeers() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<uint64_t> out;
  out.reserve(peers_.size());
  for (const auto& kv : peers_) {
    if (kv.second.healthy) out.push_back(kv.first);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::optional<uint64_t> PeerSelector::Select() {
  std::lock_guard<std::mutex> lk(mu_);
  if (peers_.empty()) return std::nullopt;

  // Build the candidate list (healthy peers only) under the lock.
  std::vector<const PeerHealth*> healthy;
  healthy.reserve(peers_.size());
  for (const auto& kv : peers_) {
    if (kv.second.healthy) healthy.push_back(&kv.second);
  }
  if (healthy.empty()) return std::nullopt;

  switch (policy_) {
    case SelectorPolicy::kLinUCB:
    case SelectorPolicy::kRoundRobin:
      break;
  }

  // Stable ordering by segment_id ascending; cursor wraps around.
  std::sort(healthy.begin(), healthy.end(),
            [](const PeerHealth* a, const PeerHealth* b) { return a->segment_id < b->segment_id; });
  uint64_t idx = rr_cursor_.fetch_add(1, std::memory_order_relaxed) % healthy.size();
  return healthy[idx]->segment_id;
}

}  // namespace pd_v2
}  // namespace ksana_llm
