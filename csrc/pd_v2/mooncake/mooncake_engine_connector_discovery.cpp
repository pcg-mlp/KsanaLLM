/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// MooncakeEngineConnector notify-plumbing and §3.3 service discovery: the
// notify-poller thread (NotifyPollerLoop / DispatchOne), the Prefill heartbeat
// broadcaster (HeartbeatLoop / BroadcastSnapshot) and its Decode-side snapshot
// receiver (OnPrefillSnapshot), plus the Decode Redis keepalive thread
// (RedisKeepaliveLoop). BroadcastSnapshot and OnPrefillSnapshot are the send /
// receive ends of the same per-peer PrefillSnapshot, so they live together.
#include "pd_v2/mooncake/mooncake_engine_connector.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <string>
#include <thread>

#include "fmt/core.h"
#include "pd_v2/mooncake/mooncake_engine_connector_internal.h"
#include "transfer_engine.h"
#include "utils/logger.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

namespace {

constexpr int kPollerIntervalMs = 5;

// Keepalive cadence (Decode role): TTL=5s in Redis, renewed every 2s so a
// missed renewal still leaves 3s of headroom before Prefill drops us.
constexpr int kAliveTtlSec = 5;
constexpr uint32_t kAliveRenewMs = 2000;

// Service-discovery key namespace, scoped by cluster_name so multiple
// independent pd_v2 clusters sharing one Redis don't discover each
// other's Decodes (KEYS pd_v2/decode_alive/<cluster>/* never crosses a
// cluster boundary). Empty cluster_name falls back to "default" to
// match python request_id.py and avoid an empty "//" segment. Both
// ends of a peer pair must agree on cluster_name (see pd_v2_config.h).
std::string AliveKeyPrefix(const std::string& cluster_name) {
  return "pd_v2/decode_alive/" + (cluster_name.empty() ? std::string("default") : cluster_name) + "/";
}

// Lightweight extractor for `{"epoch":<digits>}` — avoids pulling in a
// JSON parser for a single integer field. Returns 0 on parse failure
// (treated as "skip this entry" by callers).
uint64_t ParseEpochFromJson(const std::string& v) {
  size_t k = v.find("\"epoch\"");
  if (k == std::string::npos) return 0;
  k = v.find(':', k);
  if (k == std::string::npos) return 0;
  ++k;
  while (k < v.size() && (v[k] == ' ' || v[k] == '\t' || v[k] == '"')) ++k;
  uint64_t epoch = 0;
  bool any = false;
  while (k < v.size() && v[k] >= '0' && v[k] <= '9') {
    epoch = epoch * 10 + static_cast<uint64_t>(v[k] - '0');
    ++k;
    any = true;
  }
  return any ? epoch : 0;
}

}  // namespace

// ===== notify poller =====

void MooncakeEngineConnector::NotifyPollerLoop() {
  std::vector<Notification> drain;
  while (!poller_stop_.load()) {
    drain.clear();
    Status s = tent_->ReceiveNotifications(&drain);
    if (!s.OK()) {
      KLLM_LOG_WARNING << "NotifyPollerLoop: ReceiveNotifications failed: " << s.GetMessage();
    }
    for (auto& n : drain) DispatchOne(n);
    // Only back off when the queue was empty; keep draining at full speed
    // while notifications are pending to minimize per-notification latency.
    if (drain.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollerIntervalMs));
    }
  }
}

void MooncakeEngineConnector::DispatchOne(const Notification& n) {
  KLLM_LOG_DEBUG << "MooncakeEngineConnector::DispatchOne name=" << n.name << " msg.size=" << n.msg.size();
  if (n.name == kChPrefillReq)
    OnPrefillReq(n);
  else if (n.name == kChPrefillComplete)
    OnPrefillComplete(n);
  else if (n.name == kChPrefillSnapshot)
    OnPrefillSnapshot(n);
  else if (n.name == kChStageAck)
    OnStageAck(n);
  else
    KLLM_LOG_WARNING << "MooncakeEngineConnector: unknown channel '" << n.name << "'";
}

void MooncakeEngineConnector::OnPrefillSnapshot(const Notification& n) {
  PrefillSnapshot snap;
  if (DeserializePrefillSnapshot(reinterpret_cast<const uint8_t*>(n.msg.data()), n.msg.size(), &snap) == 0) {
    return;
  }
  // Phase 1: resolve the local SegmentID and refresh the PeerSelector
  // snapshot. peer_selector_ / peer_server_name_by_seg_ are guarded by
  // state_mu_, but openSegment() is a slow Mooncake metadata/network RPC
  // (tens to hundreds of ms, and it can block on timeout). Holding state_mu_
  // across it would stall every CanAccept / SubmitPrefillRequest / Release on
  // the Decode engine hot path and inflate TTFT. So we use double-checked
  // locking: probe the name cache under the lock, drop the lock to run the
  // slow openSegment on a miss, then re-acquire to commit the result while
  // handling a concurrent opener.
  //
  // We deliberately finish and release state_mu_ before taking
  // stage_peers_mu_ below — see the lock-order convention in the header
  // (state_mu_ -> stage_peers_mu_, never nested) so this stays consistent
  // with Release / OnStageAck.
  uint64_t local_seg_id = 0;
  if (!snap.snapshot.server_name.empty()) {
    // Discovery path: a new Prefill we've never opened sends us its first
    // snapshot. Fall back to server_name-based lookup; if unknown, openSegment
    // (outside the lock) so subsequent SubmitPrefillRequest can route via the
    // same SegmentID. server_name comes from snap.snapshot.server_name
    // (Prefill always populates its own inference_addr); seg_id from
    // snap.snapshot.segment_id is the sender's local handle (unusable here) —
    // we discard it and use our local openSegment-resolved one.
    mooncake::TransferEngine* tent_raw = nullptr;
    {
      std::lock_guard<std::mutex> lk(state_mu_);
      if (peer_selector_ == nullptr || tent_ == nullptr) return;
      auto cached = peer_selector_->SegmentIdByName(snap.snapshot.server_name);
      if (cached.has_value()) {
        local_seg_id = *cached;
      } else {
        // Capture the raw engine handle for the unlocked openSegment below;
        // tent_ outlives the notify-poller thread that runs this callback.
        tent_raw = tent_->raw();
      }
    }
    if (tent_raw != nullptr) {
      mooncake::SegmentHandle seg = tent_raw->openSegment(snap.snapshot.server_name);
      if (static_cast<int64_t>(seg) < 0) {
        KLLM_LOG_WARNING << "MooncakeEngineConnector::OnPrefillSnapshot openSegment(" << snap.snapshot.server_name
                         << ") failed; dropping snapshot";
        return;
      }
      std::lock_guard<std::mutex> lk(state_mu_);
      if (peer_selector_ == nullptr) return;
      // Re-check under the lock: another snapshot may have opened + registered
      // the same peer while we were blocked in openSegment. openSegment is
      // idempotent by name in Mooncake (returns the same handle), so prefer
      // the already-registered seg_id to keep one canonical mapping.
      auto cached = peer_selector_->SegmentIdByName(snap.snapshot.server_name);
      if (cached.has_value()) {
        local_seg_id = *cached;
      } else {
        local_seg_id = static_cast<uint64_t>(seg);
        peer_selector_->RegisterPeer(local_seg_id, snap.snapshot.server_name);
        peer_server_name_by_seg_[local_seg_id] = snap.snapshot.server_name;
        KLLM_LOG_INFO << "MooncakeEngineConnector::OnPrefillSnapshot discovered new Prefill="
                      << snap.snapshot.server_name << " local_seg_id=" << local_seg_id;
      }
    }
  } else if (snap.snapshot.segment_id != 0) {
    // Legacy / mid-life snapshot without server_name — keep old behavior
    // (update if the sender's seg_id happens to collide with ours).
    local_seg_id = snap.snapshot.segment_id;
  } else {
    KLLM_LOG_WARNING << "MooncakeEngineConnector::OnPrefillSnapshot snapshot has neither"
                     << " server_name nor segment_id; dropping";
    return;
  }
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    if (peer_selector_ == nullptr) return;
    peer_selector_->UpdateSnapshot(local_seg_id, snap.snapshot);
  }

  // Phase 2: stage assignment side-channel (Decode-only state). Each Prefill
  // unicast carries this peer's HostStagePool partition; cache it
  // keyed by Prefill server_name and lazy-init the PeerCredits ring
  // the first time we see a non-zero assignment. Zero n_slots means
  // Prefill hasn't allocated us a partition yet (or pool full) — we
  // hold off on routing to that peer until a non-zero value lands.
  if (!snap.snapshot.server_name.empty() && snap.stage.n_slots > 0) {
    std::lock_guard<std::mutex> sg(stage_peers_mu_);
    auto& entry = stage_peers_[snap.snapshot.server_name];
    const bool first_time = (entry.credits == nullptr);
    const bool changed = entry.assignment.pool_base_va != snap.stage.pool_base_va ||
                         entry.assignment.base_slot != snap.stage.base_slot ||
                         entry.assignment.n_slots != snap.stage.n_slots ||
                         entry.assignment.slot_size != snap.stage.slot_size;
    entry.assignment = snap.stage;
    if (first_time || changed) {
      // Re-init on Prefill restart: pool_base_va flipping is the
      // canonical signal a new mapping is in play. Shutting down the
      // old ring forces any in-flight AcquireSlots waiters to fail
      // fast (RET_RUNTIME_FAILED), which the Decode hook surfaces as
      // a request-level failure — preferable to silently writing into
      // a partition that no longer exists.
      // Shutdown() wakes any in-flight AcquireSlots waiter so it fails
      // fast. We then drop our reference by reassigning the slot; an
      // in-flight SubmitPrefillRequest may still hold a shared_ptr to the
      // old ring (keeping it alive for its release), but no new caller can
      // observe it once the map slot points at the fresh ring.
      if (entry.credits) entry.credits->Shutdown();
      entry.credits = std::make_shared<PeerCredits>();
      Status is = entry.credits->Init(snap.stage.base_slot, snap.stage.n_slots);
      if (!is.OK()) {
        KLLM_LOG_WARNING << "OnPrefillSnapshot: PeerCredits::Init for " << snap.snapshot.server_name
                         << " base_slot=" << snap.stage.base_slot << " n_slots=" << snap.stage.n_slots
                         << " failed: " << is.GetMessage();
        entry.credits.reset();
      } else {
        KLLM_LOG_INFO << "OnPrefillSnapshot: stage assignment " << (first_time ? "registered" : "updated")
                      << " peer=" << snap.snapshot.server_name << " pool_base_va=0x" << std::hex
                      << snap.stage.pool_base_va << std::dec << " base_slot=" << snap.stage.base_slot
                      << " n_slots=" << snap.stage.n_slots << " slot_size=" << snap.stage.slot_size;
      }
    }
  }
}

void MooncakeEngineConnector::HeartbeatLoop() {
  // §3.3 Prefill discovery + heartbeat. Wakes at snapshot_push_interval_ms,
  // calls BroadcastSnapshot which polls Redis KEYS pd_v2/decode_alive/<cluster>/* (or
  // falls back to incoming_routes_ when Redis isn't configured).
  const uint32_t period_ms =
      config_.engine.snapshot_push_interval_ms == 0 ? 1000 : config_.engine.snapshot_push_interval_ms;
  while (!heartbeat_stop_.load(std::memory_order_acquire)) {
    BroadcastSnapshot();
    // Sleep in 100ms slices so Shutdown can break out promptly.
    for (uint32_t slept = 0; slept < period_ms && !heartbeat_stop_.load(std::memory_order_acquire); slept += 100) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

void MooncakeEngineConnector::RedisKeepaliveLoop() {
  // §3.3 Decode keepalive. Boot: DEL stale + SET fresh; loop: SET every
  // 2s with TTL=5 so a single missed renew leaves 3s of headroom. On
  // shutdown the dtor path DELs the key so peers see us go immediately.
  const std::string key = AliveKeyPrefix(config_.cluster_name) + config_.inference_addr;
  const std::string value = fmt::format("{{\"epoch\":{}}}", boot_epoch_ms_);

  if (redis_alive_ == nullptr) return;

  // Boot: DEL any leftover key from a previous incarnation that didn't
  // get to clean up (crash exit). Best-effort; ignore failure.
  Status s = redis_alive_->Del(key);
  if (!s.OK()) {
    KLLM_LOG_WARNING << "RedisKeepaliveLoop: boot DEL " << key << " failed: " << s.GetMessage();
  }
  s = redis_alive_->Set(key, value, kAliveTtlSec);
  if (!s.OK()) {
    KLLM_LOG_WARNING << "RedisKeepaliveLoop: boot SET " << key << " failed: " << s.GetMessage();
  } else {
    KLLM_LOG_INFO << "RedisKeepaliveLoop: registered " << key << " value=" << value << " ttl=" << kAliveTtlSec << "s";
  }

  while (!keepalive_stop_.load(std::memory_order_acquire)) {
    // Sleep in 100ms slices for prompt shutdown.
    for (uint32_t slept = 0; slept < kAliveRenewMs && !keepalive_stop_.load(std::memory_order_acquire); slept += 100) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (keepalive_stop_.load(std::memory_order_acquire)) break;
    s = redis_alive_->Set(key, value, kAliveTtlSec);
    if (!s.OK()) {
      KLLM_LOG_WARNING << "RedisKeepaliveLoop: renew SET " << key << " failed: " << s.GetMessage()
                       << " — will retry in " << kAliveRenewMs << "ms";
    }
  }

  // Shutdown: DEL my own key so Prefill sees us drop immediately rather
  // than waiting for TTL. Failure is fine — Redis TTL still bounds it to
  // kAliveTtlSec at worst.
  s = redis_alive_->Del(key);
  if (!s.OK()) {
    KLLM_LOG_WARNING << "RedisKeepaliveLoop: shutdown DEL " << key << " failed: " << s.GetMessage();
  } else {
    KLLM_LOG_INFO << "RedisKeepaliveLoop: deregistered " << key;
  }
}

void MooncakeEngineConnector::BroadcastSnapshot() {
  if (tent_ == nullptr) return;

  // Build the per-recipient PrefillSnapshot. Snapshot.snapshot is
  // identical across recipients; snapshot.stage is per-peer (each
  // Decode owns a private partition of the HostStagePool). We
  // therefore serialize once *per* peer rather than once per cycle.
  PrefillSnapshot snap_template;
  snap_template.snapshot.server_name = config_.inference_addr;  // Decode keys by server_name
  snap_template.snapshot.segment_id = 0;                        // Decode discards (uses local openSegment)
  snap_template.snapshot.healthy = true;
  // Load fields: filled from snapshot_provider_ when the engine bootstrap
  // wired one (Prefill role). PeerSelector only uses the healthy bit for
  // round-robin remote-peer selection.
  snap_template.snapshot.free_kv_blocks = 0;
  snap_template.snapshot.last_snapshot_ms = NowMs();
  if (snapshot_provider_) {
    const PeerHealth local = snapshot_provider_();
    // Preserve server_name / segment_id / healthy / last_snapshot_ms set
    // above (BroadcastSnapshot owns sender identity); copy only the load
    // features the provider measures.
    snap_template.snapshot.free_kv_blocks = local.free_kv_blocks;
    snap_template.snapshot.future_free_kv_blocks = local.future_free_kv_blocks;
    snap_template.snapshot.waiting_reqs = local.waiting_reqs;
    snap_template.snapshot.running_reqs = local.running_reqs;
    snap_template.snapshot.pending_prefill_tokens = local.pending_prefill_tokens;
    snap_template.snapshot.step_token_num = local.step_token_num;
    snap_template.snapshot.step_latency_us = local.step_latency_us;
  }

  // §3.3 source-of-truth selection:
  //   - Redis available → KEYS pd_v2/decode_alive/<cluster>/* (true discovery, S3)
  //   - Redis missing   → fallback to incoming_routes_ (already-contacted
  //                       Decodes only, S1 behavior)
  std::vector<std::string> decode_addrs;
  if (redis_alive_ != nullptr && redis_alive_->IsConnected()) {
    std::vector<std::string> alive_keys;
    const std::string alive_prefix = AliveKeyPrefix(config_.cluster_name);
    Status ks = redis_alive_->Keys(alive_prefix + "*", &alive_keys);
    if (!ks.OK()) {
      KLLM_LOG_WARNING << "BroadcastSnapshot: Redis KEYS failed: " << ks.GetMessage() << " — skipping this cycle";
      return;
    }

    // Diff against cache: detect first-seen, restarted, and disappeared.
    std::set<std::string> current_addrs;
    const size_t kPrefixLen = alive_prefix.size();
    for (const auto& key : alive_keys) {
      if (key.size() <= kPrefixLen) continue;
      if (key.compare(0, kPrefixLen, alive_prefix) != 0) continue;
      std::string addr = key.substr(kPrefixLen);
      if (addr.empty() || addr == config_.inference_addr) continue;  // skip self

      std::string value;
      if (!redis_alive_->Get(key, &value).OK() || value.empty()) continue;
      uint64_t new_epoch = ParseEpochFromJson(value);
      if (new_epoch == 0) {
        KLLM_LOG_WARNING << "BroadcastSnapshot: malformed alive value for " << key << " ('" << value << "'); skipping";
        continue;
      }
      current_addrs.insert(addr);

      auto& entry = decode_epoch_cache_[addr];
      const bool first_seen = (entry == 0);
      const bool restarted = !first_seen && entry != new_epoch;

      if (first_seen || restarted) {
        // Force engine-process Mooncake metadata cache to re-read from
        // Redis: the open+immediate-close loop below is the only public
        // hook for cache invalidation (closeSegment drops a stale cached
        // SegmentDesc; the next openSegment re-fetches it). See design
        // §3.3.3.
        //
        // attn_tp_decode is the LOCAL engine's attn_tp (Prefill side
        // reflects Prefill's own value, not the remote Decode's). We use
        // it as a best-effort estimate for how many .exec.r{N} segments
        // the Decode published. openSegment failures are silently
        // skipped; engine-side close/open is for engine's own cache only,
        // executor-side cache flush is a separate concern.
        //
        // IMPORTANT: open+immediately close, do NOT retain the handle.
        // The engine process never RDMAs to .exec.r{N} (that's the
        // executor subprocess's domain — it has its own TransferEngine
        // instance with its own private metadata cache). Holding the
        // handle here keeps a notify QP alive per (NIC pair × exec rank)
        // for no purpose, and mooncake's notify_qp_to_endpoint_ map is
        // keyed by qp_num only — qp_num is per-context (per-NIC) in
        // mlx5, so multi-NIC + multi-exec collisions corrupt the map
        // and cause notify CQEs to be dispatched to the wrong EP
        // (manifests as Prefill receiving prefill_req with byte_len=
        // 1629 but buffer all zeros after the first long request).
        //
        // The open+close still triggers Mooncake's per-process metadata
        // refresh from Redis if the SegmentDesc was cached, so the
        // "force re-read" intent in the comment above is preserved.
        uint32_t opened = 0;
        const uint32_t n = config_.engine.attn_tp_decode == 0 ? 1 : config_.engine.attn_tp_decode;
        for (uint32_t r = 0; r < n; ++r) {
          const std::string seg_name = addr + ".exec.r" + std::to_string(r);
          mooncake::SegmentHandle seg = tent_->raw()->openSegment(seg_name);
          if (static_cast<int64_t>(seg) >= 0) {
            tent_->raw()->closeSegment(seg);
            ++opened;
          }
        }
        // The .exec.r{N} refresh above only covers the executor DATA path.
        // The engine NOTIFY path targets the bare `addr` segment (see
        // SendNotification callers), whose handle is cached separately in
        // EngineTentWrapper::peer_handle_cache_. Without this, a restarted
        // Decode leaves that cache pointing at a torn-down notify QP and every
        // subsequent notify flushes with IBV_WC_WR_FLUSH_ERR (status=5). Force
        // a metadata re-read for `addr` (open+immediate-close, same trick as
        // above) and drop the stale cached handle so the next SendNotification
        // rebuilds the notify QP against the restarted peer.
        //
        // This MUST run on first_seen too, not just epoch-refreshed: when a
        // Decode is down long enough for its Redis alive key to expire, the
        // disappeared-cleanup below erases decode_epoch_cache_[addr], so on
        // restart it reappears here as first_seen — never as restarted. The
        // stale peer_handle_cache_ entry survives that whole cycle, and
        // sendNotifyByID returns success while silently flushing on the dead
        // QP (so SendNotification's failure-path eviction never fires either).
        // Unconditional refresh is the only thing that reliably breaks the
        // stale-handle loop. On a genuine first discovery the cache is empty,
        // so InvalidatePeerHandle is a no-op and the open+close just re-reads
        // already-fresh metadata — harmless.
        mooncake::SegmentHandle notify_seg = tent_->raw()->openSegment(addr);
        if (static_cast<int64_t>(notify_seg) >= 0) {
          tent_->raw()->closeSegment(notify_seg);
        }
        tent_->InvalidatePeerHandle(addr);
        entry = new_epoch;
        KLLM_LOG_INFO << "BroadcastSnapshot: " << (first_seen ? "discovered" : "epoch-refreshed") << " Decode=" << addr
                      << " epoch=" << new_epoch << " refreshed_segs=" << opened;
      }

      decode_addrs.push_back(addr);
    }

    // Drop Decodes that disappeared from Redis (graceful exit DEL'd or TTL
    // expired). No engine-side handles are retained, so just forget them.
    for (auto it = decode_epoch_cache_.begin(); it != decode_epoch_cache_.end();) {
      if (current_addrs.count(it->first) == 0) {
        KLLM_LOG_INFO << "BroadcastSnapshot: Decode " << it->first << " disappeared from Redis, forgetting epoch";
        // TODO(shawnding): 这里只清理了 epoch 缓存，stage_pool_ 中该 addr 占用的 partition
        // 没有回收。Decode 永久下线（或地址变化）后会造成 partition 泄漏，累计
        // 出现 > max_peers 个不同地址后新 peer 将无法分配。后续应在此调用
        // stage_pool_->UnregisterPeer(it->first) 归还槽位（需先实现回收 free-list）。
        it = decode_epoch_cache_.erase(it);
      } else {
        ++it;
      }
    }
  } else {
    // Fallback: iterate incoming_routes_ (S1 behavior, no service discovery).
    std::lock_guard<std::mutex> lk(state_mu_);
    decode_addrs.reserve(incoming_routes_.size());
    for (const auto& kv : incoming_routes_) {
      decode_addrs.push_back(kv.first.first);  // IncomingKey = (decode_addr, req_id)
    }
    std::sort(decode_addrs.begin(), decode_addrs.end());
    decode_addrs.erase(std::unique(decode_addrs.begin(), decode_addrs.end()), decode_addrs.end());
  }

  if (decode_addrs.empty()) return;

  for (const auto& addr : decode_addrs) {
    // Per-peer StageAssignment: register the peer (idempotent — first
    // call consumes a partition, subsequent calls return the same one)
    // and fill the wire fields. RegisterPeer fails only when the pool
    // is already at max_peers AND this addr is new; in that case skip
    // the snapshot for this peer (Decode treats absent assignment as
    // "not ready" and won't route to us). stage_pool_ is null only if
    // we somehow got here without going through Prefill-init.
    PrefillSnapshot snap = snap_template;
    if (stage_pool_) {
      HostStagePool::PeerPartition part;
      Status rs = stage_pool_->RegisterPeer(addr, &part);
      if (!rs.OK()) {
        KLLM_LOG_WARNING << "BroadcastSnapshot: stage_pool_->RegisterPeer(" << addr << ") failed: " << rs.GetMessage()
                         << " — skipping snapshot to this peer";
        continue;
      }
      snap.stage.pool_base_va = reinterpret_cast<uint64_t>(stage_pool_->base());
      snap.stage.slot_size = static_cast<uint32_t>(HostStagePool::kSlotSize);
      snap.stage.base_slot = static_cast<uint32_t>(part.base_slot);
      snap.stage.n_slots = static_cast<uint32_t>(part.n_slots);
    }

    std::vector<uint8_t> body;
    SerializePrefillSnapshot(snap, &body);
    Notification notify{kChPrefillSnapshot, std::string(body.begin(), body.end())};
    Status s = tent_->SendNotification(addr, notify);
    if (!s.OK()) {
      // Silent on per-peer failure — Decode side TTL / S2 reactive
      // removal / next heartbeat handle recovery.
      KLLM_LOG_DEBUG << "MooncakeEngineConnector::BroadcastSnapshot send to " << addr << " failed: " << s.GetMessage();
    }
  }
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
