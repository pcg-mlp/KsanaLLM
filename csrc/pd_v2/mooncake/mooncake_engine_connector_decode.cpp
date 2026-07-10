/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// MooncakeEngineConnector Decode-role logic: admission + LinUCB routing, the
// multi-phase SubmitPrefillRequest pipeline (build wire frame → acquire peer
// stage slots → RDMA-WRITE staged payload → notify) with its rollbacks, and
// the Decode-received notify handlers OnPrefillComplete / OnStageAck. Snapshot
// reception (service discovery) lives in the discovery translation unit.
#include "pd_v2/mooncake/mooncake_engine_connector.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include "fmt/core.h"
#include "pd_v2/mooncake/mooncake_engine_connector_internal.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

bool MooncakeEngineConnector::CanAccept(uint64_t req_id, uint32_t /*input_token_count*/) {
  if (peer_selector_ == nullptr) {
    KLLM_LOG_ERROR << "CanAccept before Initialize, req=" << req_id;
    return false;
  }
  std::lock_guard<std::mutex> lk(state_mu_);
  if (in_flight_.count(req_id) != 0) return false;
  auto picked = peer_selector_->Select();
  if (!picked.has_value()) return false;
  auto name_it = peer_server_name_by_seg_.find(*picked);
  if (name_it == peer_server_name_by_seg_.end()) {
    KLLM_LOG_WARNING << "CanAccept: selected segment_id " << *picked
                     << " has no cached server_name, dropping admission for req " << req_id;
    return false;
  }
  AdmissionState st;
  st.peer_server_name = name_it->second;
  in_flight_.emplace(req_id, st);
  return true;
}

std::optional<LinUCBVec> MooncakeEngineConnector::EraseInFlightLocked(uint64_t req_id) {
  auto it = in_flight_.find(req_id);
  if (it == in_flight_.end()) return std::nullopt;
  std::optional<LinUCBVec> remote_feat;
  if (linucb_ != nullptr && it->second.has_router_features) {
    remote_feat = it->second.router_arm_features;
  }
  in_flight_.erase(it);
  return remote_feat;
}

void MooncakeEngineConnector::EraseInFlight(uint64_t req_id) {
  std::optional<LinUCBVec> remote_feat;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    remote_feat = EraseInFlightLocked(req_id);
  }
  // Control-plane abort on the REMOTE path: the peer was never notified, so
  // OnPrefillComplete will never fire to settle this arm. Penalize the REMOTE
  // arm with a zero reward here (outside state_mu_ — RecordReward takes the
  // router's own mutex and must never nest under it).
  if (remote_feat.has_value()) {
    linucb_->RecordReward(RouteArm::kRemotePeer, *remote_feat, 0.0);
  }
}

void MooncakeEngineConnector::SampleLocalReward(const LocalArmContext& ctx) {
  if (linucb_ == nullptr) return;
  std::vector<LocalPending> pending;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    if (router_local_pending_.empty()) return;
    pending.reserve(router_local_pending_.size());
    for (auto& kv : router_local_pending_) pending.push_back(kv.second);
    router_local_pending_.clear();
  }
  // Multiplicative AND gate, settled one step late: ttft_credit was sampled at
  // select time (how badly REMOTE would have blown the TTFT budget); tpot_head
  // is sampled now (how much TPOT headroom remains after the earlier kLocal
  // request(s) folded into local load). Keeping prefill local earns reward
  // only when REMOTE would have been slow AND we still had headroom to absorb
  // it.
  // Gate, don't multiply. ttft_credit (sampled at select time) is the trigger:
  // > 0 means a REMOTE dispatch would have breached the TTFT budget, so keeping
  // prefill local is worth considering. tpot_head (sampled now) is the
  // amplitude: how much TPOT headroom we still have to absorb the local
  // prefill. A multiplicative product collapses the amplitude (e.g. 0.19*0.20
  // ≈ 0.04) and loses to the REMOTE arm's residual reward, so the bandit never
  // exploits free TPOT headroom. Gating keeps LOCAL at the full tpot_head once
  // the trigger fires, while still honoring "only keep local when REMOTE would
  // be slow AND we have headroom" (ttft_credit gate AND tpot_head > 0).
  const double tpot_head = linucb_->LocalReward(ctx);
  for (const auto& p : pending) {
    const double reward = p.ttft_credit > 0.0 ? tpot_head : 0.0;
    linucb_->RecordReward(RouteArm::kLocal, p.features, reward);
  }
}

std::optional<RouteArm> MooncakeEngineConnector::RouteRequest(uint64_t req_id, const LocalArmContext& ctx) {
  // Non-kLinUCB (linucb_ == null): behave exactly like CanAccept — forward
  // when a peer is available, otherwise fail. Zero behavior change.
  if (linucb_ == nullptr) {
    return CanAccept(req_id, ctx.remaining_prefill_tokens) ? std::optional<RouteArm>(RouteArm::kRemotePeer)
                                                           : std::nullopt;
  }

  // Settle deferred LOCAL rewards against the current load before deciding.
  SampleLocalReward(ctx);

  LinUCBVec feat{};
  const RouteArm arm = linucb_->SelectArm(ctx, &feat);

  if (arm == RouteArm::kLocal) {
    // Keep prefill local: no peer round-trip, no in_flight_ entry. Stash the
    // select-time features AND the select-time ttft_credit (how badly a REMOTE
    // dispatch would have blown the TTFT budget right now, derived from the
    // aggregate remote round-trip EWMA) for the deferred LOCAL reward. Read the
    // EWMA under peer_latency_mu_ first to avoid nesting it under state_mu_.
    double rtt_ewma_us;
    {
      std::lock_guard<std::mutex> ll(peer_latency_mu_);
      rtt_ewma_us = remote_rtt_ewma_us_;
    }
    const double ttft_credit = linucb_->TtftCredit(rtt_ewma_us);
    std::lock_guard<std::mutex> lk(state_mu_);
    router_local_pending_[req_id] = LocalPending{feat, ttft_credit};
    return RouteArm::kLocal;
  }

  // REMOTE: reuse the existing admission path (PeerSelector::Select() +
  // in_flight_ bookkeeping). nullopt (no healthy peer) fails the request.
  if (!CanAccept(req_id, ctx.remaining_prefill_tokens)) {
    return std::nullopt;
  }
  // Stash select-time features so OnPrefillComplete rewards the REMOTE arm
  // with the matching context.
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    auto it = in_flight_.find(req_id);
    if (it != in_flight_.end()) {
      it->second.router_arm_features = feat;
      it->second.has_router_features = true;
    }
  }
  return RouteArm::kRemotePeer;
}

Status MooncakeEngineConnector::BuildPrefillRequest(uint64_t req_id, const SamplingConfigWire& sampling_config,
                                                    uint32_t attn_dp_group_id, uint32_t cached_block_count,
                                                    uint32_t decode_prefix_cache_offset, PrefillRequest* req) {
  // decode_targets[].kv_pool_base_va is filled here per local Decode
  // attn-TP rank via GetKvBufferVa (IPC RPC to the per-rank executor
  // process, cached after first miss).
  req->decode_engine_inference_addr = config_.inference_addr;
  req->req_id = req_id;
  req->sampling_config = sampling_config;
  req->decode_attn_dp_group_id = attn_dp_group_id;

  const uint32_t attn_tp = config_.engine.attn_tp_decode == 0 ? 1 : config_.engine.attn_tp_decode;
  req->decode_targets.reserve(attn_tp);
  for (uint32_t r = 0; r < attn_tp; ++r) {
    DecodeTarget t;
    t.decode_attn_tp_rank_in_group = r;
    // Each request belongs to exactly one attn_dp_group
    // (BasicWorkloadBalancer assigns it at admission). The r-th
    // DecodeTarget addresses the executor that owns rank
    // (attn_dp_group_id * attn_tp + r) — i.e. the r-th attn rank inside
    // this request's DP group. Without folding the DP-group offset in,
    // every request would be routed to attn_dp_group=0 since
    // GetKvBufferVa(rank) routes IPC by executor_id == device_id.
    const int device_id = static_cast<int>(attn_dp_group_id) * static_cast<int>(attn_tp) + static_cast<int>(r);
    uint64_t base_va = 0, total = 0;

    // 所以这里 SubmitPrefillRequest 每次调用 GetKvBufferVa，但语义上是“获取可用地址”，
    // 实际实现是 先查 Engine 本地缓存，miss 才 IPC。steady state 下基本都是内存 map 查询，
    // 不会每个请求都跨进程。
    Status s = GetKvBufferVa(device_id, &base_va, &total);
    if (!s.OK()) {
      KLLM_LOG_ERROR << "SubmitPrefillRequest: GetKvBufferVa(device_id=" << device_id
                     << " attn_dp_group_id=" << attn_dp_group_id << " r=" << r << ") failed: " << s.GetMessage()
                     << ", aborting req " << req_id;
      return s;
    }
    t.kv_pool_base_va = base_va;
    // kv_block_ids no longer carried per-target on the wire — moved to
    // PrefillStagePayload to stay under the 64 KB Tent notify limit.
    // Prefill receiver populates each target's kv_block_ids from the
    // stage payload after RDMA read.
    // pd_v2 prefix-cache hint (block-level + token-level). Hook computed
    // these from req->prefix_cache_len; we just propagate to wire so the
    // peer Prefill dispatcher can skip re-writing the leading cached
    // blocks. Same value applies to all decode_targets (every rank
    // shares the same block layout for a single request).
    t.cached_block_count = cached_block_count;
    t.decode_prefix_cache_offset = decode_prefix_cache_offset;
    req->decode_targets.push_back(std::move(t));
  }
  return Status();
}

Status MooncakeEngineConnector::AcquirePeerStageSlots(const std::string& peer_server_name, size_t stage_bytes,
                                                      int64_t timeout_ms, std::shared_ptr<PeerCredits>* out_credits,
                                                      uint32_t* out_start_slot, uint32_t* out_n_slots,
                                                      uint64_t* out_target_offset) {
  // Look up the peer's StageAssignment + PeerCredits ring (received via
  // PrefillSnapshot). Refuse to send if the peer hasn't published a
  // stage assignment yet — that's a degraded state Decode shouldn't
  // route into (CanAccept ideally already filters those, but this is
  // a belt-and-braces check; the alternative is silently writing into
  // an unregistered slot).
  //
  // Snapshot credits + slot_size + pool_base_va together under a single
  // lock acquisition. These three values must be mutually consistent:
  // n_slots is derived from slot_size and target_offset is derived from
  // pool_base_va + slot_size, and AcquireSlots() reserves a range against
  // *this* ring. A concurrent OnPrefillSnapshot (Prefill restart) can
  // atomically replace all three at once, so reading slot_size/pool_base_va
  // in a second, later lock section would risk pairing a start_slot from
  // the old partition with a base VA / slot_size from the new one →
  // an out-of-bounds remote VA and a silent RDMA WRITE into the wrong
  // memory. Taking one snapshot here removes that TOCTOU window. We hold a
  // shared_ptr copy of the ring so the blocking AcquireSlots below (and any
  // later ReleaseSlots by the caller) operate on the ring we sized against,
  // even if OnPrefillSnapshot swaps stage_peers_[peer] out from under us.
  std::shared_ptr<PeerCredits> credits;
  uint32_t n_slots = 0;
  uint64_t pool_base_va = 0;
  uint32_t slot_size = 0;
  std::string failure_reason;
  {
    std::lock_guard<std::mutex> sg(stage_peers_mu_);
    auto it = stage_peers_.find(peer_server_name);
    if (it == stage_peers_.end()) {
      // No stage_peers_ entry for this peer. Either it was chosen at
      // RouteRequest time and then evicted (S2 send-failure eviction /
      // PeerSelector::UnregisterPeer) before this submit ran — a
      // RouteRequest→Submit TOCTOU — or the peer never published a stage
      // assignment snapshot at all. Call it out explicitly so operators
      // don't mistake a peer-churn failure for stage-slot-pool pressure.
      failure_reason = fmt::format(
          "peer {} not in stage_peers_ (evicted between RouteRequest and Submit, "
          "or stage snapshot never received)",
          peer_server_name);
    } else if (it->second.credits == nullptr) {
      failure_reason =
          fmt::format("peer {} stage ring not initialized (PeerCredits::Init failed or ring reset on Prefill restart)",
                      peer_server_name);
    } else if (it->second.assignment.slot_size == 0) {
      failure_reason = fmt::format("peer {} stage assignment not published yet (slot_size=0)", peer_server_name);
    } else {
      credits = it->second.credits;
      slot_size = it->second.assignment.slot_size;
      pool_base_va = it->second.assignment.pool_base_va;
      n_slots = static_cast<uint32_t>((stage_bytes + slot_size - 1) / slot_size);
    }
  }
  if (credits == nullptr) {
    KLLM_LOG_ERROR << "SubmitPrefillRequest: " << failure_reason << ", aborting";
    return Status(RET_RUNTIME_FAILED, failure_reason);
  }

  uint32_t start_slot = 0;
  Status as = credits->AcquireSlots(n_slots, timeout_ms, &start_slot);
  if (!as.OK()) {
    // Non-blocking try (timeout_ms == 0) failing is the common fallback
    // probe, not an error — log at DEBUG so the peer-hopping path stays
    // quiet. A blocking acquire timing out is the real backpressure event.
    if (timeout_ms == 0) {
      KLLM_LOG_DEBUG << "SubmitPrefillRequest: try-acquire(n=" << n_slots << ") for peer " << peer_server_name
                     << " unavailable: " << as.GetMessage();
    } else {
      KLLM_LOG_ERROR << "SubmitPrefillRequest: AcquireSlots(n=" << n_slots << ") for peer " << peer_server_name
                     << " failed: " << as.GetMessage() << ", aborting";
    }
    return as;
  }

  // Compute target_offset from the same snapshot we sized the acquire
  // against — start_slot is an index into this ring's partition, so it must
  // be paired with this snapshot's pool_base_va + slot_size.
  //
  // Mooncake quirk: TransferRequest::target_offset is the *absolute*
  // remote VA (segment.cpp:findBuffer checks `entry.addr <= base &&
  // base + length <= entry.addr + entry.length`), not an offset from
  // the registered buffer's start. So we add the remote pool's base VA
  // — Decode learned it via the PrefillSnapshot.stage.pool_base_va
  // field. Passing 0 / a relative offset trips "Unable to find
  // registered buffer" and the WRITE terminal-fails.
  uint64_t target_offset = pool_base_va + static_cast<uint64_t>(start_slot) * static_cast<uint64_t>(slot_size);

  *out_credits = std::move(credits);
  *out_start_slot = start_slot;
  *out_n_slots = n_slots;
  *out_target_offset = target_offset;
  return Status();
}

Status MooncakeEngineConnector::AcquirePeerStageSlotsWithFallback(const std::string& primary_peer, size_t stage_bytes,
                                                                  std::string* out_used_peer,
                                                                  std::shared_ptr<PeerCredits>* out_credits,
                                                                  uint32_t* out_start_slot, uint32_t* out_n_slots,
                                                                  uint64_t* out_target_offset) {
  // Step 1: non-blocking try on the primary peer chosen at admission.
  Status s = AcquirePeerStageSlots(primary_peer, stage_bytes, /*timeout_ms=*/0, out_credits, out_start_slot,
                                   out_n_slots, out_target_offset);
  if (s.OK()) {
    *out_used_peer = primary_peer;
    return s;
  }

  // Step 2: the primary peer's stage pool is momentarily full. Try every
  // OTHER healthy peer non-blocking before we resort to blocking — a
  // single full peer shouldn't stall the request while siblings have
  // room. Resolve segment_id -> server_name under state_mu_ once, then
  // probe outside the lock (AcquireSlots takes stage_peers_mu_).
  const std::vector<uint64_t> healthy =
      peer_selector_ != nullptr ? peer_selector_->HealthyPeers() : std::vector<uint64_t>{};
  std::vector<std::string> candidate_names;
  candidate_names.reserve(healthy.size());
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    for (uint64_t seg : healthy) {
      auto it = peer_server_name_by_seg_.find(seg);
      if (it == peer_server_name_by_seg_.end()) continue;
      if (it->second == primary_peer) continue;
      candidate_names.push_back(it->second);
    }
  }
  for (const std::string& name : candidate_names) {
    Status alt = AcquirePeerStageSlots(name, stage_bytes, /*timeout_ms=*/0, out_credits, out_start_slot, out_n_slots,
                                       out_target_offset);
    if (alt.OK()) {
      KLLM_LOG_INFO << "SubmitPrefillRequest: primary peer " << primary_peer << " stage pool full, rerouted to "
                    << name;
      *out_used_peer = name;
      return alt;
    }
  }

  // Step 3: all peers are momentarily full. Block up to the configured
  // acquire timeout on the primary peer — this is genuine backpressure
  // (the whole fleet's stage pools are saturated). It runs on a submit
  // worker, so it never stalls the EDS scheduling thread.
  Status blocked = AcquirePeerStageSlots(primary_peer, stage_bytes, config_.engine.control_stage_acquire_timeout_ms,
                                         out_credits, out_start_slot, out_n_slots, out_target_offset);
  if (blocked.OK()) {
    *out_used_peer = primary_peer;
  }
  return blocked;
}

Status MooncakeEngineConnector::StageWriteAndWait(const std::string& peer_server_name, uint64_t target_offset,
                                                  const std::vector<uint8_t>& stage_buf, uint32_t n_slots) {
  // Acquire local write-side slots and copy the serialized stage_buf into the
  // registered write-side scratch arena. Mooncake's submitTransfer rejects
  // sources that aren't in registered memory — a heap-allocated
  // std::vector wouldn't be found in the engine's registration table.
  // The local write-side slot range is independent of the peer's
  // read-side slot range; we release it locally as soon as the WRITE drains.
  if (tx_credits_ == nullptr || tx_pool_ == nullptr) {
    return Status(RET_RUNTIME_FAILED,
                  "SubmitPrefillRequest: write-side scratch not initialized "
                  "(node_role != kDecode at boot?)");
  }
  uint32_t tx_start_slot = 0;
  Status txas = tx_credits_->AcquireSlots(n_slots, config_.engine.control_stage_acquire_timeout_ms, &tx_start_slot);
  if (!txas.OK()) {
    KLLM_LOG_ERROR << "SubmitPrefillRequest: write-side AcquireSlots(n=" << n_slots << ") failed: " << txas.GetMessage()
                   << ", aborting";
    return txas;
  }
  void* tx_ptr = tx_pool_->SlotPtr(tx_start_slot);
  std::memcpy(tx_ptr, stage_buf.data(), stage_buf.size());

  uint64_t batch_id = 0;
  Status ws = tent_->SubmitStageWriteAsync(peer_server_name, target_offset, tx_ptr, stage_buf.size(), &batch_id);
  if (!ws.OK()) {
    tx_credits_->ReleaseSlots(tx_start_slot, n_slots);
    KLLM_LOG_ERROR << "SubmitPrefillRequest: SubmitStageWriteAsync failed: " << ws.GetMessage() << ", aborting";
    return ws;
  }

  // Block until the WRITE drains. The control-plane stage write is
  // small enough (~tens of KB to a few MB for 32K-token prompts) that
  // a short busy-poll is acceptable; we don't want to send the notify
  // before Prefill can read the data. Cap the wait at the configured
  // acquire timeout (it's the same time budget for the whole stage
  // handshake). Free the batch handle either way.
  //
  // Poll with graduated backoff: small payloads drain within a few µs,
  // so a fixed 200µs interval would add ~100µs avg latency straight onto
  // TTFT for the common case. Start at 2µs and double up to a 200µs cap
  // so fast WRITEs return almost immediately while slow/large transfers
  // still settle into a cheap long-interval poll.
  constexpr int64_t kMinPollUs = 2;
  constexpr int64_t kMaxPollUs = 200;
  const int64_t wait_budget_ms = config_.engine.control_stage_acquire_timeout_ms;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_budget_ms < 0 ? 60'000 : wait_budget_ms);
  bool wrote = false;
  int64_t poll_us = kMinPollUs;
  while (true) {
    bool done = false, failed = false;
    Status gs = tent_->GetBatchStatus(batch_id, &done, &failed);
    if (!gs.OK() || failed) {
      KLLM_LOG_ERROR << "SubmitPrefillRequest: stage write batch " << batch_id << " failed: " << gs.GetMessage();
      break;
    }
    if (done) {
      wrote = true;
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      KLLM_LOG_ERROR << "SubmitPrefillRequest: stage write batch " << batch_id << " timed out after " << wait_budget_ms
                     << "ms";
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(poll_us));
    poll_us = std::min(poll_us * 2, kMaxPollUs);
  }
  tent_->FreeBatchID(batch_id);
  // The local write-side slot range is no longer needed once the WRITE is drained
  // (success) or has terminally failed — Mooncake won't read from the
  // source after that. Release before erroring out / returning.
  tx_credits_->ReleaseSlots(tx_start_slot, n_slots);
  if (!wrote) {
    return Status(RET_RUNTIME_FAILED, fmt::format("stage write to {} failed", peer_server_name));
  }
  return Status();
}

void MooncakeEngineConnector::HandleSendFailure(uint64_t req_id, const std::string& peer_server_name,
                                                std::shared_ptr<PeerCredits> credits, uint32_t start_slot,
                                                uint32_t n_slots) {
  // S2 reactive removal: send-side failure → drop the peer immediately
  // so PeerSelector::Select() stops handing it out. Next PrefillSnapshot
  // from this Prefill (after it recovers / restarts) re-registers it
  // via OnPrefillSnapshot's discovery path. Look up the segment_id by
  // server_name (the in_flight_ entry doesn't track it; peer_server_name_by_seg_
  // is the reverse map). Also release the just-allocated slots so they
  // aren't pinned forever.
  credits->ReleaseSlots(start_slot, n_slots);
  std::optional<LinUCBVec> remote_feat;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    stage_alloc_by_req_.erase(req_id);
    if (peer_selector_ != nullptr) {
      uint64_t stale_seg_id = 0;
      for (const auto& [seg, name] : peer_server_name_by_seg_) {
        if (name == peer_server_name) {
          stale_seg_id = seg;
          break;
        }
      }
      if (stale_seg_id != 0) {
        peer_selector_->UnregisterPeer(stale_seg_id);
        peer_server_name_by_seg_.erase(stale_seg_id);
        KLLM_LOG_WARNING << "MooncakeEngineConnector::SubmitPrefillRequest send fail → "
                         << "removed peer " << peer_server_name << " (seg_id=" << stale_seg_id << ")";
      }
    }
    remote_feat = EraseInFlightLocked(req_id);
  }
  // Notify never reached the peer → OnPrefillComplete will never settle this
  // REMOTE arm. Penalize it with a zero reward outside state_mu_ (RecordReward
  // takes the router's own mutex and must never nest under it).
  if (remote_feat.has_value()) {
    linucb_->RecordReward(RouteArm::kRemotePeer, *remote_feat, 0.0);
  }
}

Status MooncakeEngineConnector::SubmitPrefillRequest(uint64_t req_id, const std::vector<int>& input_tokens,
                                                     const std::vector<int>& kv_pool_block_ids,
                                                     const SamplingConfigWire& sampling_config,
                                                     uint32_t attn_dp_group_id, uint32_t cached_block_count,
                                                     uint32_t decode_prefix_cache_offset,
                                                     PdRequestCompleteCb on_complete, void* ctx) {
  // Phase 1 (on the EDS scheduling thread, fast + non-blocking): admission
  // lookup + register the completion callback. The heavy, potentially
  // blocking pipeline (build wire frame → acquire stage slots → RDMA-WRITE
  // → notify) is deferred to a submit worker so a stuck / full / slow peer
  // can never stall Decode scheduling (head-of-line blocking of the whole
  // engine). On async failure the worker fires on_complete(kFailed), which
  // the Decode hook unwinds exactly like a failed peer round-trip.
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    auto it = in_flight_.find(req_id);
    if (it == in_flight_.end()) {
      // Not in in_flight_ means the req was never admitted via the REMOTE path
      // (CanAccept / RouteRequest→kRemotePeer). The most likely caller bug is
      // invoking SubmitPrefillRequest after RouteRequest returned kLocal — that
      // path keeps prefill local and intentionally skips admission, so it must
      // NOT reach here. Other possibilities: admission was declined (no healthy
      // peer), or the req already completed/was released.
      return Status(RET_INVALID_ARGUMENT,
                    fmt::format("SubmitPrefillRequest: req {} not in in_flight_ (not admitted via kRemotePeer; "
                                "kLocal-routed reqs must not call SubmitPrefillRequest, or it was already "
                                "completed/released)",
                                req_id));
    }
    it->second.submitted = true;
    // 注册回调, 在PrefillComplete 中被调用 (也用于异步 submit 失败回调)
    it->second.on_complete = on_complete;
    it->second.on_complete_ctx = ctx;
    it->second.submit_time_ms = static_cast<int64_t>(NowMs());
  }

  // Enqueue the payload (owning copies — the worker outlives this frame)
  // and return immediately. The worker resolves the peer from in_flight_
  // when it runs, so a peer reroute in the fallback path is transparent.
  PendingSubmit task;
  task.req_id = req_id;
  task.input_tokens = input_tokens;
  task.kv_pool_block_ids = kv_pool_block_ids;
  task.sampling_config = sampling_config;
  task.attn_dp_group_id = attn_dp_group_id;
  task.cached_block_count = cached_block_count;
  task.decode_prefix_cache_offset = decode_prefix_cache_offset;
  {
    std::lock_guard<std::mutex> lk(submit_mu_);
    submit_queue_.push_back(std::move(task));
  }
  submit_cv_.notify_one();
  return Status();
}

void MooncakeEngineConnector::SubmitWorkerLoop() {
  while (true) {
    PendingSubmit task;
    {
      std::unique_lock<std::mutex> lk(submit_mu_);
      submit_cv_.wait(lk, [this] { return submit_stop_.load() || !submit_queue_.empty(); });
      if (submit_stop_.load() && submit_queue_.empty()) return;
      task = std::move(submit_queue_.front());
      submit_queue_.pop_front();
    }
    ProcessSubmit(task);
  }
}

void MooncakeEngineConnector::FailSubmit(uint64_t req_id, const Status& why) {
  PdRequestCompleteCb cb = nullptr;
  void* ctx = nullptr;
  std::string peer;
  std::optional<LinUCBVec> remote_feat;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    auto it = in_flight_.find(req_id);
    if (it == in_flight_.end()) {
      // Already erased by a racing Release (req destroyed mid-flight). The
      // hook's OnDestroy path owns cleanup; nothing to fire here.
      return;
    }
    cb = it->second.on_complete;
    ctx = it->second.on_complete_ctx;
    peer = it->second.peer_server_name;
    remote_feat = EraseInFlightLocked(req_id);
  }
  KLLM_LOG_ERROR << "MooncakeEngineConnector::SubmitPrefillRequest(async) req=" << req_id
                 << " failed: " << why.GetMessage() << " → on_complete(kFailed)";
  // Penalize the kLinUCB REMOTE arm outside state_mu_ (RecordReward takes
  // the router's own mutex and must never nest under state_mu_).
  if (remote_feat.has_value() && linucb_ != nullptr) {
    linucb_->RecordReward(RouteArm::kRemotePeer, *remote_feat, 0.0);
  }
  if (cb != nullptr) {
    cb(req_id, PdCompleteStatus::kFailed, /*generated_tokens=*/{}, /*prefill_metrics=*/{}, peer, ctx);
  }
}

void MooncakeEngineConnector::ProcessSubmit(const PendingSubmit& task) {
  const uint64_t req_id = task.req_id;

  // Re-check in_flight_ (a Release may have fired between enqueue and now)
  // and resolve the primary peer chosen at admission.
  std::string primary_peer;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    auto it = in_flight_.find(req_id);
    if (it == in_flight_.end()) {
      KLLM_LOG_INFO << "ProcessSubmit: req " << req_id << " released before dispatch, dropping";
      return;
    }
    primary_peer = it->second.peer_server_name;
  }

  // Phase 2: build the lightweight PrefillRequest wire frame (the stage
  // slot reference is stamped after the write drains).
  PrefillRequest req;
  Status bs = BuildPrefillRequest(req_id, task.sampling_config, task.attn_dp_group_id, task.cached_block_count,
                                  task.decode_prefix_cache_offset, &req);
  if (!bs.OK()) {
    FailSubmit(req_id, bs);
    return;
  }

  // Phase 3: serialize the heavy input_tokens into a PrefillStagePayload
  // for RDMA-WRITE into the Prefill peer's HostStagePool. The lightweight
  // notify only carries the slot reference.
  PrefillStagePayload stage_payload;
  stage_payload.input_tokens = task.input_tokens;
  stage_payload.kv_block_ids = task.kv_pool_block_ids;
  std::vector<uint8_t> stage_buf;
  SerializePrefillStagePayload(stage_payload, &stage_buf);

  // Phase 4: reserve the peer read-side slot range with peer fallback (try primary
  // non-blocking → try other healthy peers non-blocking → block on
  // primary). On any later failure we own releasing it.
  std::string used_peer;
  std::shared_ptr<PeerCredits> credits;
  uint32_t start_slot = 0;
  uint32_t n_slots = 0;
  uint64_t target_offset = 0;
  Status ps = AcquirePeerStageSlotsWithFallback(primary_peer, stage_buf.size(), &used_peer, &credits, &start_slot,
                                                &n_slots, &target_offset);
  if (!ps.OK()) {
    FailSubmit(req_id, ps);
    return;
  }

  // If the fallback rerouted to a different peer, update in_flight_ so the
  // completion path (OnPrefillComplete latency/reward, Release) attributes
  // to the peer that actually served the request. If the req was released
  // meanwhile, roll back the slots and stop.
  if (used_peer != primary_peer) {
    std::lock_guard<std::mutex> lk(state_mu_);
    auto it = in_flight_.find(req_id);
    if (it == in_flight_.end()) {
      credits->ReleaseSlots(start_slot, n_slots);
      KLLM_LOG_INFO << "ProcessSubmit: req " << req_id << " released during acquire, rolled back slots";
      return;
    }
    it->second.peer_server_name = used_peer;
  }

  // Phase 5: RDMA-WRITE the staged payload and block until it drains.
  // StageWriteAndWait fully owns the local write-side slot lifetime; we still own the
  // peer read-side slots acquired above.
  Status ws = StageWriteAndWait(used_peer, target_offset, stage_buf, n_slots);
  if (!ws.OK()) {
    credits->ReleaseSlots(start_slot, n_slots);
    FailSubmit(req_id, ws);
    return;
  }

  // Phase 6: stamp the slot reference into the notify body and remember
  // the allocation so OnStageAck can release the partition slots even if
  // PrefillRequestComplete has already evicted in_flight_. Re-check
  // in_flight_ under the lock: if Release ran during the stage write, roll
  // back the slots and stop (don't send a notify for a dead request).
  req.stage_start_slot = start_slot;
  req.stage_n_slots = n_slots;
  req.stage_payload_bytes = static_cast<uint32_t>(stage_buf.size());
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    if (in_flight_.find(req_id) == in_flight_.end()) {
      credits->ReleaseSlots(start_slot, n_slots);
      KLLM_LOG_INFO << "ProcessSubmit: req " << req_id << " released during stage write, rolled back slots";
      return;
    }
    stage_alloc_by_req_[req_id] = {used_peer, start_slot, n_slots};
  }

  // Phase 7: serialize + send the notify; on send failure evict the peer.
  std::vector<uint8_t> body;
  SerializePrefillRequest(req, &body);
  Notification notify{kChPrefillReq, std::string(body.begin(), body.end())};
  // PD 链路关键点[2/4]: decode 实际把 PrefillReq 通知发往 peer Prefill (保留 INFO 便于阻塞定位)。
  KLLM_LOG_INFO << "MooncakeEngineConnector::ProcessSubmit req=" << req.req_id << " peer=" << used_peer
                << " stage_bytes=" << stage_buf.size() << " stage_start_slot=" << start_slot
                << " stage_n_slots=" << n_slots << " notify_body=" << body.size();
  Status s = tent_->SendNotification(used_peer, notify);
  if (!s.OK()) {
    // Capture the callback before HandleSendFailure erases in_flight_. In
    // the old synchronous design the send failure was surfaced via the
    // return value and the Decode hook drove kFailAndEnd itself; now the
    // hook is parked in kWaitAsyncResult, so we must fire on_complete here
    // to unwind it. HandleSendFailure already records the kLinUCB penalty
    // and evicts the peer, so we only deliver the terminal status.
    PdRequestCompleteCb cb = nullptr;
    void* ctx = nullptr;
    {
      std::lock_guard<std::mutex> lk(state_mu_);
      auto it = in_flight_.find(req_id);
      if (it != in_flight_.end()) {
        cb = it->second.on_complete;
        ctx = it->second.on_complete_ctx;
      }
    }
    HandleSendFailure(req_id, used_peer, credits, start_slot, n_slots);
    if (cb != nullptr) {
      cb(req_id, PdCompleteStatus::kFailed, /*generated_tokens=*/{}, /*prefill_metrics=*/{}, used_peer, ctx);
    }
  }
}

Status MooncakeEngineConnector::Release(uint64_t req_id) {
  // Local cleanup only: erase in_flight_ bookkeeping and force-release any
  // stage credits this req still holds (normally freed by OnStageAck, but
  // that may never arrive on peer crash). Prefill is not notified — there
  // is no "cancel" concept on the PD wire: cancel/timeout while
  // pd_v2_awaiting_prefill is handled on Decode by deferring Stop until
  // OnPrefillComplete. Idempotent for unknown req_id. On the Prefill /
  // executor / probe paths stage_alloc_by_req_ is empty so the stage
  // release is a harmless no-op.
  bool had_stage_alloc = false;
  StageAlloc stage_alloc;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    in_flight_.erase(req_id);

    auto sit = stage_alloc_by_req_.find(req_id);
    if (sit != stage_alloc_by_req_.end()) {
      had_stage_alloc = true;
      stage_alloc = sit->second;
      stage_alloc_by_req_.erase(sit);
    }
  }
  if (had_stage_alloc) {
    std::lock_guard<std::mutex> sg(stage_peers_mu_);
    auto pit = stage_peers_.find(stage_alloc.peer_server_name);
    if (pit != stage_peers_.end() && pit->second.credits) {
      pit->second.credits->ReleaseSlots(stage_alloc.start_slot, stage_alloc.n_slots);
    }
  }
  return Status();
}

bool MooncakeEngineConnector::HasHealthyPrefillPeer() const {
  // Prefill-role connectors never build a PeerSelector; report false so the
  // caller's /health does not gate on a peer table that doesn't exist.
  if (peer_selector_ == nullptr) {
    return false;
  }
  return peer_selector_->HealthyPeerCount() > 0;
}

std::optional<uint32_t> MooncakeEngineConnector::GetPeerLatencyUs(const std::string& server_name) const {
  std::lock_guard<std::mutex> lk(peer_latency_mu_);
  auto it = peer_latency_ewma_.find(server_name);
  if (it == peer_latency_ewma_.end()) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(it->second);
}

void MooncakeEngineConnector::OnPrefillComplete(const Notification& n) {
  PrefillRequestComplete msg;
  if (DeserializePrefillRequestComplete(reinterpret_cast<const uint8_t*>(n.msg.data()), n.msg.size(), &msg) == 0) {
    KLLM_LOG_WARNING << "OnPrefillComplete: malformed, dropped (msg.size=" << n.msg.size() << ")";
    return;
  }
  // PD 链路关键点[4/4]: decode 收到 Prefill 回发的完成通知 (保留 INFO 便于阻塞定位)。
  KLLM_LOG_INFO << "MooncakeEngineConnector::OnPrefillComplete req_id=" << msg.req_id
                << " status=" << static_cast<int>(msg.status) << " tokens=" << msg.generated_tokens.size();
  PdRequestCompleteCb cb = nullptr;
  void* ctx = nullptr;
  std::string peer_server_name;
  int64_t submit_time_ms = 0;
  LinUCBVec router_feat{};
  bool has_router_feat = false;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    if (peer_selector_ != nullptr && msg.snapshot.segment_id != 0) {
      peer_selector_->UpdateSnapshot(msg.snapshot.segment_id, msg.snapshot);
    }
    auto it = in_flight_.find(msg.req_id);
    if (it == in_flight_.end()) {
      KLLM_LOG_WARNING << "OnPrefillComplete: unknown req_id " << msg.req_id << " (in_flight size=" << in_flight_.size()
                       << ")";
      return;
    }
    cb = it->second.on_complete;
    ctx = it->second.on_complete_ctx;
    peer_server_name = it->second.peer_server_name;
    submit_time_ms = it->second.submit_time_ms;
    router_feat = it->second.router_arm_features;
    has_router_feat = it->second.has_router_features;
    in_flight_.erase(it);
  }

  // Update the per-peer round-trip latency EWMA outside state_mu_ (own
  // lock, no nesting). Only successful completions with a recorded submit
  // time carry a meaningful TTFT signal.
  if (submit_time_ms != 0 && !peer_server_name.empty()) {
    const int64_t elapsed_ms = static_cast<int64_t>(NowMs()) - submit_time_ms;
    const double latency_us = std::max<int64_t>(0, elapsed_ms) * 1000.0;
    constexpr double kAlpha = 0.2;
    std::lock_guard<std::mutex> lk(peer_latency_mu_);
    auto& ewma = peer_latency_ewma_[peer_server_name];
    ewma = ewma == 0.0 ? latency_us : kAlpha * latency_us + (1.0 - kAlpha) * ewma;
    // Aggregate (peer-agnostic) remote round-trip EWMA — the TTFT proxy the
    // kLocal arm scores against in RouteRequest (it has no peer of its own).
    remote_rtt_ewma_us_ =
        remote_rtt_ewma_us_ == 0.0 ? latency_us : kAlpha * latency_us + (1.0 - kAlpha) * remote_rtt_ewma_us_;
  }

  // Reward the LinUCB REMOTE arm (selector_policy == kLinUCB). Faster
  // Submit->Complete round-trip => higher reward; failed/cancelled => 0.
  // RecordReward takes the router's own mutex — never nested under state_mu_.
  if (linucb_ != nullptr && has_router_feat) {
    double reward = 0.0;
    if (msg.status == PdCompleteStatus::kOk && submit_time_ms != 0) {
      const double elapsed_us = std::max<int64_t>(0, static_cast<int64_t>(NowMs()) - submit_time_ms) * 1000.0;
      reward = linucb_->RemoteReward(elapsed_us);
    }
    linucb_->RecordReward(RouteArm::kRemotePeer, router_feat, reward);
  }
  if (cb != nullptr) {
    KLLM_LOG_DEBUG << "OnPrefillComplete: invoking decode hook cb for req_id=" << msg.req_id;
    PdPrefillMetrics prefill_metrics;
    prefill_metrics.prefix_cache_hit_tokens = msg.pf_prefix_cache_hit_tokens;
    prefill_metrics.queue_us = msg.pf_queue_us;
    prefill_metrics.compute_us = msg.pf_compute_us;
    prefill_metrics.send_us = msg.pf_send_us;
    cb(msg.req_id, msg.status, msg.generated_tokens, prefill_metrics, peer_server_name, ctx);
  } else {
    KLLM_LOG_WARNING << "OnPrefillComplete: cb=null for req_id=" << msg.req_id;
  }
}

void MooncakeEngineConnector::OnStageAck(const Notification& n) {
  StageAck ack;
  if (DeserializeStageAck(reinterpret_cast<const uint8_t*>(n.msg.data()), n.msg.size(), &ack) == 0) {
    KLLM_LOG_WARNING << "OnStageAck: malformed, dropped (msg.size=" << n.msg.size() << ")";
    return;
  }
  std::string peer;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    auto it = stage_alloc_by_req_.find(ack.req_id);
    if (it == stage_alloc_by_req_.end()) {
      // Already force-released by Release(), or never allocated (race
      // with PrefillRequestComplete that arrived before StageAck).
      KLLM_LOG_DEBUG << "OnStageAck: unknown req_id=" << ack.req_id << " (already released?)";
      return;
    }
    peer = it->second.peer_server_name;
    stage_alloc_by_req_.erase(it);
  }
  std::lock_guard<std::mutex> sg(stage_peers_mu_);
  auto it = stage_peers_.find(peer);
  if (it != stage_peers_.end() && it->second.credits) {
    it->second.credits->ReleaseSlots(ack.start_slot, ack.n_slots);
  }
  if (ack.err_code != 0) {
    KLLM_LOG_WARNING << "OnStageAck: peer=" << peer << " req_id=" << ack.req_id << " err_code=" << ack.err_code
                     << " (slot range released; failure handled via PrefillRequestComplete)";
  }
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
