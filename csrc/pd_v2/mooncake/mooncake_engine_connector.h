/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "pd_v2/decode/linucb_router.h"
#include "pd_v2/decode/peer_selector.h"
#include "pd_v2/mooncake/engine_tent_wrapper.h"
#include "pd_v2/mooncake/host_stage_pool.h"
#include "pd_v2/mooncake/peer_credits.h"
#include "pd_v2/mooncake/redis_alive_store.h"
#include "pd_v2/public/pd_v2_connector.h"
#include "pd_v2/wire/prefill_request_wire.h"

namespace mooncake {
class TransferEngine;
}  // namespace mooncake

namespace ksana_llm {

class IpcFuncWrapper;

namespace pd_v2 {
namespace mooncake_backend {

// PdV2ConnectorInterface implementation for the engine-deployed
// connector. One instance per ksana_engine process.
//
// Responsibilities:
//   - Boot: WaitForKvBufferVas drains the same-node SHM IPC populated
//     by per-rank ksana_executor processes
//   - Per-request Decode side: CanAccept (PeerSelector pick) →
//     SubmitPrefillRequest (build PrefillRequest wire frame, send via
//     EngineTentWrapper notify) → on_complete fires from the
//     notify-poller thread when peer Prefill engine answers
//   - Per-request Prefill side: ReceiveNotifications dispatches
//     incoming PrefillRequest to RegisterIncomingHook, SendPrefillComplete
//     packs reply
//
// Executor-side Boot methods (RegisterKvPool / PublishKvBufferVa) and
// Executor-side KV-write methods live on MooncakeExecutorConnector; calling them here
// returns RET_INVALID_ARGUMENT.
class MooncakeEngineConnector : public PdV2ConnectorInterface {
 public:
  MooncakeEngineConnector();
  ~MooncakeEngineConnector() override;

  Status Initialize(const PdV2ConnectorConfig& config) override;
  Status Shutdown() override;

  // Bind the engine-process IpcFuncWrapper used to round-trip
  // GetKvBufferVa to the per-rank ksana_executor processes. The engine
  // bootstrap calls this exactly once after the connector is constructed
  // and the wrapper is built. Raw pointer (no ownership).
  void SetIpcFuncWrapper(IpcFuncWrapper* ipc_func_wrapper) override;

  // [Prefill role] Inject the local-load PeerHealth provider consumed by
  // BroadcastSnapshot. Set once by the engine bootstrap after construction.
  void SetSnapshotProvider(std::function<PeerHealth()> provider) override;

  // ===== Boot-time =====
  //
  // RegisterKvPool is invalid on engine connector (returns
  // RET_INVALID_ARGUMENT). GetKvBufferVa runs an IpcFuncWrapper RPC
  // to the executor process for `rank`, caches the result in
  // rank_kv_va_cache_, and returns. First call per rank pays the
  // round-trip; subsequent calls are cache hits.

  Status RegisterKvPool(void* /*kv_pool_base*/, size_t /*total_bytes*/, size_t /*block_size*/, size_t /*num_blocks*/,
                        size_t /*num_layers*/) override;
  Status GetKvBufferVa(int rank, uint64_t* out_kv_pool_base_va, uint64_t* out_total_bytes) override;

  // ===== Decode side =====

  bool CanAccept(uint64_t req_id, uint32_t input_token_count) override;
  // LinUCB LOCAL-vs-REMOTE routing. selector_policy != kLinUCB (linucb_ ==
  // null) forwards to CanAccept unchanged. kLinUCB consults the bandit:
  // kLocal stashes select-time features in router_local_pending_ (no
  // in_flight_ entry), kRemotePeer reuses CanAccept + stashes features in the
  // AdmissionState for the deferred remote reward. See pd_v2_connector.h.
  std::optional<RouteArm> RouteRequest(uint64_t req_id, const LocalArmContext& ctx) override;
  Status SubmitPrefillRequest(uint64_t req_id, const std::vector<int>& input_tokens,
                              const std::vector<int>& kv_pool_block_ids, const SamplingConfigWire& sampling_config,
                              uint32_t attn_dp_group_id, uint32_t cached_block_count,
                              uint32_t decode_prefix_cache_offset, PdRequestCompleteCb on_complete, void* ctx) override;
  Status Release(uint64_t req_id) override;

  // Health: whether at least one Prefill peer is currently healthy. False
  // when this connector has no PeerSelector (Prefill role). See interface.
  bool HasHealthyPrefillPeer() const override;

  // Decode-side per-peer round-trip latency EWMA (microseconds), keyed by
  // peer Prefill server_name. Measured locally as SubmitPrefillRequest →
  // OnPrefillComplete. Returns nullopt when the peer has no sample yet.
  // No consumer in this step; reserved as a future LinUCB feature / reward.
  std::optional<uint32_t> GetPeerLatencyUs(const std::string& server_name) const;

  // ===== Prefill side =====

  void RegisterIncomingHook(PdIncomingReqCb on_incoming, void* ctx) override;
  Status SendPrefillComplete(const std::string& decode_engine_inference_addr, uint64_t req_id, PdCompleteStatus status,
                             const std::vector<int>& generated_tokens, const PeerHealth& snapshot,
                             const PdPrefillMetrics& prefill_metrics) override;

 private:
  // Per-request admission + on_complete tracking on Decode side.
  struct AdmissionState {
    std::string peer_server_name;
    PdRequestCompleteCb on_complete = nullptr;
    void* on_complete_ctx = nullptr;
    bool submitted = false;
    // Wall-clock (ms) when SubmitPrefillRequest registered the callback;
    // used at OnPrefillComplete to derive the round-trip latency feeding
    // peer_latency_ewma_. 0 = not yet submitted.
    int64_t submit_time_ms = 0;
    // LinUCB select-time feature vector (selector_policy == kLinUCB only).
    // Stashed by RouteRequest's kRemotePeer branch so OnPrefillComplete can
    // reward the REMOTE arm with the matching context. Unused otherwise.
    LinUCBVec router_arm_features{};
    bool has_router_features = false;
  };

  // ===== SubmitPrefillRequest helpers (Decode side) =====
  //
  // SubmitPrefillRequest is a multi-phase pipeline (build wire frame →
  // acquire peer stage slots → RDMA-WRITE the staged payload → notify);
  // each phase has its own rollback. These helpers carve the phases out
  // so the top-level method reads as the happy-path sequence and each
  // helper owns the cleanup for the resources it acquires.

  // Erase the in_flight_ bookkeeping for req_id under state_mu_. The
  // error paths below call this after releasing any slots they hold.
  // When the erased entry carried a stashed REMOTE-arm context
  // (has_router_features, selector_policy == kLinUCB), this is a
  // control-plane abort on the REMOTE path: the peer is never notified, so
  // OnPrefillComplete will never fire to settle the arm. Settle a zero
  // reward here instead so the REMOTE arm is penalized for the failed
  // dispatch (otherwise only successful round-trips feed the bandit, biasing
  // it optimistically toward REMOTE exactly when the control plane is
  // unstable). RecordReward is taken outside state_mu_ — never nested.
  void EraseInFlight(uint64_t req_id);

  // Erase the in_flight_ entry for req_id; MUST be called with state_mu_
  // held. If the entry carried a stashed REMOTE-arm feature vector
  // (kLinUCB), returns it so the caller can RecordReward(kRemotePeer, .,
  // 0.0) AFTER releasing state_mu_ (RecordReward takes the router's own
  // mutex and must never nest under state_mu_). Returns nullopt when the
  // req is unknown or has no router features.
  std::optional<LinUCBVec> EraseInFlightLocked(uint64_t req_id);

  // Build the lightweight PrefillRequest wire frame (everything except
  // the stage slot reference, which is stamped after the stage write
  // drains). Fills req->decode_targets by querying each local Decode
  // attn-TP rank's KV buffer VA. Acquires/releases nothing; returns
  // non-OK (the GetKvBufferVa error) if any rank lookup fails.
  Status BuildPrefillRequest(uint64_t req_id, const SamplingConfigWire& sampling_config, uint32_t attn_dp_group_id,
                             uint32_t cached_block_count, uint32_t decode_prefix_cache_offset, PrefillRequest* req);

  // Reserve a stage slot range on the peer's read-side PeerCredits ring sized
  // for stage_bytes and compute the absolute remote target_offset for
  // the WRITE. Blocks at most `timeout_ms` in AcquireSlots (0 =
  // non-blocking try, negative = wait forever). On success *out_credits
  // owns the [*out_start_slot, *out_start_slot + *out_n_slots) range —
  // the caller must ReleaseSlots it on any later failure. On failure
  // nothing is acquired (returns the reason: no stage assignment yet,
  // ring exhausted, or acquire timed out).
  Status AcquirePeerStageSlots(const std::string& peer_server_name, size_t stage_bytes, int64_t timeout_ms,
                               std::shared_ptr<PeerCredits>* out_credits, uint32_t* out_start_slot,
                               uint32_t* out_n_slots, uint64_t* out_target_offset);

  // Fallback wrapper around AcquirePeerStageSlots run on a submit worker:
  //   1. non-blocking try-acquire on the primary peer (chosen at admission);
  //   2. if full, non-blocking try-acquire across every other healthy peer
  //      (PeerSelector::HealthyPeers) so a single full peer doesn't stall
  //      the request while others have room;
  //   3. only if all peers are momentarily full, block up to
  //      control_stage_acquire_timeout_ms on the primary peer (backpressure).
  // *out_used_peer is the peer the slots were actually reserved on (may
  // differ from primary_peer after step 2); the caller must send the
  // notify to and record the stage alloc against that peer.
  Status AcquirePeerStageSlotsWithFallback(const std::string& primary_peer, size_t stage_bytes,
                                           std::string* out_used_peer, std::shared_ptr<PeerCredits>* out_credits,
                                           uint32_t* out_start_slot, uint32_t* out_n_slots,
                                           uint64_t* out_target_offset);

  // Copy the serialized stage payload into a local registered write-side slot
  // run, RDMA-WRITE it to the peer's read-side slot range at target_offset, and
  // block until the WRITE drains (or the configured budget elapses).
  // Fully owns the local write slot lifetime: these slots are always released by
  // the time this returns, success or failure. The caller still owns the
  // peer read-side slots. n_slots is the slot count from AcquirePeerStageSlots
  // and is reused for the local write-side scratch range.
  Status StageWriteAndWait(const std::string& peer_server_name, uint64_t target_offset,
                           const std::vector<uint8_t>& stage_buf, uint32_t n_slots);

  // Send-side failure cleanup for the notify (S2 reactive removal):
  // release the peer read-side slots, drop the stage alloc + in_flight_ entry,
  // and evict the peer from PeerSelector so it stops being handed out
  // until the next PrefillSnapshot re-registers it.
  void HandleSendFailure(uint64_t req_id, const std::string& peer_server_name, std::shared_ptr<PeerCredits> credits,
                         uint32_t start_slot, uint32_t n_slots);

  // ===== Async SubmitPrefillRequest offload (Decode side) =====
  //
  // SubmitPrefillRequest used to run its whole pipeline synchronously on
  // the caller (the EDS scheduling thread, under infer_request_mutex_).
  // The acquire-slots and stage-write-drain phases can block for up to
  // control_stage_acquire_timeout_ms when a peer's stage pool is full or
  // an RDMA WRITE stalls — freezing ALL Decode scheduling, not just the
  // one request (head-of-line blocking). To isolate that, the scheduling
  // thread now only registers the callback + enqueues a PendingSubmit and
  // returns immediately; a pool of worker threads runs the blocking
  // pipeline. Failures surface via the same on_complete callback the peer
  // completion path uses (PdCompleteStatus::kFailed), so the Decode hook
  // unwinds a failed async submit exactly like a failed peer round-trip.

  // One queued SubmitPrefillRequest. Carries owning copies of the payload
  // vectors because the worker outlives the caller's stack frame.
  struct PendingSubmit {
    uint64_t req_id = 0;
    std::vector<int> input_tokens;
    std::vector<int> kv_pool_block_ids;
    SamplingConfigWire sampling_config;
    uint32_t attn_dp_group_id = 0;
    uint32_t cached_block_count = 0;
    uint32_t decode_prefix_cache_offset = 0;
  };

  // Worker loop: pops PendingSubmit tasks and runs ProcessSubmit. Started
  // in Initialize (Decode role), joined in Shutdown.
  void SubmitWorkerLoop();

  // Run phases 2-7 of the submit pipeline for one queued request (build
  // wire frame → serialize → acquire stage slots with peer fallback →
  // RDMA-WRITE → stamp + record → notify). On any failure fires the
  // request's on_complete with kFailed. Re-checks in_flight_ across the
  // blocking phases so a concurrent Release (req destroyed mid-flight)
  // is handled by rolling back the just-acquired slots.
  void ProcessSubmit(const PendingSubmit& task);

  // Fail an in-flight async submit: snapshot + erase the in_flight_ entry
  // (penalizing the kLinUCB REMOTE arm), then fire on_complete(kFailed)
  // outside state_mu_ so the Decode hook ends the request. No-op if the
  // req was already erased (e.g. by a racing Release).
  void FailSubmit(uint64_t req_id, const Status& why);

  // Notify-poller thread loop. Drains EngineTentWrapper, dispatches by
  // channel name to OnPrefillReq / OnPrefillComplete / OnPrefillSnapshot.
  // Started in Initialize, joined in Shutdown.
  void NotifyPollerLoop();
  void DispatchOne(const Notification& n);

  void OnPrefillReq(const Notification& n);
  void OnPrefillComplete(const Notification& n);
  void OnPrefillSnapshot(const Notification& n);
  // Decode-side: Prefill peer signals that the stage slot range for a
  // given req_id has been consumed and the slots can be returned to
  // PeerCredits. err_code != 0 means Prefill couldn't parse the stage
  // payload — Decode still releases slots so the bitmap doesn't leak,
  // but downstream PrefillRequestComplete{kFailed} handles the
  // request-level failure.
  void OnStageAck(const Notification& n);

  PdV2ConnectorConfig config_;

  std::unique_ptr<EngineTentWrapper> tent_;
  std::unique_ptr<PeerSelector> peer_selector_;
  // LinUCB LOCAL-vs-REMOTE router. Non-null only when
  // config_.engine.selector_policy == kLinUCB; constructed in Initialize.
  // Owned here (sibling of peer_selector_) because all peer/in-flight state
  // lives on the connector. Has its own internal mutex.
  std::unique_ptr<LinUCBRouter> linucb_;
  IpcFuncWrapper* ipc_func_wrapper_ = nullptr;

  // SegmentID-keyed cache of peer Mooncake server_name. Populated by
  // OnPrefillSnapshot's discovery path (first PrefillSnapshot from a
  // never-before-seen Prefill triggers an openSegment + RegisterPeer +
  // cache insert). Used by CanAccept to translate PeerSelector::Select()
  // (returns SegmentID) → server_name for SendNotification.
  std::unordered_map<uint64_t, std::string> peer_server_name_by_seg_;

  // Engine-side cache: rank → (kv_pool_base_va, total_bytes).
  // Populated lazily by GetKvBufferVa via IpcFuncWrapper RPC.
  struct KvVaCacheEntry {
    uint64_t kv_pool_base_va = 0;
    uint64_t total_bytes = 0;
  };
  mutable std::mutex kv_va_cache_mu_;
  std::unordered_map<int, KvVaCacheEntry> rank_kv_va_cache_;

  // Notify-dispatch thread.
  std::thread poller_;
  std::atomic<bool> poller_stop_{false};

  // Decode role only: async SubmitPrefillRequest worker pool + queue.
  // The scheduling thread enqueues under submit_mu_ and signals
  // submit_cv_; workers pop and run ProcessSubmit. Sized by
  // EngineConfig.control_stage_async_workers (>= 1). submit_stop_ +
  // notify_all wakes idle workers at Shutdown; PeerCredits::Shutdown
  // (called before the join in Shutdown) unblocks any worker parked in a
  // blocking AcquireSlots so the join can't hang on a stuck peer.
  std::vector<std::thread> submit_workers_;
  mutable std::mutex submit_mu_;
  std::condition_variable submit_cv_;
  std::deque<PendingSubmit> submit_queue_;
  std::atomic<bool> submit_stop_{false};

  // Heartbeat thread (Prefill role only). Wakes every
  // EngineConfig.snapshot_push_interval_ms, calls BroadcastSnapshot
  // which drives §3.3 service discovery: KEYS pd_v2/decode_alive/<cluster>/* on
  // Redis to find live Decodes, manages decode_epoch_cache_ (close+open
  // .exec.rN segments on epoch mismatch), pushes a PrefillSnapshot to
  // each. Falls back to incoming_routes_ iteration when metadata_addr
  // is not redis:// (P2PHANDSHAKE single-machine dev).
  std::thread heartbeat_;
  std::atomic<bool> heartbeat_stop_{false};
  void HeartbeatLoop();
  void BroadcastSnapshot();

  // Decode role only: Redis keepalive thread. Boot DEL+SET stale,
  // 2s SET renew loop with `{"epoch":<boot_ms>}` EX 5. On Shutdown,
  // DEL the alive key so Prefill peers see us go immediately rather
  // than after TTL.
  std::thread keepalive_;
  std::atomic<bool> keepalive_stop_{false};
  uint64_t boot_epoch_ms_ = 0;
  void RedisKeepaliveLoop();

  // Shared hiredis client (one connection per process). Constructed by
  // Initialize when metadata_addr is redis://; null otherwise. Used by
  // both Decode keepalive thread and Prefill broadcast thread (single
  // role per process so no contention in practice).
  std::unique_ptr<RedisAliveStore> redis_alive_;

  // Lock-ordering convention for the two Decode-side maps' mutexes
  // (state_mu_ and stage_peers_mu_): if a code path ever needs both, it
  // MUST acquire them in the order state_mu_ -> stage_peers_mu_, never the
  // reverse. In practice no path nests them: Release / OnStageAck /
  // OnPrefillSnapshot all finish (and release) state_mu_ first, then take
  // stage_peers_mu_ on its own. Keep it that way — taking state_mu_ while
  // holding stage_peers_mu_ would invert the order and risk deadlock.

  // Admission state for in-flight Decode-side requests.
  mutable std::mutex state_mu_;
  std::unordered_map<uint64_t, AdmissionState> in_flight_;

  // LinUCB only: requests routed kLocal whose deferred LOCAL-arm reward has
  // not yet been applied, keyed by req_id. The reward is multiplicative:
  // ttft_credit (sampled at select time from the remote round-trip EWMA — how
  // badly a REMOTE dispatch would have blown the TTFT budget) times the
  // TPOT-headroom signal sampled one step later (next RouteRequest's ctx).
  // Guarded by state_mu_; touched only on the scheduler thread (RouteRequest),
  // never by the notify thread.
  struct LocalPending {
    LinUCBVec features{};
    double ttft_credit = 0.0;  // select-time TtftCredit(remote RTT EWMA)
  };
  std::unordered_map<uint64_t, LocalPending> router_local_pending_;

  // Apply deferred LOCAL-arm rewards for everything in router_local_pending_,
  // then clear. The reward is pending.ttft_credit * LocalReward(ctx) — the
  // multiplicative AND gate. Copies out under state_mu_ and calls RecordReward
  // (router's own mutex) outside it — no lock nesting.
  void SampleLocalReward(const LocalArmContext& ctx);

  // Decode role only: per-peer round-trip latency EWMA (microseconds),
  // keyed by peer Prefill server_name (stable across epoch-driven
  // segment_id rotation). Updated in OnPrefillComplete; read by
  // GetPeerLatencyUs. Off the wire — never enters PeerHealth.
  mutable std::mutex peer_latency_mu_;
  std::unordered_map<std::string, double> peer_latency_ewma_;
  // Aggregate (peer-agnostic) round-trip EWMA (microseconds) — overall remote
  // congestion, used as the TTFT proxy when scoring the kLocal arm (which has
  // no peer of its own). Updated alongside peer_latency_ewma_ under
  // peer_latency_mu_. 0 until the first successful remote completion.
  double remote_rtt_ewma_us_ = 0.0;

  // Prefill role only: per-Decode epoch tracking for §3.3 close/open
  // semantics. Keyed by decode_inference_addr. On Decode restart (epoch
  // mismatch) or first discovery, BroadcastSnapshot does
  // openSegment(name) + immediate closeSegment to flush the engine's
  // process-local Mooncake metadata cache (it does NOT retain the
  // handle — see the long comment in BroadcastSnapshot for why holding
  // it corrupts mooncake's notify_qp_to_endpoint_ map). Per-rank
  // executor cache invalidation is a follow-up (executor process owns
  // its own TransferEngine and cache; engine close/open here is
  // necessary but not sufficient for executor-side RDMA writes).
  std::unordered_map<std::string, uint64_t> decode_epoch_cache_;

  // Reverse-route cache for Prefill side: (decode_engine_inference_addr,
  // req_id) → decode_engine_inference_addr (technically same value;
  // kept as a pair so SendPrefillComplete can drop the entry on send).
  using IncomingKey = std::pair<std::string, uint64_t>;
  struct IncomingKeyHash {
    size_t operator()(const IncomingKey& k) const noexcept {
      return std::hash<std::string>{}(k.first) ^ (std::hash<uint64_t>{}(k.second) << 1);
    }
  };
  std::unordered_map<IncomingKey, bool, IncomingKeyHash> incoming_routes_;

  // Prefill role only: HostStagePool the Decode peers RDMA-WRITE into.
  // Allocated in Initialize when node_role == kPrefill, then registered
  // with EngineTentWrapper so the tent's RDMA transport advertises this
  // memory range. Per-peer partitions are assigned lazily in
  // BroadcastSnapshot's discovery loop (RegisterPeer is idempotent), and
  // the resulting StageAssignment is piggybacked on the unicast
  // PrefillSnapshot the same cycle. Null on Decode-role connectors.
  std::unique_ptr<HostStagePool> stage_pool_;

  // Decode role only: registered host write-side scratch the local engine uses as
  // submitTransfer source for stage WRITEs. Mooncake's TCP transport (and
  // the RDMA transport's ibv_post_send) require the source buffer to live
  // in registered memory; a heap-allocated `std::vector<uint8_t>::data()`
  // would be rejected with "Unable to find registered buffer". We allocate
  // a single flat HostStagePool (max_peers=1 → one global partition) and
  // an unpartitioned PeerCredits over its full slot range, copy the
  // serialized PrefillStagePayload into a write-side slot run before submitting,
  // and release the slots after the WRITE drains. Lifetime is independent
  // of any peer — one write-side arena serves all Prefill destinations.
  std::unique_ptr<HostStagePool> tx_pool_;
  std::unique_ptr<PeerCredits> tx_credits_;

  // Decode role only: cached per-Prefill StageAssignment (received via
  // PrefillSnapshot) plus the PeerCredits ring that allocates slots
  // within the assigned partition. Keyed by peer Prefill addr (=
  // PrefillSnapshot::snapshot.server_name). The PeerCredits is created
  // and Init'd the first time a non-zero StageAssignment arrives for
  // that peer; subsequent snapshots just refresh the assignment fields
  // (they should be stable for the pool's lifetime, but a Prefill
  // restart would surface as a different pool_base_va and we
  // re-init the ring then). Lazy + scoped to the Decode-role process.
  struct StagePeerEntry {
    StageAssignment assignment;
    // shared_ptr (not unique_ptr): SubmitPrefillRequest captures a copy of
    // this ring under stage_peers_mu_, then drops the lock to block in
    // AcquireSlots and later ReleaseSlots. A concurrent OnPrefillSnapshot
    // (Prefill restart) replaces this slot with a fresh ring; shared
    // ownership keeps the captured ring alive for the in-flight caller so
    // the blocking acquire / release can't touch freed memory.
    std::shared_ptr<PeerCredits> credits;
  };
  // Guards stage_peers_. Lock-order: state_mu_ -> stage_peers_mu_ (see the
  // convention note at state_mu_'s declaration). Never acquire state_mu_
  // while holding this.
  mutable std::mutex stage_peers_mu_;
  std::unordered_map<std::string, StagePeerEntry> stage_peers_;

  // Decode role only: per-in-flight stage slot bookkeeping. We need the
  // (peer_server_name, start_slot, n_slots) tuple at StageAck time to
  // route the release to the right PeerCredits ring; PrefillRequest
  // notify can race with PrefillRequestComplete (Prefill may finish
  // generating before its OnPrefillReq path finishes sending the
  // StageAck), so we keep this independent of in_flight_'s lifetime —
  // it lives until OnStageAck or a Release-time force-release prunes it.
  struct StageAlloc {
    std::string peer_server_name;
    uint32_t start_slot = 0;
    uint32_t n_slots = 0;
  };
  std::unordered_map<uint64_t, StageAlloc> stage_alloc_by_req_;

  // Prefill side incoming callback.
  PdIncomingReqCb on_incoming_ = nullptr;
  void* on_incoming_ctx_ = nullptr;

  // Prefill role only: local-load snapshot provider injected by the engine
  // bootstrap (SetSnapshotProvider). When set, BroadcastSnapshot uses its
  // output to fill the PeerHealth load fields instead of zeros. Null on
  // Decode-role connectors / when never wired.
  std::function<PeerHealth()> snapshot_provider_;
};

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
