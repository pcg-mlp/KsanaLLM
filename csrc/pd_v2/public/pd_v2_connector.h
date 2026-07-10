/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pd_v2/public/pd_v2_config.h"
#include "pd_v2/public/pd_v2_types.h"
#include "pd_v2/wire/sampling_config_wire.h"
#include "utils/status.h"

namespace ksana_llm {

class IpcFuncWrapper;

namespace pd_v2 {

// LinUCB router (selector_policy == kLinUCB) local-arm context: this Decode
// node's live load at the moment a request is routed. The router turns this
// into the 4-dim feature vector shared by both arms. Sourced by the Decode
// hook from ReadLocalLoad + the per-step EWMA it already tracks.
struct LocalArmContext {
  uint32_t running_reqs = 0;              // decoding_reqs.size(); low = idle, waiting on Prefill
  uint32_t step_latency_us = 0;           // EWMA per-step decode latency, local TPOT proxy
  uint32_t step_latency_us_inst = 0;      // most recent (non-EWMA) per-step latency; the 50ms TPOT
                                          // gate reads this so a load spike is caught on the next
                                          // decision instead of lagging behind the EWMA
  uint32_t remaining_prefill_tokens = 0;  // input tokens left to prefill after local prefix-cache hit
};

// LinUCB binary decision: keep this request's prefill LOCAL, or forward it to a
// remote Prefill peer. The remote *peer* choice is still made by PeerSelector;
// this enum only names the two bandit arms. Values are stable arm indices.
enum class RouteArm : uint8_t {
  kRemotePeer = 0,  // forward to a Prefill peer (current default behavior)
  kLocal = 1,       // run remaining prefill + decode on this node
};

// pd_v2 connector ABI. One instance per process (engine or executor).
// Methods are split into engine-side and executor-side blocks below; the
// `deployment_role` field of the supplied PdV2ConnectorConfig selects which
// block is exercised. Calling an executor-side method on an engine-deployed
// connector (or vice versa) returns RET_INVALID_ARGUMENT.
//
// Threading: implementations must be safe to call concurrently from a single
// engine event loop and any number of executor worker threads. Per-request
// state is owned by the connector internally; callers only pass req_id.
class PdV2ConnectorInterface {
 public:
  virtual ~PdV2ConnectorInterface() = default;

  virtual Status Initialize(const PdV2ConnectorConfig& config) = 0;
  virtual Status Shutdown() = 0;

  // Inject the engine-process IpcFuncWrapper. Engine-side backends use it
  // to round-trip GetKvBufferVa to the per-rank ksana_executor processes
  // (executor side answers from its locally stored RegisterKvPool values).
  // Backends that don't need cross-process plumbing leave the default
  // no-op. Must be called before any GetKvBufferVa call. The pointer must
  // outlive the connector — typically owned by InferenceEngine.
  virtual void SetIpcFuncWrapper(IpcFuncWrapper* /*ipc_func_wrapper*/) {}

  // [Prefill role] Inject a provider that produces the current local
  // PeerHealth load snapshot. The connector calls it from its periodic
  // BroadcastSnapshot heartbeat to fill the load fields pushed to Decode
  // peers (free/future blocks, waiting/running/pending, per-step EWMA).
  // Backends without periodic snapshots (mock) leave the default no-op.
  // The per-completion path is fed separately via the endpoint's snapshot_fn.
  virtual void SetSnapshotProvider(std::function<PeerHealth()> /*provider*/) {}

  // ===== Boot-time (服务启动期一次性) =====
  //
  // See design doc §2.0 (服务启动 Boot 流程). Two methods only:
  //   - executor: RegisterKvPool stores (base_va, total_bytes, ...)
  //               internally + calls mooncake registerLocalMemory.
  //   - engine:   GetKvBufferVa returns the cached (base_va, bytes) for
  //               a given rank, populated lazily via existing
  //               IpcFuncWrapper RPC to the executor process. Boot
  //               doesn't need a separate "Wait until all ranks
  //               publish" gate — the engine simply queries each rank
  //               on first use, blocking on the RPC until that rank
  //               answers.
  //
  // Naming note: real Mooncake doesn't return per-buffer segment_id
  // from registerLocalMemory. A SegmentID identifies a remote
  // TransferEngine instance (one per process), and remote peers obtain
  // it by calling openSegment(server_name) at handshake time. What
  // crosses the wire is therefore the remote-VA range of the KV pool —
  // peer Prefill writes via submitTransfer(target_id =
  // openSegment(decode_inference_addr), target_offset =
  // kv_pool_base_va + block_id * block_size).

  // [executor] Register the local KV pool memory with the underlying
  // transport so peer Prefill nodes can RDMA-WRITE into it. Implementation
  // also stores (kv_pool_base_va, total_bytes) internally so the engine
  // process can retrieve them via IpcFuncWrapper RPC (GetKvBufferVa
  // below).
  //   - mooncake: executor_mooncake.registerLocalMemory(base, total_bytes,
  //               location=cuda:0|cpu:0, remote_accessible=true).
  // `block_size` / `num_blocks` / `num_layers` are required so peer
  // Prefill can compute target_offset = kv_pool_base_va +
  // block_id * block_size + chunk_layer_offset without an extra
  // metadata exchange.
  // Returns error if the connector is engine-deployed.
  virtual Status RegisterKvPool(void* kv_pool_base, size_t total_bytes, size_t block_size, size_t num_blocks,
                                size_t num_layers) = 0;

  // [engine] Look up the (kv_pool_base_va, total_bytes) for a given
  // local rank. Engine-side implementation runs an IpcFuncWrapper RPC
  // to the executor process for that rank (which answers from the
  // values it stored at RegisterKvPool time), caches the result in
  // a local map, and returns. First call per rank pays the IPC
  // round-trip; subsequent calls hit the cache.
  // Called by PdV2DecodeHook at SubmitPrefillRequest time to populate
  // decode_targets[]. Returns error on RPC failure or executor not
  // ready (RegisterKvPool not yet called on that rank).
  // Returns error if the connector is executor-deployed.
  virtual Status GetKvBufferVa(int rank, uint64_t* out_kv_pool_base_va, uint64_t* out_total_bytes) = 0;

  // ===== ksana_engine process (per-request) =====

  // ----- Decode side -----

  // Admission control: PeerSelector picks a peer. There is no per-request
  // resource reservation on the Decode side (no HBP slot — tokens are
  // returned via the completion notify). All-or-nothing; on false return
  // the caller does not need to clean up.
  virtual bool CanAccept(uint64_t req_id, uint32_t input_token_count) = 0;

  // LinUCB routing decision (selector_policy == kLinUCB). Returns:
  //   kRemotePeer — admit + forward to a Prefill peer (peer chosen by the same
  //                 PeerSelector path CanAccept uses; the request is recorded
  //                 in_flight_ exactly as CanAccept would).
  //   kLocal      — keep the remaining prefill on this Decode node; the caller
  //                 does NOT SubmitPrefillRequest and does NOT enter in_flight_.
  //   nullopt     — no healthy peer for a remote decision (== CanAccept false);
  //                 caller fails the request.
  // Default (and every non-kLinUCB policy) is equivalent to CanAccept: forward
  // when a peer is available, otherwise fail. Only the mooncake backend with
  // kLinUCB consults the bandit. ctx carries this node's live load.
  virtual std::optional<RouteArm> RouteRequest(uint64_t req_id, const LocalArmContext& ctx) {
    return CanAccept(req_id, ctx.remaining_prefill_tokens) ? std::optional<RouteArm>(RouteArm::kRemotePeer)
                                                           : std::nullopt;
  }

  // Compose the PrefillRequest wire frame for the assigned peer and send
  // it over the engine<->engine Mooncake notify channel. Returns
  // immediately; on_complete fires on the engine notify-dispatch thread
  // when the peer Prefill engine reports PrefillRequestComplete (carrying
  // generated_tokens when status == kOk).
  //
  // attn_dp_group_id: which Decode attn_dp group this request belongs to
  // (BasicWorkloadBalancer assigns it at admission; req->attn_dp_group_id
  // on the InferRequest). Drives which executors' KV pools we look up
  // when packing decode_targets[]: the r-th target's base_va is fetched
  // via GetKvBufferVa(attn_dp_group_id * attn_TP + r). Without this,
  // Decode TP > attn_TP (i.e. attn_DP > 1) silently routes every
  // request to attn_dp_group=0's KV pool.
  virtual Status SubmitPrefillRequest(uint64_t req_id, const std::vector<int>& input_tokens,
                                      const std::vector<int>& kv_pool_block_ids,
                                      const SamplingConfigWire& sampling_config, uint32_t attn_dp_group_id,
                                      // pd_v2 prefix-cache awareness. Decode side computes hit length
                                      // (= req->prefix_cache_len) before calling. cached_block_count is
                                      // the block-level form (= prefix_cache_len / block_token_num);
                                      // the dispatcher iterates kv_pool_block_ids[cached_block_count..]
                                      // for RDMA writes (skipping leading cached blocks). See
                                      // wire DecodeTarget for full rationale. Both default 0 = no
                                      // prefix cache hit, all blocks transferred (current behavior).
                                      uint32_t cached_block_count, uint32_t decode_prefix_cache_offset,
                                      PdRequestCompleteCb on_complete, void* ctx) = 0;

  // [Decode role] Health: whether at least one Prefill peer is currently
  // healthy (selectable) in the PeerSelector table. Used by the Decode node's
  // /health endpoint to gate on Prefill reachability. Read-only; never
  // affects routing. Default false for backends / roles without a peer table
  // (Prefill side, mock backend, etc.).
  virtual bool HasHealthyPrefillPeer() const { return false; }

  // Local cleanup / safety-net, called when an InferRequest is destroyed
  // or a Submit fails — regardless of whether on_complete already fired.
  // Erases in_flight_ bookkeeping and force-releases any held stage
  // credits (normally freed by OnStageAck). Does not notify Prefill:
  // there is no "cancel" concept on the PD wire — cancel/timeout while
  // pd_v2_awaiting_prefill is handled on Decode by deferring Stop until
  // OnPrefillComplete. Idempotent for unknown req_id.
  virtual Status Release(uint64_t req_id) = 0;

  // ----- Prefill side -----

  // Register the engine's incoming-PrefillRequest handler. The handler
  // (PdV2PrefillIntakeWrapper) builds a minimal InferRequest with
  // sampling_config translated from
  // IncomingPrefillRequest.sampling_config_wire and submits it to Prefill
  // EDS. There is no per-request setup IPC to the executor; the executor
  // connector subscribes to ModelExecutor per-layer events at boot and
  // dispatches based on the wire-supplied decode_targets.
  virtual void RegisterIncomingHook(PdIncomingReqCb on_incoming, void* ctx) = 0;

  // Reply path called by the application after Prefill's first-step
  // LM_head + sampler produced the requested tokens. The connector
  // reverse-routes via the decode_engine_inference_addr remembered at
  // incoming time — the same string the wrapper got via
  // IncomingPrefillRequest.
  virtual Status SendPrefillComplete(const std::string& decode_engine_inference_addr, uint64_t req_id,
                                     PdCompleteStatus status, const std::vector<int>& generated_tokens,
                                     const PeerHealth& snapshot, const PdPrefillMetrics& prefill_metrics) = 0;

  // ===== ksana_executor process =====
  //
  // Per-request setup is intentionally absent. Decode-side getNotifies
  // poller and Prefill-side per-layer-event subscription are wired up in
  // Initialize and run for the lifetime of the connector.

  // ===== pd_v2 Prefill executor-side per-layer KV-write surface =====
  //
  // These methods are called by PdV2KvDispatcher (per-layer hook) and
  // PdV2KvBarrier (post-forward sender-side barrier). Defaulted to
  // no-op so backends without a real Mooncake transport (mock-style
  // or non-Prefill processes) compile + dlopen cleanly.
  //
  // Backend authors implementing these should keep semantics identical
  // to the Mooncake reference impl in mooncake_executor_connector.cpp.

  // Read back the local KV pool shape registered via RegisterKvPool.
  // PdV2KvDispatcher consumes block_size + num_layers to compute
  // element_size = block_size / num_layers. Default returns false
  // (no KV pool registered).
  virtual bool GetLocalKvPoolShape(uint64_t* /*out_kv_pool_base_va*/, uint64_t* /*out_total_bytes*/,
                                   size_t* /*out_block_size*/, size_t* /*out_num_blocks*/,
                                   size_t* /*out_num_layers*/) const {
    return false;
  }

  // Submit one-Request batch (WRITE) and track the returned BatchID
  // for `req_id` so barrier can iterate it. *out_batch_id
  // is the Mooncake BatchID on success, 0 on failure. No-op default.
  virtual Status SubmitWriteAsyncForRequest(const std::string& /*target_server_name*/, uint64_t /*target_va*/,
                                            void* /*source*/, size_t /*length*/, uint64_t /*req_id*/,
                                            uint64_t* out_batch_id) {
    if (out_batch_id) *out_batch_id = 0;
    return Status();
  }

  // One coalesced RDMA-WRITE target for SubmitWriteBatchAsyncForRequest.
  // target_va is the absolute remote VA (= remote kv_pool_base_va +
  // block_id * block_size + chunk_layer_offset); source is the local GPU
  // pointer for that block's chunk; length is the per-block chunk bytes.
  struct KvWriteEntry {
    uint64_t target_va = 0;
    void* source = nullptr;
    size_t length = 0;
  };

  // Batched variant of SubmitWriteAsyncForRequest: submit all `entries`
  // (all to the same peer `target_server_name`) as ONE Mooncake batch —
  // a single openSegment + allocateBatchID + submitTransfer — and track
  // the single returned BatchID for `req_id`. This collapses the
  // previous per-block submit (O(blocks) syscalls/locks on the caller's
  // thread, O(blocks) batch ids for the barrier to poll) into O(1) per
  // (req, chunk). *out_batch_id is the BatchID on success, 0 on failure
  // or when `entries` is empty. No-op default.
  virtual Status SubmitWriteBatchAsyncForRequest(const std::string& /*target_server_name*/,
                                                 const std::vector<KvWriteEntry>& /*entries*/, uint64_t /*req_id*/,
                                                 uint64_t* out_batch_id) {
    if (out_batch_id) *out_batch_id = 0;
    return Status();
  }

  // Snapshot in-flight Mooncake batch ids for a request. Returned to the
  // barrier thread for polling. Empty default.
  virtual std::vector<uint64_t> GetInflightBatchIds(uint64_t /*req_id*/) const { return {}; }

  // Non-blocking status poll for a single batch — wraps Mooncake
  // getBatchTransferStatus. Both *out_completed and *out_failed false
  // means still in flight. Default reports completed=true (no async
  // backend → nothing to wait on).
  virtual Status PollBatchStatus(uint64_t /*batch_id*/, bool* out_completed, bool* out_failed) {
    if (out_completed) *out_completed = true;
    if (out_failed) *out_failed = false;
    return Status();
  }

  // Erase a batch id from the in-flight tracking once observed
  // completed by the barrier. No-op default.
  virtual void EraseInflightBatch(uint64_t /*req_id*/, uint64_t /*batch_id*/) {}
};

// Factory entrypoint exported by every pd_v2 backend shared library.
// libpd_v2_connector_mooncake.so exports this symbol; the application
// calls it after dlopen+dlsym.
//
// ABI note: the entire PdV2ConnectorInterface ABI — this factory's
// PdV2ConnectorConfig argument, the returned shared_ptr, and every virtual
// method — passes STL types (std::string, std::vector, std::function,
// std::optional, std::shared_ptr) across the dlopen boundary. This is sound
// ONLY because the .so is built from the same source tree, toolchain, and
// libstdc++ (same _GLIBCXX_USE_CXX11_ABI) as the host ksana_engine /
// ksana_executor process, in a single CMake build. Mixing libstdc++ versions
// or CXX11 ABI settings between the host and this .so changes STL memory
// layout and causes silent corruption — so do NOT rebuild this .so in
// isolation with a different toolchain. The extern "C" linkage here only
// stabilizes the symbol name, not the STL layout. (The --exclude-libs,ALL
// link option in this backend's CMakeLists isolates vendored gflags/glog
// statics and is unrelated to STL ABI.)
extern "C" std::shared_ptr<PdV2ConnectorInterface> ksana_pd_v2_create_connector(const PdV2ConnectorConfig& cfg);

}  // namespace pd_v2
}  // namespace ksana_llm
