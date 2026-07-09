/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "pd_v2/public/pd_v2_types.h"
#include "utils/status.h"

// Forward decl to keep mooncake headers out of public include path.
namespace mooncake {
class TransferEngine;
}  // namespace mooncake

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

// Engine-side wrapper around mooncake::TransferEngine. Used by
// MooncakeEngineConnector for two roles:
//   1. Control-plane notifies (sendNotification, getNotifies) — the
//      original TCP-only path. This is what
//      pd_v2/prefill_req|complete|stage_ack|... rides on.
//   2. Control-stage RDMA buffer: Prefill registers its HostStagePool
//      via RegisterStageMemory; Decode submits RDMA WRITE batches into
//      that buffer via SubmitStageWriteAsync. This carries the heavy
//      prefill_req payload (32K+ tokens) so the small notify can stay
//      under the 64 KB Tent notify cap. See
//      docs/technology/design/pd_v2_control_stage.md §5.
//
// Lifecycle: constructor stores config, Initialize() builds the
// underlying TransferEngine and installs both RDMA + TCP transports,
// Shutdown() tears it down. Init must be called exactly once per
// instance.
//
// Threading: SendNotification / SubmitStageWriteAsync / GetBatchStatus
// can be called concurrently from multiple threads. ReceiveNotifications
// must be called from a single dedicated poller thread (MooncakeEngineConnector
// spawns one).
//
// Why dual-transport on the engine instance: in pre-stage builds
// the engine TransferEngine was TCP-only because notifies are tiny.
// Adding the stage RDMA channel here (rather than spinning up a third
// TransferEngine) keeps the wire/segment graph simple — Prefill registers
// once on its engine TransferEngine, Decode resolves once via openSegment,
// and the existing control-plane notify name carries the slot reference.
class EngineTentWrapper {
 public:
  EngineTentWrapper();
  ~EngineTentWrapper();

  EngineTentWrapper(const EngineTentWrapper&) = delete;
  EngineTentWrapper& operator=(const EngineTentWrapper&) = delete;

  // Initialize the underlying mooncake::TransferEngine.
  //   metadata_addr  - "http://host:port/" / "etcd://..." / "P2PHANDSHAKE"
  //                    forwarded to TransferEngine::init.
  //   local_server_name - this engine's identity in the metadata server.
  //                       Format "ip:port" (Mooncake convention).
  // Returns RET_RUNTIME_FAILED on any mooncake init failure.
  Status Initialize(const std::string& metadata_addr, const std::string& local_server_name);

  Status Shutdown();

  // Send a notification to the named peer engine. The notification's
  // `name` field selects the channel (pd_v2/prefill_req|complete|...).
  // `peer_server_name` must be the remote engine's identity from its
  // own Initialize call (Mooncake matches by name; segment lookup
  // happens internally via openSegment).
  Status SendNotification(const std::string& peer_server_name, const Notification& notify);

  // Drop the cached notify SegmentHandle for `peer_server_name` so the next
  // SendNotification re-runs openSegment (re-reading the peer's metadata from
  // Redis and rebuilding the notify QP). Call this when the peer is known to
  // have restarted (epoch mismatch) — otherwise the cached handle points at a
  // torn-down QP and every send flushes with IBV_WC_WR_FLUSH_ERR. No-op if the
  // peer was never cached.
  void InvalidatePeerHandle(const std::string& peer_server_name);

  // Drain pending notifies. Called by the connector's poller thread on
  // a fixed interval. Empties the mooncake-side queue into `out`.
  // Returns OK with empty `out` if nothing pending; never blocks.
  Status ReceiveNotifications(std::vector<Notification>* out);

  // Boot-time: register HostStagePool memory with mooncake so peer
  // Decode engines can RDMA-WRITE into it. Called by Prefill side once
  // at engine init. `base` and `total_bytes` come straight from
  // HostStagePool::base() / HostStagePool::total_bytes(). Memory class
  // is hard-coded to "cpu:0" — stage buffer is host-side mmap, not
  // device memory.
  Status RegisterStageMemory(void* base, size_t total_bytes);

  // Submit an async RDMA WRITE into a remote engine's stage buffer.
  // Returns the BatchID immediately; caller polls GetBatchStatus for
  // completion. `target_offset` is the absolute remote VA: typically
  // `<remote stage_base_va> + start_slot * kSlotSize` where
  // stage_base_va is communicated through the Prefill heartbeat
  // (see service-discovery payload in design doc §3.3 / §5.1).
  // `*out_batch_id` is set to 0 on failure with non-OK return.
  Status SubmitStageWriteAsync(const std::string& target_server_name, uint64_t target_offset, void* source,
                               size_t length, uint64_t* out_batch_id);

  // Non-blocking status query for a batch returned from
  // SubmitStageWriteAsync. *out_completed is true when the batch
  // reached COMPLETED; *out_failed is true on FAILED / TIMEOUT /
  // CANCELED / INVALID. Both can be false simultaneously (still in
  // flight).
  Status GetBatchStatus(uint64_t batch_id, bool* out_completed, bool* out_failed);

  // Best-effort batch teardown. Mirrors ExecutorTentWrapper::FreeBatchID;
  // failures are logged but not surfaced.
  Status FreeBatchID(uint64_t batch_id);

  // Read-only accessor for the underlying TransferEngine pointer.
  // Used by the connector to share the same TransferEngine with the
  // executor wrapper if both run in the same process (single-process
  // dev mode); production has separate engine + executor processes
  // each with their own TransferEngine.
  mooncake::TransferEngine* raw() { return engine_.get(); }

 private:
  std::unique_ptr<mooncake::TransferEngine> engine_;
  std::string local_server_name_;
  bool initialized_ = false;

  // Per-peer cached SegmentHandle for sendNotifyByID. Never evicted —
  // mooncake's sendNotifyByName open/send/close cycle races on concurrent
  // calls to the same peer (closeRemote unconditionally erases the handle
  // from SegmentManager's map, so an in-flight sendNotification with the
  // erased handle fails with "Invalid segment handle"). Caching avoids
  // calling close at all; teardown happens in engine_.reset().
  // SegmentHandle is a mooncake integer alias; int64_t keeps mooncake
  // headers out of the public include path.
  std::unordered_map<std::string, int64_t> peer_handle_cache_;
  std::mutex peer_handle_mutex_;
};

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
