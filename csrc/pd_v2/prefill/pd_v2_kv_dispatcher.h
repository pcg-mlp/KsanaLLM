/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pd_v2/public/pd_v2_connector.h"

namespace ksana_llm {

class ExecutorRequestManager;
class ExecutorInferRequest;

namespace pd_v2 {

class PdV2KvBarrier;

// PdV2KvDispatcher (executor-side, per-rank): the per-layer KV-write
// hook fired by PdV2LayerEventTracker. On each layer-completion event:
//   1. Snapshot live ExecutorInferRequests from ExecutorRequestManager.
//   2. For each, skip if pd_v2_decode_targets is empty (non-pd_v2 req)
//      or infer_stage != kContext (Prefill draft phase doesn't write).
//   3. Apply ShouldHandleEntry(my_rank, decode_target) — MHA/GQA: same
//      rank; MLA: req_id mod prefill_TP. Returns the matching target
//      or nullptr.
//   4. Chunk-end check — only emit submitTransfer at the last layer of
//      a chunk (or model's final layer).
//   5. For each block_id in matching target, compute source/target VA
//      offsets and call ExecutorTentWrapper::SubmitWriteAsync.
//   6. Track returned BatchID via MooncakeExecutorConnector::TrackInflightBatch
//      so the post-finish barrier can iterate them.
//
// Block size + num_layers are sourced from the connector's
// RegisterKvPool record (set at boot from contiguous KV pool shape).
//
// Threading: OnLayerComplete fires on PdV2LayerEventTracker monitor
// thread (host-side, no compute stream). All Mooncake calls inside are
// thread-safe per Mooncake docs. The dispatcher holds no per-step state
// (idempotent across forward steps).
class PdV2KvDispatcher {
 public:
  struct Deps {
    PdV2ConnectorInterface* connector = nullptr;
    ExecutorRequestManager* request_manager = nullptr;
    // pd_v2 transfer chunking: 0 means "transfer everything at the
    // model's final layer only" (one big batch); N>0 means "every N
    // layers fire one batch" (smaller batches, lower memory/latency).
    uint32_t transfer_layer_chunk_size = 1;
    // num_layers / total Prefill TP — needed for chunk math + MLA
    // ShouldHandleEntry. Filled by InferenceExecutor at boot.
    uint32_t num_layers = 0;
    uint32_t prefill_tp = 1;
    // attn_TP per Prefill DP group (= prefill_tp / prefill_attn_dp).
    // Dispatcher uses this to compute the per-process
    // BlockAllocatorGroup index inside the local DP group:
    //   attn_dp_rank_id = my_rank % prefill_attn_tp_per_dp
    // The req's KV blocks live in the executor's own
    // DeviceBlockAllocator at that index (other slots in the group
    // are nullptr because each executor process holds one device).
    // Hardcoding 0 only worked for prefill_tp == 1 or when the
    // selected rank happened to be slot 0.
    uint32_t prefill_attn_tp_per_dp = 1;
    // [deprecated] Kept for boot-time logging only. Dispatch is now
    // unified by physical device_id mapping (design doc §6 matrix):
    //   target_device_id = decode_attn_dp_group_id * d_attn_TP +
    //                      decode_attn_tp_rank_in_group
    //   participate iff my_rank == target_device_id
    // — same formula handles MHA/GQA and MLA. Earlier MLA branch
    // (req_id mod prefill_tp) was superseded.
    bool use_mla = false;
    // Optional: barrier to signal once the last-layer submitTransfer
    // batch has fired for a request. Dispatcher calls
    // barrier->WaitAndNotify(req_id) after the last-layer dispatch
    // succeeds. May be nullptr (tests / no barrier wiring).
    PdV2KvBarrier* barrier = nullptr;
  };

  explicit PdV2KvDispatcher(Deps deps);
  ~PdV2KvDispatcher() = default;

  PdV2KvDispatcher(const PdV2KvDispatcher&) = delete;
  PdV2KvDispatcher& operator=(const PdV2KvDispatcher&) = delete;

  // Layer-completion entry (hooked into PdV2LayerEventTracker). epoch_id
  // identifies which batch this layer belongs to — different batches map
  // to different EpochSlots in the tracker, so callbacks for different
  // batches are isolated and can't confuse each other's RDMA dispatch
  // state.
  void OnLayerComplete(int rank, uint64_t epoch_id, int layer_index);

  // Epoch-end hook (wired to PdV2LayerEventTracker::RegisterEpochEndCallback).
  // Fires once per epoch when its tracker slot is sealed — the single
  // guaranteed cleanup point even when Forward was aborted before the final
  // layer. Erases this epoch's entry from dispatch_plans_by_epoch_, releasing
  // the strong ExecutorInferRequest refs (so the reqs can destruct and return
  // their GPU blocks). Idempotent with the final-layer erase in
  // OnLayerComplete: on the normal completion path the entry is already gone
  // and this is a no-op.
  void OnEpochEnd(int rank, uint64_t epoch_id);

 private:
  // B: cache the KV pool shape. GetLocalKvPoolShape returns values that
  // are stable post-boot (block_size / num_layers), but it was being
  // called on every chunk-end callback. Resolve once on the first
  // successful fetch and serve the derived (block_size, element_size)
  // from cache thereafter. Returns false until the connector has
  // registered its KV pool (caller skips that callback and retries).
  // Thread-safe: OnLayerComplete can fire from both the tracker monitor
  // thread and the T3 ResetState force-fire path.
  bool TryGetKvShape(size_t* out_block_size, size_t* out_element_size);

  // Per-req dispatch plan, resolved ONCE per epoch (all fields are
  // forward-pass-invariant): the routing result (which DecodeTarget this
  // rank handles), the peer server_name, the local block GPU pointers,
  // and the prefix-cache skip count. The per-layer OnLayerComplete loop
  // then only computes layer-dependent chunk offsets + submits. Reqs not
  // routed to this rank (picked == nullptr) or failing block validation
  // are never added, so per-layer work never revisits them. `req` is held
  // by strong ref to keep blocks/targets alive across the forward pass.
  struct ReqDispatchPlan {
    std::shared_ptr<ExecutorInferRequest> req;
    const DecodeTargetLite* picked = nullptr;
    std::string target_server_name;
    std::vector<void*> block_ptrs;
    uint32_t cached = 0;
  };

  // Snapshot the live request manager and resolve a ReqDispatchPlan for
  // every req routed to `rank`. Called once per epoch (under
  // batch_reqs_mu_) on the first chunk-end callback. Routing / server_name
  // / GetBlockPtrs / prefix-cache count are all forward-invariant, so
  // doing them here instead of per layer removes the bulk of the
  // redundant per-chunk-end work.
  std::vector<ReqDispatchPlan> BuildDispatchPlans(int rank) const;

  // Returns true iff `layer_index` is the last layer of its chunk OR
  // the model's final layer. For chunk_size==0 emits only at final.
  bool IsChunkEndLayer(uint32_t layer_index) const;
  // Layer offset within the chunk that ends at `layer_index`. Used to
  // compute the source / target_va base. For chunk_size==0 returns 0
  // (whole-model start).
  uint32_t ChunkStartOffset(uint32_t layer_index) const;
  uint32_t ChunkSize(uint32_t layer_index) const;

  Deps deps_;

  // B: cached KV pool shape (see TryGetKvShape). Filled once on first
  // successful GetLocalKvPoolShape; stable for the connector's lifetime.
  mutable std::mutex kv_shape_mu_;
  bool kv_shape_cached_ = false;
  size_t cached_block_size_ = 0;
  size_t cached_element_size_ = 0;

  // 记录每个请求在 dispatcher 侧的提交失败。最终层通知 barrier 时带上该标记，
  // 确保部分 SubmitWriteAsync 失败也会上报 kFailed，而不是让 Decode 读取半写入的 KV。
  mutable std::mutex dispatch_failed_mu_;
  std::unordered_set<uint64_t> dispatch_failed_reqs_;

  // Per-epoch dispatch plans, keyed by tracker's epoch_id (one per
  // Forward batch). Built on the first chunk-end callback of each epoch
  // by snapshotting the live ExecutorRequestManager — the executor thread
  // (T2) is still inside Forward at that point, so the snapshot is
  // guaranteed to hold this batch's reqs. Each plan's strong shared_ptr
  // keeps the req alive even after the next batch's HandlePreprocess calls
  // EraseRequest on it. The entry is erased
  // on the model's final layer after WaitAndNotify fires. Stored behind a
  // shared_ptr so per-layer callbacks copy only the pointer (not the
  // block_ptrs vectors / strings) out from under the lock and process the
  // submits outside it.
  //
  // Why epoch-keyed: callbacks from different batches can interleave
  // (monitor lag → batch B's layer 26 fires after batch C's layer 0).
  // Keying by epoch_id ensures B's WaitAndNotify uses B's plans, and C's
  // uses C's — never mixing.
  mutable std::mutex batch_reqs_mu_;
  std::unordered_map<uint64_t, std::shared_ptr<std::vector<ReqDispatchPlan>>> dispatch_plans_by_epoch_;

  // Take-and-clear: returns true if req was previously marked failed and
  // erases the entry. Called by OnLayerComplete on the final-layer dispatch
  // signal path, immediately before barrier->WaitAndNotify(req_id, failed).
  bool TakeDispatchFailed(uint64_t req_id);
  // Mark a req as having had a dispatch-side submit failure.
  void MarkDispatchFailed(uint64_t req_id);
};

}  // namespace pd_v2
}  // namespace ksana_llm
