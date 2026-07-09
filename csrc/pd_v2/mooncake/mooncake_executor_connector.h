/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "pd_v2/mooncake/executor_tent_wrapper.h"
#include "pd_v2/mooncake/redis_alive_store.h"
#include "pd_v2/public/pd_v2_connector.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

// PdV2ConnectorInterface implementation for the executor-deployed
// connector. One instance per ksana_executor process (per-rank).
//
// Responsibilities:
//   - Boot: RegisterKvPool stores (base_va, total_bytes) internally
//     and forwards to ExecutorTentWrapper::RegisterKvPool (which calls
//     mooncake registerLocalMemory). The engine queries this stored
//     value via IpcFuncWrapper RPC at Decode-side Submit time.
//   - Per-request: TrackInflightBatch / EraseInflightBatch for barrier polling
//     after submitTransfer.
//
// Engine-side methods (CanAccept / SubmitPrefillRequest / Release /
// RegisterIncomingHook / SendPrefillComplete / GetKvBufferVa)
// return RET_INVALID_ARGUMENT.
class MooncakeExecutorConnector : public PdV2ConnectorInterface {
 public:
  MooncakeExecutorConnector();
  ~MooncakeExecutorConnector() override;

  Status Initialize(const PdV2ConnectorConfig& config) override;
  Status Shutdown() override;

  // Bind the executor-process IpcFuncWrapper. The override registers
  // `this` as the back-pointer for the GetKvPoolBaseVa IPC callable so
  // the engine's RPC can resolve to GetLocalKvPool. Raw pointer (no
  // ownership): the IpcFuncWrapper outlives this connector.
  void SetIpcFuncWrapper(IpcFuncWrapper* ipc_func_wrapper) override;

  // ===== Boot-time =====

  Status RegisterKvPool(void* kv_pool_base, size_t total_bytes, size_t block_size, size_t num_blocks,
                        size_t num_layers) override;

  Status GetKvBufferVa(int /*rank*/, uint64_t* /*out_kv_pool_base_va*/, uint64_t* /*out_total_bytes*/) override;

  // Executor-only state accessor — used by the IpcFuncWrapper handler
  // GetKvPoolBaseVaCallable to answer the engine's RPC. The handler
  // resolves the right MooncakeExecutorConnector instance via a
  // registered global singleton (set in factory.cpp).
  bool GetLocalKvPool(uint64_t* out_kv_pool_base_va, uint64_t* out_total_bytes, size_t* out_block_size,
                      size_t* out_num_blocks, size_t* out_num_layers) const;

  // Override of the abstract GetLocalKvPoolShape — same impl as the
  // legacy GetLocalKvPool above; PdV2KvDispatcher consumes via the
  // abstract interface so it doesn't have to depend on this concrete
  // class.
  bool GetLocalKvPoolShape(uint64_t* out_kv_pool_base_va, uint64_t* out_total_bytes, size_t* out_block_size,
                           size_t* out_num_blocks, size_t* out_num_layers) const override;

  // High-level wrapper used by PdV2KvDispatcher: wraps tent_
  // ->SubmitWriteAsync + TrackInflightBatch into one atomic step. On
  // success *out_batch_id is non-zero and the connector remembers it
  // for PdV2KvBarrier to consume. Returns non-OK on
  // submitTransfer failure, in which case nothing is tracked.
  Status SubmitWriteAsyncForRequest(const std::string& target_server_name, uint64_t target_va, void* source,
                                    size_t length, uint64_t req_id, uint64_t* out_batch_id) override;

  // Batched variant (A): coalesce all blocks of a (req, chunk) into ONE
  // Mooncake batch via ExecutorTentWrapper::SubmitWriteBatchAsync, then
  // track the single returned BatchID for `req_id`. Returns non-OK on
  // submitTransfer failure, in which case nothing is tracked.
  Status SubmitWriteBatchAsyncForRequest(const std::string& target_server_name,
                                         const std::vector<KvWriteEntry>& entries, uint64_t req_id,
                                         uint64_t* out_batch_id) override;

  // Snapshot in-flight batch ids for a request — used by PdV2KvBarrier
  // to wait for all transfers to drain post-forward. Returns empty
  // vector when the req has no surviving in-flight batches.
  std::vector<uint64_t> GetInflightBatchIds(uint64_t req_id) const override;

  // Non-blocking polling of a single batch's status — wraps
  // ExecutorTentWrapper::GetBatchStatus.
  Status PollBatchStatus(uint64_t batch_id, bool* out_completed, bool* out_failed) override;

  // Per-request Mooncake BatchID bookkeeping for PdV2KvBarrier.
  void TrackInflightBatch(uint64_t req_id, uint64_t batch_id);
  void EraseInflightBatch(uint64_t req_id, uint64_t batch_id) override;

  // ===== Decode side (engine-only) — all return invalid =====

  bool CanAccept(uint64_t /*req_id*/, uint32_t /*input_token_count*/) override { return false; }
  Status SubmitPrefillRequest(uint64_t /*req_id*/, const std::vector<int>& /*input_tokens*/,
                              const std::vector<int>& /*kv_pool_block_ids*/,
                              const SamplingConfigWire& /*sampling_config*/, uint32_t /*attn_dp_group_id*/,
                              uint32_t /*cached_block_count*/, uint32_t /*decode_prefix_cache_offset*/,
                              PdRequestCompleteCb /*on_complete*/, void* /*ctx*/) override;
  Status Release(uint64_t /*req_id*/) override;

  // ===== Prefill side (engine-only) — all return invalid =====

  void RegisterIncomingHook(PdIncomingReqCb /*on_incoming*/, void* /*ctx*/) override {}
  Status SendPrefillComplete(const std::string& /*decode_engine_inference_addr*/, uint64_t /*req_id*/,
                             PdCompleteStatus /*status*/, const std::vector<int>& /*generated_tokens*/,
                             const PeerHealth& /*snapshot*/, const PdPrefillMetrics& /*prefill_metrics*/) override;

 private:
  PdV2ConnectorConfig config_;
  std::unique_ptr<ExecutorTentWrapper> tent_;

  // Stored at RegisterKvPool, served to engine via IpcFuncWrapper RPC.
  mutable std::mutex kv_state_mu_;
  uint64_t kv_pool_base_va_ = 0;
  uint64_t kv_pool_total_bytes_ = 0;
  size_t block_size_ = 0;
  size_t num_blocks_ = 0;
  size_t num_layers_ = 0;
  bool kv_pool_registered_ = false;

  // req_id → in-flight Mooncake batch ids.
  mutable std::mutex inflight_mu_;
  std::unordered_map<uint64_t, std::vector<uint64_t>> inflight_batches_;

  // §3.3 executor-side epoch tracking. Decode restarts invalidate the
  // local Mooncake segment metadata cache (base_va / rkey change), so
  // submitTransfer with the cached SegmentID would either fail or
  // write to stale memory. This thread polls the same Redis keyspace
  // the engine uses (pd_v2/decode_alive/<cluster>/*), maintains a per-peer
  // epoch view, and on mismatch invalidates the local Mooncake cache
  // for every <addr>.exec.r{N} the executor might transfer to.
  std::unique_ptr<RedisAliveStore> redis_alive_;
  std::thread epoch_poll_;
  std::atomic<bool> epoch_poll_stop_{false};
  // addr -> latest epoch known to this executor. Mutated only by
  // EpochPollLoop, no other reader.
  std::unordered_map<std::string, uint64_t> decode_epoch_cache_;
  void EpochPollLoop();
  void InvalidatePeerSegments(const std::string& decode_addr);
};

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
