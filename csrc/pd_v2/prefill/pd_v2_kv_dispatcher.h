/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pd_v2/public/pd_v2_connector.h"
#include "utils/blocking_queue.h"

namespace ksana_llm {

class ExecutorRequestManager;
class ExecutorInferRequest;

namespace pd_v2 {

class PdV2KvBarrier;

// Per-rank KV-write dispatcher. OnLayerComplete enqueues chunk-end work to a
// submit worker; OnEpochEnd enqueues kSeal for ordered reclaim.
class PdV2KvDispatcher {
 public:
  struct Deps {
    PdV2ConnectorInterface* connector = nullptr;
    ExecutorRequestManager* request_manager = nullptr;
    // 0: transfer all at final layer; N>0: every N layers one batch.
    uint32_t transfer_layer_chunk_size = 1;
    uint32_t num_layers = 0;
    uint32_t prefill_tp = 1;
    // attn_TP per Prefill DP group; local BlockAllocatorGroup index =
    // my_rank % prefill_attn_tp_per_dp.
    uint32_t prefill_attn_tp_per_dp = 1;
    // Deprecated: kept for boot-time logging. Routing uses device_id mapping.
    bool use_mla = false;
    // Optional; nullptr in tests / when unwired.
    PdV2KvBarrier* barrier = nullptr;
  };

  explicit PdV2KvDispatcher(Deps deps);
  ~PdV2KvDispatcher();

  PdV2KvDispatcher(const PdV2KvDispatcher&) = delete;
  PdV2KvDispatcher& operator=(const PdV2KvDispatcher&) = delete;

  // Tracker monitor thread: cheap chunk-end filter, then enqueue kChunkEnd.
  void OnLayerComplete(int rank, uint64_t epoch_id, int layer_index);

  void OnEpochEnd(int rank, uint64_t epoch_id);

  // Pin this batch's reqs for BuildDispatchPlans (keyed by epoch_id).
  void CaptureEpochReqs(uint64_t epoch_id, const std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs);

  // Test-only: epochs with unreclaimed dispatch plans.
  size_t PendingEpochCountForTest() const;

  bool WaitUntilIdleForTest(int timeout_ms = 3000);

 private:
  // Cache KV pool shape from connector; false until registered.
  bool TryGetKvShape(size_t* out_block_size, size_t* out_element_size);

  // Per-req plan resolved once per epoch (routing + blocks + prefix skip).
  struct ReqDispatchPlan {
    std::shared_ptr<ExecutorInferRequest> req;
    const DecodeTargetLite* picked = nullptr;
    std::string target_server_name;
    std::vector<void*> block_ptrs;
    uint32_t cached = 0;
  };

  std::vector<ReqDispatchPlan> BuildDispatchPlans(int rank, uint64_t epoch_id) const;

  bool IsChunkEndLayer(uint32_t layer_index) const;
  uint32_t ChunkStartOffset(uint32_t layer_index) const;
  uint32_t ChunkSize(uint32_t layer_index) const;

  struct SubmitTask {
    enum class Kind { kChunkEnd, kSeal };
    Kind kind = Kind::kChunkEnd;
    int rank = 0;
    uint64_t epoch_id = 0;
    int layer_index = 0;
  };

  void WorkerLoop();

  // Submit Mooncake batches; on final layer notify barrier and reclaim.
  void DispatchChunkEnd(int rank, uint64_t epoch_id, int layer_index);

  // Reclaim when final layer never fired.
  void SealEpoch(int rank, uint64_t epoch_id);

  Deps deps_;

  // Single submit worker; submitted/processed counters for WaitUntilIdleForTest.
  std::atomic<bool> running_{false};
  BlockingQueue<SubmitTask> submit_queue_;
  std::thread submit_worker_;
  std::atomic<uint64_t> submitted_tasks_{0};
  std::atomic<uint64_t> processed_tasks_{0};

  mutable std::mutex kv_shape_mu_;
  bool kv_shape_cached_ = false;
  size_t cached_block_size_ = 0;
  size_t cached_element_size_ = 0;

  // Submit failures; final-layer barrier notify reports kFailed if set.
  mutable std::mutex dispatch_failed_mu_;
  std::unordered_set<uint64_t> dispatch_failed_reqs_;

  // Epoch-keyed plans (strong refs); not the live manager. Erased on final
  // layer / SealEpoch. shared_ptr: copy under lock, submit outside.
  mutable std::mutex batch_reqs_mu_;
  std::unordered_map<uint64_t, std::shared_ptr<std::vector<ReqDispatchPlan>>> dispatch_plans_by_epoch_;

  // BeginBatch capture; erased with dispatch_plans_by_epoch_.
  std::unordered_map<uint64_t, std::vector<std::shared_ptr<ExecutorInferRequest>>> reqs_by_epoch_;

  bool TakeDispatchFailed(uint64_t req_id);
  void MarkDispatchFailed(uint64_t req_id);

  friend class PdV2KvDispatcherEpochTestPeer;
};

}  // namespace pd_v2
}  // namespace ksana_llm
