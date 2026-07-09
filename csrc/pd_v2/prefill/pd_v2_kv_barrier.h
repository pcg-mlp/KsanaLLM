/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include "pd_v2/public/pd_v2_connector.h"
#include "utils/logger.h"

namespace ksana_llm {

class IpcFuncWrapper;

namespace pd_v2 {

// PdV2KvBarrier (executor-side, per-rank): post-forward sender-side
// barrier for cross-node KV writes. Caller (model_runner finish path)
// invokes WaitAndNotify(req_id) when a request finishes its forward
// pass; barrier picks up the in-flight Mooncake batch ids the dispatcher
// stashed in MooncakeExecutorConnector::inflight_batches_ and polls
// each via PollBatchStatus until all reach COMPLETED (or any reaches
// FAILED). Then fires IpcFuncWrapper::ReportPdV2BarrierDone so the
// engine can release SendPrefillComplete back to peer Decode.
//
// Threading: a single dedicated background thread drains a work queue.
// WaitAndNotify is non-blocking; it just enqueues the req_id.
class PdV2KvBarrier {
 public:
  struct Deps {
    PdV2ConnectorInterface* connector = nullptr;
    IpcFuncWrapper* ipc_func_wrapper = nullptr;
    // Polling cadence — how long the worker yields between status
    // sweeps when batches are still in flight. 100us is cheap and
    // fast enough for sub-ms transfers; tune larger for slow links.
    std::chrono::microseconds poll_interval{100};
    // Per-req timeout. Once exceeded, barrier reports kFailed even if
    // some batches are still mid-flight (Mooncake CQE may be stuck or
    // peer dropped). Default 30s — generous for typical forward times.
    std::chrono::milliseconds per_req_timeout{30 * 1000};
  };

  explicit PdV2KvBarrier(Deps deps);
  ~PdV2KvBarrier();

  PdV2KvBarrier(const PdV2KvBarrier&) = delete;
  PdV2KvBarrier& operator=(const PdV2KvBarrier&) = delete;

  // Enqueue a request for barrier processing. Non-blocking. Caller is
  // expected to invoke this from the dispatcher's final-layer path right
  // after all chunks for the request are submitted (or skipped — see
  // dispatch_failed below). Always reports a barrier outcome to engine,
  // even when no batches were submitted (e.g. full prefix-cache hit on
  // Decode side, dispatcher skipped all blocks). Empty inflight batches still need a report,
  // otherwise Decode side can wait forever at kWaitAsyncResult.
  //
  // dispatch_failed: dispatcher passes true when any SubmitWriteAsync
  // for this req returned non-OK. Barrier seeds its status with
  // kBarrierStatusFailed in that case, so partial-submit failures
  // (some blocks tracked-by-connector, one block silently skipped due
  // to submit failure) surface as kFailed instead of false-OK kOk.
  // Default false preserves the normal "no submit-side failure" path.
  void WaitAndNotify(uint64_t req_id, bool dispatch_failed = false);

  // Test access.
  size_t CompletedCountForTest() const { return completed_count_.load(std::memory_order_relaxed); }
  size_t FailedCountForTest() const { return failed_count_.load(std::memory_order_relaxed); }
  // Peak queue backlog observed since construction. Also surfaced in the
  // periodic backlog LOG so prod can spot barrier head-of-line buildup (single
  // worker drains serially, a slow long-context req stalls short reqs behind it).
  size_t MaxQueueDepthForTest() const { return max_queue_depth_.load(std::memory_order_relaxed); }

 private:
  void WorkerLoop();
  void ProcessOne(uint64_t req_id, bool dispatch_failed);

  // Periodically (interval-gated) log current queue backlog so operators can
  // detect head-of-line buildup without a metrics pipeline. Only ever called
  // from the worker thread, matching how other queues log their depth from
  // the scheduler thread. in_flight_req_id is the req currently draining.
  void MaybeLogBacklog(uint64_t in_flight_req_id);

  Deps deps_;

  // Queue stores (req_id, dispatch_failed) so the worker can seed ProcessOne's
  // initial barrier status with kFailed when dispatcher-side submit failed.
  // Pair instead of separate maps keeps ordering deterministic with FIFO semantics.
  struct QueueEntry {
    uint64_t req_id;
    bool dispatch_failed;
    // Wall time the req entered the queue; the worker computes in-queue wait
    // latency from it when it picks the entry up. That wait is the direct
    // measure of head-of-line blocking and is included in the periodic LOG.
    std::chrono::steady_clock::time_point enqueue_time;
  };

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<QueueEntry> queue_;
  std::atomic<bool> running_{false};
  std::thread worker_;

  std::atomic<size_t> completed_count_{0};
  std::atomic<size_t> failed_count_{0};
  // Peak backlog ever seen (monotonic). Updated by producer threads in
  // WaitAndNotify, read by the worker thread for the periodic LOG.
  std::atomic<size_t> max_queue_depth_{0};

  // In-queue wait of the most recently dequeued req, in ms. Worker-thread only.
  uint64_t last_dequeue_wait_ms_{0};

  // Interval-gated logger for the backlog line. 10s matches BatchScheduler's
  // ReportTotalState cadence — a low-frequency health signal, not a per-req
  // trace. Worker-thread only, so no extra locking needed.
  IntervalLogger backlog_logger_{10000};
};

}  // namespace pd_v2
}  // namespace ksana_llm
