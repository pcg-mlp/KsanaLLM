/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/prefill/pd_v2_kv_barrier.h"

#include "ipc/ipc_func_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {
namespace pd_v2 {

namespace {
constexpr uint8_t kBarrierStatusOk = 0;
constexpr uint8_t kBarrierStatusFailed = 1;
}  // namespace

PdV2KvBarrier::PdV2KvBarrier(Deps deps) : deps_(std::move(deps)) {
  if (deps_.connector == nullptr) {
    KLLM_LOG_ERROR << "PdV2KvBarrier: connector is null";
  }
  if (deps_.ipc_func_wrapper == nullptr) {
    KLLM_LOG_ERROR << "PdV2KvBarrier: ipc_func_wrapper is null";
  }
  running_.store(true, std::memory_order_release);
  worker_ = std::thread(&PdV2KvBarrier::WorkerLoop, this);
}

PdV2KvBarrier::~PdV2KvBarrier() {
  running_.store(false, std::memory_order_release);
  cv_.notify_all();
  if (worker_.joinable()) worker_.join();
}

void PdV2KvBarrier::WaitAndNotify(uint64_t req_id, bool dispatch_failed) {
  if (deps_.connector == nullptr) return;
  // WaitAndNotify 只会由 dispatcher 针对本 rank 参与的 pd_v2 请求调用。即使没有 inflight batch
  // （例如 Decode 侧全 prefix cache 命中，无需 RDMA 写），也必须入队上报；ProcessOne 会把空 batch
  // 当作 immediate kOk 处理。
  size_t depth = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    queue_.push_back({req_id, dispatch_failed, std::chrono::steady_clock::now()});
    depth = queue_.size();
  }
  // Track peak backlog (lock-free max). The single worker drains FIFO, so a
  // slow long-context req at the head stalls every short req behind it; the
  // periodic backlog LOG (worker side) surfaces this peak.
  size_t prev = max_queue_depth_.load(std::memory_order_relaxed);
  while (depth > prev && !max_queue_depth_.compare_exchange_weak(prev, depth, std::memory_order_relaxed)) {
  }
  cv_.notify_one();
}

void PdV2KvBarrier::WorkerLoop() {
  while (running_.load(std::memory_order_acquire)) {
    QueueEntry entry{};
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [this] { return !running_.load() || !queue_.empty(); });
      if (!running_.load() && queue_.empty()) return;
      if (queue_.empty()) continue;
      entry = queue_.front();
      queue_.pop_front();
    }
    // How long this entry waited in queue before the worker reached it — the
    // in-queue wait directly captures head-of-line blocking from a preceding
    // slow req. Stash it for the periodic backlog LOG.
    last_dequeue_wait_ms_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - entry.enqueue_time)
            .count());
    MaybeLogBacklog(entry.req_id);
    ProcessOne(entry.req_id, entry.dispatch_failed);
  }
}

void PdV2KvBarrier::ProcessOne(uint64_t req_id, bool dispatch_failed) {
  const auto start = std::chrono::steady_clock::now();
  // dispatcher 已标记提交失败时，barrier 直接以 kFailed 为初始状态，避免最终误报 kOk。
  uint8_t status = dispatch_failed ? kBarrierStatusFailed : kBarrierStatusOk;
  while (true) {
    auto batches = deps_.connector->GetInflightBatchIds(req_id);
    if (batches.empty()) {
      // All batches drained.
      break;
    }
    bool any_in_flight = false;
    bool any_failed = false;
    for (uint64_t b : batches) {
      bool completed = false;
      bool failed = false;
      Status s = deps_.connector->PollBatchStatus(b, &completed, &failed);
      if (!s.OK()) {
        // Polling itself failed — be conservative, mark req failed.
        any_failed = true;
        break;
      }
      if (failed) {
        any_failed = true;
        break;
      }
      if (completed) {
        deps_.connector->EraseInflightBatch(req_id, b);
      } else {
        any_in_flight = true;
      }
    }
    if (any_failed) {
      status = kBarrierStatusFailed;
      break;
    }
    if (!any_in_flight) {
      // All polled were completed and erased; loop top will see empty.
      continue;
    }
    if (std::chrono::steady_clock::now() - start > deps_.per_req_timeout) {
      KLLM_LOG_ERROR << "PdV2KvBarrier: req_id=" << req_id << " timed out after " << deps_.per_req_timeout.count()
                     << "ms; reporting kFailed";
      status = kBarrierStatusFailed;
      break;
    }
    // A single long-context req can hold the worker here for a while; keep the
    // backlog LOG ticking (interval-gated) so a stuck head still surfaces the
    // reqs piling up behind it.
    MaybeLogBacklog(req_id);
    std::this_thread::sleep_for(deps_.poll_interval);
  }

  if (status == kBarrierStatusOk) {
    completed_count_.fetch_add(1, std::memory_order_relaxed);
  } else {
    failed_count_.fetch_add(1, std::memory_order_relaxed);
  }

  if (deps_.ipc_func_wrapper != nullptr) {
    Status s = deps_.ipc_func_wrapper->ReportPdV2BarrierDone(static_cast<int64_t>(req_id), status);
    if (!s.OK()) {
      KLLM_LOG_WARNING << "PdV2KvBarrier: ReportPdV2BarrierDone failed req_id=" << req_id << " err=" << s.GetMessage();
    }
  }
}

void PdV2KvBarrier::MaybeLogBacklog(uint64_t in_flight_req_id) {
  if (!backlog_logger_.ShouldLog()) return;
  size_t pending = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    pending = queue_.size();
  }
  KLLM_LOG_INFO << "PdV2KvBarrier backlog: pending=" << pending
                << " peak=" << max_queue_depth_.load(std::memory_order_relaxed)
                << " last_wait_ms=" << last_dequeue_wait_ms_ << " processing_req_id=" << in_flight_req_id
                << " completed=" << completed_count_.load(std::memory_order_relaxed)
                << " failed=" << failed_count_.load(std::memory_order_relaxed);
}

}  // namespace pd_v2
}  // namespace ksana_llm
