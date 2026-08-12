/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/prefill/pd_v2_kv_dispatcher.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "executor/executor_request_manager.h"
#include "pd_v2/prefill/pd_v2_kv_barrier.h"
#include "runtime/executor_infer_request.h"
#include "runtime/infer_stage.h"
#include "utils/logger.h"

namespace ksana_llm {
namespace pd_v2 {

PdV2KvDispatcher::PdV2KvDispatcher(Deps deps) : deps_(std::move(deps)) {
  if (deps_.connector == nullptr) {
    KLLM_LOG_ERROR << "PdV2KvDispatcher: connector is null";
  }
  if (deps_.request_manager == nullptr) {
    KLLM_LOG_ERROR << "PdV2KvDispatcher: request_manager is null";
  }
  if (deps_.num_layers == 0) {
    KLLM_LOG_ERROR << "PdV2KvDispatcher: num_layers=0; chunk math is invalid";
  }
  running_.store(true, std::memory_order_release);
  submit_worker_ = std::thread([this] { WorkerLoop(); });
  KLLM_LOG_INFO << "PdV2KvDispatcher initialized num_layers=" << deps_.num_layers << " prefill_tp=" << deps_.prefill_tp
                << " transfer_layer_chunk_size=" << deps_.transfer_layer_chunk_size << " use_mla=" << deps_.use_mla;
}

PdV2KvDispatcher::~PdV2KvDispatcher() {
  // Stop worker before deps_ / members are destroyed; queued tasks are dropped.
  running_.store(false, std::memory_order_release);
  submit_queue_.Stop();
  if (submit_worker_.joinable()) submit_worker_.join();
}

void PdV2KvDispatcher::WorkerLoop() {
  while (running_.load(std::memory_order_acquire)) {
    SubmitTask task = submit_queue_.Get();
    if (!running_.load(std::memory_order_acquire)) break;  // queue stopped
    switch (task.kind) {
      case SubmitTask::Kind::kChunkEnd:
        DispatchChunkEnd(task.rank, task.epoch_id, task.layer_index);
        break;
      case SubmitTask::Kind::kSeal:
        SealEpoch(task.rank, task.epoch_id);
        break;
    }
    // After side effects land (WaitUntilIdleForTest relies on this).
    processed_tasks_.fetch_add(1, std::memory_order_release);
  }
}

bool PdV2KvDispatcher::IsChunkEndLayer(uint32_t layer_index) const {
  if (deps_.num_layers == 0) return false;
  const uint32_t last_layer = deps_.num_layers - 1;
  if (layer_index == last_layer) return true;
  // chunk_size == 0 → "transfer once at final layer only"
  if (deps_.transfer_layer_chunk_size == 0) return false;
  // (layer_index + 1) % chunk_size == 0 means we just finished a chunk
  return ((layer_index + 1) % deps_.transfer_layer_chunk_size) == 0;
}

uint32_t PdV2KvDispatcher::ChunkStartOffset(uint32_t layer_index) const {
  if (deps_.transfer_layer_chunk_size == 0) return 0;
  return (layer_index / deps_.transfer_layer_chunk_size) * deps_.transfer_layer_chunk_size;
}

uint32_t PdV2KvDispatcher::ChunkSize(uint32_t layer_index) const {
  const uint32_t start = ChunkStartOffset(layer_index);
  if (deps_.transfer_layer_chunk_size == 0) {
    // Whole-model in one shot at final layer
    return deps_.num_layers;
  }
  return std::min(deps_.transfer_layer_chunk_size, deps_.num_layers - start);
}

bool PdV2KvDispatcher::TryGetKvShape(size_t* out_block_size, size_t* out_element_size) {
  std::lock_guard<std::mutex> lk(kv_shape_mu_);
  if (!kv_shape_cached_) {
    uint64_t base_va = 0, total_bytes = 0;
    size_t block_size = 0, num_blocks = 0, num_layers_local = 0;
    if (!deps_.connector->GetLocalKvPoolShape(&base_va, &total_bytes, &block_size, &num_blocks, &num_layers_local)) {
      return false;  // KV pool not registered yet — caller skips + retries.
    }
    if (block_size == 0 || num_layers_local == 0) {
      return false;
    }
    cached_block_size_ = block_size;
    cached_element_size_ = block_size / num_layers_local;
    kv_shape_cached_ = true;
  }
  if (out_block_size) *out_block_size = cached_block_size_;
  if (out_element_size) *out_element_size = cached_element_size_;
  return true;
}

std::vector<PdV2KvDispatcher::ReqDispatchPlan> PdV2KvDispatcher::BuildDispatchPlans(int rank, uint64_t epoch_id) const {
  std::vector<ReqDispatchPlan> plans;
  const std::vector<std::shared_ptr<ExecutorInferRequest>>* live_reqs_ptr = nullptr;
  auto reqs_it = reqs_by_epoch_.find(epoch_id);
  if (reqs_it != reqs_by_epoch_.end()) live_reqs_ptr = &reqs_it->second;
  if (live_reqs_ptr == nullptr) return plans;  // batch not captured / already cleaned
  for (auto& req : *live_reqs_ptr) {
    if (req == nullptr) continue;
    if (req->pd_v2_decode_targets.empty()) continue;
    if (req->infer_stage != InferStage::kContext && req->step != 0) continue;
    if (!req->is_last_prefill_step) {
      KLLM_LOG_DEBUG << "PdV2KvDispatcher[split_fuse_defer] req=" << req->req_id << " my_rank=" << rank
                     << " non-final prefill chunk — deferring KV write to last chunk";
      continue;
    }

    const uint32_t d_attn_tp = static_cast<uint32_t>(req->pd_v2_decode_targets.size());
    const DecodeTargetLite* picked = nullptr;
    uint32_t target_device_id = 0;
    for (const auto& t : req->pd_v2_decode_targets) {
      const uint32_t did = req->pd_v2_decode_attn_dp_group_id * d_attn_tp + t.decode_attn_tp_rank_in_group;
      if (static_cast<int>(did) == rank) {
        picked = &t;
        target_device_id = did;
        break;
      }
    }
    if (picked == nullptr) {
      // Not this Prefill rank's DecodeTarget.
      KLLM_LOG_DEBUG << "PdV2KvDispatcher[skip] req=" << req->req_id << " my_rank=" << rank
                     << " d_gid=" << req->pd_v2_decode_attn_dp_group_id << " d_attn_tp=" << d_attn_tp
                     << " prefill_tp=" << deps_.prefill_tp;
      continue;
    }
    KLLM_LOG_DEBUG << "PdV2KvDispatcher[route] req=" << req->req_id << " my_rank=" << rank
                   << " d_gid=" << req->pd_v2_decode_attn_dp_group_id << " d_attn_tp=" << d_attn_tp
                   << " r=" << picked->decode_attn_tp_rank_in_group << " target_device_id=" << target_device_id;

    ReqDispatchPlan plan;
    plan.req = req;
    plan.picked = picked;
    plan.target_server_name = req->pd_v2_decode_engine_inference_addr + ".exec.r" + std::to_string(target_device_id);
    const int attn_dp_rank_id =
        deps_.prefill_attn_tp_per_dp == 0 ? 0 : (rank % static_cast<int>(deps_.prefill_attn_tp_per_dp));
    plan.block_ptrs = req->GetBlockPtrs(attn_dp_rank_id);
    plan.cached = picked->cached_block_count;
    if (plan.cached > 0) {
      KLLM_LOG_DEBUG << "PdV2KvDispatcher[prefix_cache_skip] req=" << req->req_id
                     << " cached_block_count=" << plan.cached
                     << " (decode_prefix_cache_offset_tokens=" << picked->decode_prefix_cache_offset
                     << ") total_blocks=" << picked->kv_block_ids.size() << " writing_from=" << plan.cached;
    }
    if (plan.block_ptrs.size() < picked->kv_block_ids.size()) {
      KLLM_LOG_ERROR << "PdV2KvDispatcher: kv_cache_blocks size " << plan.block_ptrs.size()
                     << " < target.kv_block_ids size " << picked->kv_block_ids.size() << " for req " << req->req_id;
      continue;  // malformed — skip (matches prior per-layer behavior)
    }
    plans.push_back(std::move(plan));
  }
  return plans;
}

void PdV2KvDispatcher::OnLayerComplete(int rank, uint64_t epoch_id, int layer_index) {
  if (deps_.connector == nullptr || deps_.request_manager == nullptr) return;
  if (layer_index < 0 || static_cast<uint32_t>(layer_index) >= deps_.num_layers) return;
  const uint32_t li = static_cast<uint32_t>(layer_index);
  if (!IsChunkEndLayer(li)) return;

  KLLM_LOG_DEBUG << "PdV2KvDispatcher::OnLayerComplete rank=" << rank << " epoch=" << epoch_id
                 << " layer=" << layer_index << " chunk-end -> enqueue";
  submitted_tasks_.fetch_add(1, std::memory_order_relaxed);
  submit_queue_.Put(SubmitTask{SubmitTask::Kind::kChunkEnd, rank, epoch_id, layer_index});
}

void PdV2KvDispatcher::DispatchChunkEnd(int rank, uint64_t epoch_id, int layer_index) {
  const uint32_t li = static_cast<uint32_t>(layer_index);
  size_t block_size = 0, element_size = 0;
  if (!TryGetKvShape(&block_size, &element_size)) {
    KLLM_LOG_WARNING << "PdV2KvDispatcher: KV pool shape unavailable, skipping layer " << layer_index;
    return;
  }
  const uint32_t chunk_start = ChunkStartOffset(li);
  const uint32_t chunk_size = ChunkSize(li);
  const size_t chunk_bytes = element_size * static_cast<size_t>(chunk_size);
  const size_t chunk_byte_offset = element_size * static_cast<size_t>(chunk_start);

  // Build plans once per epoch (routing/blocks are forward-invariant); strong
  // refs keep reqs alive after EraseRequest. Erased on final layer.
  std::shared_ptr<std::vector<ReqDispatchPlan>> plans;
  {
    std::lock_guard<std::mutex> lk(batch_reqs_mu_);
    auto it = dispatch_plans_by_epoch_.find(epoch_id);
    if (it == dispatch_plans_by_epoch_.end()) {
      auto built = std::make_shared<std::vector<ReqDispatchPlan>>(BuildDispatchPlans(rank, epoch_id));
      it = dispatch_plans_by_epoch_.emplace(epoch_id, std::move(built)).first;
      constexpr size_t kPendingEpochWarnThreshold = 64;
      if (dispatch_plans_by_epoch_.size() >= kPendingEpochWarnThreshold) {
        KLLM_LOG_WARNING << "PdV2KvDispatcher: dispatch_plans_by_epoch_ size=" << dispatch_plans_by_epoch_.size()
                         << " exceeds threshold; epochs may be leaking (OnEpochEnd not firing?)";
      }
    }
    plans = it->second;  // copy shared_ptr; submit outside lock
  }
  for (auto& plan : *plans) {
    const auto& req = plan.req;
    const DecodeTargetLite* picked = plan.picked;
    const std::string& target_server_name = plan.target_server_name;
    const std::vector<void*>& block_ptrs = plan.block_ptrs;
    const uint32_t cached = plan.cached;

    // Skip leading prefix-cached blocks already valid on Decode.
    if (cached >= picked->kv_block_ids.size()) {
      // Full prefix hit: nothing to write; still notify barrier on final layer.
      if (li == deps_.num_layers - 1 && deps_.barrier != nullptr) {
        const bool dispatch_failed = TakeDispatchFailed(static_cast<uint64_t>(req->req_id));
        deps_.barrier->WaitAndNotify(static_cast<uint64_t>(req->req_id), dispatch_failed);
      }
      continue;
    }
    // Coalesce blocks into Mooncake batches (bounded by kMaxEntriesPerBatch).
    constexpr size_t kMaxEntriesPerBatch = 256;
    const size_t to_write = picked->kv_block_ids.size() - static_cast<size_t>(cached);
    std::vector<PdV2ConnectorInterface::KvWriteEntry> entries;
    entries.reserve(std::min(kMaxEntriesPerBatch, to_write));
    size_t submitted = 0;
    bool any_submit_failed = false;

    auto flush_entries = [&]() {
      if (entries.empty()) return;
      uint64_t batch_id = 0;
      Status s = deps_.connector->SubmitWriteBatchAsyncForRequest(target_server_name, entries,
                                                                  static_cast<uint64_t>(req->req_id), &batch_id);
      if (!s.OK()) {
        KLLM_LOG_WARNING << "PdV2KvDispatcher: SubmitWriteBatchAsync failed req=" << req->req_id
                         << " blocks=" << entries.size() << " err=" << s.GetMessage();
        // 任一 sub-batch 提交失败都标记整个请求 dispatch_failed，避免 Decode 读取半写入的 KV。
        any_submit_failed = true;
      } else {
        submitted += entries.size();
      }
      entries.clear();
    };

    for (size_t i = static_cast<size_t>(cached); i < picked->kv_block_ids.size(); ++i) {
      void* source = static_cast<uint8_t*>(block_ptrs[i]) + chunk_byte_offset;
      // Same in-block offset on both sides (identical KV layout).
      const uint64_t target_offset =
          picked->kv_pool_base_va + static_cast<uint64_t>(picked->kv_block_ids[i]) * static_cast<uint64_t>(block_size) +
          static_cast<uint64_t>(chunk_byte_offset);

      entries.push_back({target_offset, source, chunk_bytes});
      if (entries.size() >= kMaxEntriesPerBatch) flush_entries();
    }
    flush_entries();

    if (any_submit_failed) {
      MarkDispatchFailed(static_cast<uint64_t>(req->req_id));
    }
    if (layer_index == 0) {
      KLLM_LOG_DEBUG << "PdV2KvDispatcher[send] req=" << req->req_id << " peer_segment=" << target_server_name
                     << " submitted_blocks=" << submitted << "/" << to_write
                     << " first_dst_block_id=" << picked->kv_block_ids[cached] << " chunk_bytes=" << chunk_bytes;
    }
    KLLM_LOG_DEBUG << "PdV2KvDispatcher: req=" << req->req_id << " layer=" << layer_index
                   << " chunk_start=" << chunk_start << " chunk_size=" << chunk_size << " submitted=" << submitted
                   << "/" << picked->kv_block_ids.size();

    // Final layer: notify barrier (with dispatch_failed if any submit failed).
    if (li == deps_.num_layers - 1 && deps_.barrier != nullptr) {
      const bool dispatch_failed = TakeDispatchFailed(static_cast<uint64_t>(req->req_id));
      if (dispatch_failed) {
        KLLM_LOG_WARNING << "PdV2KvDispatcher: WaitAndNotify with dispatch_failed=true rank=" << rank
                         << " req=" << req->req_id;
      }
      deps_.barrier->WaitAndNotify(static_cast<uint64_t>(req->req_id), dispatch_failed);
    }
  }
  // Final layer: reclaim after all WaitAndNotify calls.
  if (li == deps_.num_layers - 1) {
    std::lock_guard<std::mutex> lk(batch_reqs_mu_);
    dispatch_plans_by_epoch_.erase(epoch_id);
    reqs_by_epoch_.erase(epoch_id);
  }
}

void PdV2KvDispatcher::OnEpochEnd(int rank, uint64_t epoch_id) {
  // kSeal after any queued chunk-end tasks for this epoch (FIFO).
  submitted_tasks_.fetch_add(1, std::memory_order_relaxed);
  submit_queue_.Put(SubmitTask{SubmitTask::Kind::kSeal, rank, epoch_id, -1});
}

void PdV2KvDispatcher::SealEpoch(int rank, uint64_t epoch_id) {
  std::shared_ptr<std::vector<ReqDispatchPlan>> plans;
  {
    std::lock_guard<std::mutex> lk(batch_reqs_mu_);
    reqs_by_epoch_.erase(epoch_id);  // always; OnEpochEnd fires once per BeginBatch
    auto it = dispatch_plans_by_epoch_.find(epoch_id);
    if (it == dispatch_plans_by_epoch_.end()) {
      return;  // already reclaimed on final layer
    }
    plans = std::move(it->second);
    dispatch_plans_by_epoch_.erase(it);
  }

  if (plans->empty()) {
    KLLM_LOG_DEBUG << "PdV2KvDispatcher::OnEpochEnd: epoch=" << epoch_id << " rank=" << rank
                   << " had no dispatch plans to reclaim (non-dispatching or non-final split-fuse chunk) — no-op";
    return;
  }
  // Sealed without final layer — reclaim leftover plans.
  KLLM_LOG_WARNING << "PdV2KvDispatcher::OnEpochEnd reclaiming uncompleted epoch=" << epoch_id << " rank=" << rank
                   << " plans=" << plans->size();
  for (auto& plan : *plans) {
    if (plan.req != nullptr) TakeDispatchFailed(static_cast<uint64_t>(plan.req->req_id));
  }
}

void PdV2KvDispatcher::CaptureEpochReqs(uint64_t epoch_id,
                                        const std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs) {
  if (epoch_id == 0) return;  // non-pd_v2 / BeginBatch no-op
  std::lock_guard<std::mutex> lk(batch_reqs_mu_);
  reqs_by_epoch_[epoch_id] = reqs;  // pin this batch (shared_ptr copy)
}

size_t PdV2KvDispatcher::PendingEpochCountForTest() const {
  std::lock_guard<std::mutex> lk(batch_reqs_mu_);
  return dispatch_plans_by_epoch_.size();
}

bool PdV2KvDispatcher::WaitUntilIdleForTest(int timeout_ms) {
  // submitted_ bumped before Put; processed_ after side effects.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (processed_tasks_.load(std::memory_order_acquire) == submitted_tasks_.load(std::memory_order_relaxed)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return processed_tasks_.load(std::memory_order_acquire) == submitted_tasks_.load(std::memory_order_relaxed);
}

void PdV2KvDispatcher::MarkDispatchFailed(uint64_t req_id) {
  std::lock_guard<std::mutex> lk(dispatch_failed_mu_);
  dispatch_failed_reqs_.insert(req_id);
}

bool PdV2KvDispatcher::TakeDispatchFailed(uint64_t req_id) {
  std::lock_guard<std::mutex> lk(dispatch_failed_mu_);
  auto it = dispatch_failed_reqs_.find(req_id);
  if (it == dispatch_failed_reqs_.end()) {
    return false;
  }
  dispatch_failed_reqs_.erase(it);
  return true;
}

}  // namespace pd_v2
}  // namespace ksana_llm
