/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/prefill/pd_v2_kv_dispatcher.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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
  KLLM_LOG_INFO << "PdV2KvDispatcher initialized num_layers=" << deps_.num_layers << " prefill_tp=" << deps_.prefill_tp
                << " transfer_layer_chunk_size=" << deps_.transfer_layer_chunk_size << " use_mla=" << deps_.use_mla;
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

std::vector<PdV2KvDispatcher::ReqDispatchPlan> PdV2KvDispatcher::BuildDispatchPlans(int rank) const {
  std::vector<ReqDispatchPlan> plans;
  auto live_reqs = deps_.request_manager->SnapshotInferRequests();
  for (auto& req : live_reqs) {
    if (req == nullptr) continue;
    if (req->pd_v2_decode_targets.empty()) continue;
    // Skip steady-state decode reqs (step >= 1). HandleFullPrefixCacheHit
    // (full_cache_hit_handler.h) flips infer_stage from kContext to kDecode
    // when prefill-side prefix cache covers 100% of the input — but step
    // stays 0 (still on the first-and-only forward pass). The cached KV
    // must still be RDMA-written to the Decode peer (decode-side cache may
    // be empty). Filtering only by infer_stage would lose these prefix-hit
    // reqs and deadlock Decode at pd_v2_awaiting_prefill.
    if (req->infer_stage != InferStage::kContext && req->step != 0) continue;

    // ShouldHandleEntry: physical device_id one-to-one mapping (design
    // doc §6). target_device_id = decode_attn_dp_group_id * d_attn_TP +
    // t.decode_attn_tp_rank_in_group; the Prefill rank with my_rank ==
    // target_device_id participates, others skip. Forward-invariant, so
    // resolved here once per epoch instead of on every chunk-end callback.
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
      // Not our turn — each DecodeTarget is handled by exactly one Prefill
      // rank, so this rank emits no barrier report for this req.
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
    // Decode side registers each TP rank's KV pool under a distinct
    // server_name ("<inference_addr>.exec.r<device_id>").
    plan.target_server_name = req->pd_v2_decode_engine_inference_addr + ".exec.r" + std::to_string(target_device_id);
    // Resolve local block GPU pointers. attn_dp_rank_id = my_rank inside
    // the local Prefill DP group; each executor process owns one device,
    // so the matching BlockAllocator slot is at that index.
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
  // Per-layer per-rank: 62 × 8 = ~500 entries per request on DSR1 — DEBUG only.
  KLLM_LOG_DEBUG << "PdV2KvDispatcher::OnLayerComplete rank=" << rank << " epoch=" << epoch_id
                 << " layer=" << layer_index << " is_chunk_end=" << IsChunkEndLayer(static_cast<uint32_t>(layer_index));
  if (deps_.connector == nullptr || deps_.request_manager == nullptr) return;
  if (layer_index < 0 || static_cast<uint32_t>(layer_index) >= deps_.num_layers) return;
  const uint32_t li = static_cast<uint32_t>(layer_index);
  if (!IsChunkEndLayer(li)) return;

  // Connector RegisterKvPool record carries block_size + num_layers; we
  // need block_size to compute target_offset and element_size for chunk
  // math. Values are stable post-Boot — TryGetKvShape caches them (B) so
  // we don't re-query the connector on every chunk-end callback.
  size_t block_size = 0, element_size = 0;
  if (!TryGetKvShape(&block_size, &element_size)) {
    KLLM_LOG_WARNING << "PdV2KvDispatcher: KV pool shape unavailable, skipping layer " << layer_index;
    return;
  }
  const uint32_t chunk_start = ChunkStartOffset(li);
  const uint32_t chunk_size = ChunkSize(li);
  const size_t chunk_bytes = element_size * static_cast<size_t>(chunk_size);
  const size_t chunk_byte_offset = element_size * static_cast<size_t>(chunk_start);

  // Per-epoch dispatch plans. Different batches arrive on different
  // epoch_ids (PdV2LayerEventTracker assigns one slot per BeginBatch). We
  // build plans on the FIRST chunk-end callback for an epoch (resolving
  // routing / server_name / block_ptrs / prefix-cache count once — all
  // forward-invariant) and key them by epoch_id behind a shared_ptr. By
  // the time the layer N-1 callback runs, the next batch's
  // HandlePreprocess has already EraseRequest'd this batch's prefill-only
  // reqs; each plan's strong ref keeps them
  // alive, and epoch-keying keeps batch B's WaitAndNotify on B's plans
  // even when B's layer 26 fires after C's layer 0. The entry is erased on
  // this batch's final layer.
  std::shared_ptr<std::vector<ReqDispatchPlan>> plans;
  {
    std::lock_guard<std::mutex> lk(batch_reqs_mu_);
    auto it = dispatch_plans_by_epoch_.find(epoch_id);
    if (it == dispatch_plans_by_epoch_.end()) {
      auto built = std::make_shared<std::vector<ReqDispatchPlan>>(BuildDispatchPlans(rank));
      it = dispatch_plans_by_epoch_.emplace(epoch_id, std::move(built)).first;
      // Leak guard / metric: in steady state only a handful of epochs are
      // in flight (tracker pipeline depth ~2). A growing map means epochs
      // are not being reclaimed (OnEpochEnd should always seal them) — warn
      // so the leak surfaces in production logs before KV cache exhaustion.
      constexpr size_t kPendingEpochWarnThreshold = 64;
      if (dispatch_plans_by_epoch_.size() >= kPendingEpochWarnThreshold) {
        KLLM_LOG_WARNING << "PdV2KvDispatcher: dispatch_plans_by_epoch_ size=" << dispatch_plans_by_epoch_.size()
                         << " exceeds threshold; epochs may be leaking (OnEpochEnd not firing?)";
      }
    }
    plans = it->second;  // copy the shared_ptr only; submits run outside the lock
  }
  for (auto& plan : *plans) {
    const auto& req = plan.req;
    const DecodeTargetLite* picked = plan.picked;
    const std::string& target_server_name = plan.target_server_name;
    const std::vector<void*>& block_ptrs = plan.block_ptrs;
    const uint32_t cached = plan.cached;

    // pd_v2 prefix-cache aware: skip the leading `cached` blocks (Decode
    // already has valid K/V there from a shared-prefix request). Source
    // (Prefill local pool) and target (Decode pool) indices advance in
    // lockstep, so iteration starts at `cached` instead of 0.
    if (cached >= picked->kv_block_ids.size()) {
      // Entire prompt is prefix-cached on Decode (extreme edge: every
      // block of this request is already valid on Decode). Nothing to
      // RDMA-write — Prefill still ran forward + sampled the next
      // token, which gets returned via PrefillRequestComplete.
      //
      // 即使没有需要 RDMA 写的 block，最终层仍要触发 barrier，否则 Decode 侧会一直等待该 rank 上报。
      if (li == deps_.num_layers - 1 && deps_.barrier != nullptr) {
        const bool dispatch_failed = TakeDispatchFailed(static_cast<uint64_t>(req->req_id));
        deps_.barrier->WaitAndNotify(static_cast<uint64_t>(req->req_id), dispatch_failed);
      }
      continue;
    }
    // A: coalesce every block of this (req, chunk) into ONE Mooncake
    // batch. All blocks share `target_server_name`, so one openSegment +
    // allocateBatchID + submitTransfer replaces the previous per-block
    // submit, and the barrier polls one batch id instead of N. Very long
    // prompts are flushed in bounded sub-batches (kMaxEntriesPerBatch) to
    // keep allocateBatchID sizes sane; the barrier already tracks
    // multiple batch ids per req.
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
      // Source: local block i base + chunk_byte_offset.
      void* source = static_cast<uint8_t*>(block_ptrs[i]) + chunk_byte_offset;
      // Target: remote KV pool base + remote block_id * block_size +
      // chunk_byte_offset (same byte offset within the block since the
      // layout is identical on both sides — invariant).
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
      // One log line per req on layer 0 — captures the (req, peer,
      // block count, first dst_block, length) tuple for verifying RDMA
      // write topology under any model/MHA/MLA combination.
      KLLM_LOG_DEBUG << "PdV2KvDispatcher[send] req=" << req->req_id << " peer_segment=" << target_server_name
                     << " submitted_blocks=" << submitted << "/" << to_write
                     << " first_dst_block_id=" << picked->kv_block_ids[cached] << " chunk_bytes=" << chunk_bytes;
    }
    KLLM_LOG_DEBUG << "PdV2KvDispatcher: req=" << req->req_id << " layer=" << layer_index
                   << " chunk_start=" << chunk_start << " chunk_size=" << chunk_size << " submitted=" << submitted
                   << "/" << picked->kv_block_ids.size();

    // 模型最终层提交完成后，通知 barrier 开始轮询；同时带上提交失败标记，
    // 让 barrier 在任一提交失败时直接以 kFailed 为初始状态。
    if (li == deps_.num_layers - 1 && deps_.barrier != nullptr) {
      const bool dispatch_failed = TakeDispatchFailed(static_cast<uint64_t>(req->req_id));
      if (dispatch_failed) {
        KLLM_LOG_WARNING << "PdV2KvDispatcher: WaitAndNotify with dispatch_failed=true rank=" << rank
                         << " req=" << req->req_id;
      }
      deps_.barrier->WaitAndNotify(static_cast<uint64_t>(req->req_id), dispatch_failed);
    }
  }
  // Drop the captured plans for this epoch so its slot can be reused by a
  // future BeginBatch (the tracker recycles EpochSlot pointers, but
  // epoch_id is monotonic — the map entry would otherwise leak). Done
  // after the loop (and thus after all per-req WaitAndNotify calls) at
  // the model's final layer only.
  if (li == deps_.num_layers - 1) {
    std::lock_guard<std::mutex> lk(batch_reqs_mu_);
    dispatch_plans_by_epoch_.erase(epoch_id);
  }
}

void PdV2KvDispatcher::OnEpochEnd(int rank, uint64_t epoch_id) {
  std::shared_ptr<std::vector<ReqDispatchPlan>> plans;
  {
    std::lock_guard<std::mutex> lk(batch_reqs_mu_);
    auto it = dispatch_plans_by_epoch_.find(epoch_id);
    if (it == dispatch_plans_by_epoch_.end()) {
      // Normal completion path: the final-layer OnLayerComplete already
      // erased this epoch after WaitAndNotify. Nothing to reclaim.
      return;
    }
    plans = std::move(it->second);
    dispatch_plans_by_epoch_.erase(it);
  }
  // Only reached when the epoch was sealed without its final layer firing
  // (Forward aborted, executor short-circuit, or the final-layer event was
  // never recorded). The plans' strong ExecutorInferRequest refs would
  // otherwise pin the requests forever and starve the BlockAllocator. Drop
  // any leftover dispatch-failed marks for these reqs (the take-and-clear in
  // WaitAndNotify never ran for them) before releasing the refs.
  KLLM_LOG_WARNING << "PdV2KvDispatcher::OnEpochEnd reclaiming uncompleted epoch=" << epoch_id << " rank=" << rank
                   << " plans=" << plans->size();
  for (auto& plan : *plans) {
    if (plan.req != nullptr) TakeDispatchFailed(static_cast<uint64_t>(plan.req->req_id));
  }
  // `plans` (and thus the strong req refs) drops here as it goes out of scope.
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
