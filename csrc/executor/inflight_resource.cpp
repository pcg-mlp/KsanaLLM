/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/inflight_resource.h"

namespace ksana_llm {

void InflightResourceManager::RegisterResource(const std::string& name,
                                               std::shared_ptr<InflightResourceInterface> resource) {
  if (!resource) {
    KLLM_LOG_WARNING << "InflightResourceManager: ignoring null registration for name '" << name << "'";
    return;
  }
  KLLM_CHECK_WITH_INFO(resources_.find(name) == resources_.end(),
                       fmt::format("InflightResourceManager: resource '{}' is already registered. "
                                   "Double-registration would silently replace the existing resource and "
                                   "leave any cached raw pointers (e.g. slot_resource_) dangling.",
                                   name));
  resources_[name] = std::move(resource);
}

void InflightResourceManager::ProcessScheduleOutput(const ScheduleOutput& schedule_output) {
  if (resources_.empty()) {
    return;
  }

  // Thread-safety note: Alloc and Free below run in the preprocess thread, while
  // CompressorSlotResource::Get() runs in the execute thread (single H2D stream path)
  // or in the preprocess thread (dual H2D stream path, where PrepareCompressor is called
  // from PrepareModelInputGpu before the model_input is queued).  The two threads can
  // run concurrently, so one might ask whether Free() here can race with a concurrent
  // Get() in the execute thread.
  //
  // It cannot, because of the following cross-component invariant:
  //
  //   finish_req_ids / recompute_req_ids in schedule_output are populated by the
  //   scheduler (SyncStopRequest in continuous_batching.cpp) only AFTER the scheduler
  //   receives the generation result for those requests.  That result is produced by
  //   the sampling stage, which runs only AFTER the execute thread's Forward() for the
  //   corresponding step has completed.  Therefore, by the time this ScheduleOutput
  //   arrives at the preprocess thread and Free() is called for req X, the execute
  //   thread has already finished every Forward() step that used X's slot.
  //
  //   In async-scheduling mode (enable_async_) the scheduler pre-generates SOs, but
  //   TryToLaunchPlannedScheduleOutput and the HasInflightTask guard in
  //   ProcessAsyncStoppedRequest prevent any SO from being dispatched while a request
  //   still has in-flight forward steps.
  //
  // This invariant is NOT enforced by code in this file.  If the scheduler is ever
  // changed to emit finish_req_ids speculatively (before results return), the Free/Get
  // race would become real.  Any such change must be reviewed against this assumption.

  // ---- Free first: release resources held by finished or recomputed requests ----
  // Free must run before Alloc.  When the slot pool is at capacity (all
  // max_batch_size * max_pp_batch_num slots in use) and a schedule output
  // simultaneously marks some requests finished and adds new ones, Alloc for
  // the new requests would exhaust the pool before the finished slots are
  // returned.  Free-first mirrors the original design (EvictCompletedCompressorSlots
  // before Forward → PrepareCompressor) and ensures freed slots are available when
  // Alloc runs.
  // Free() is idempotent (returns early if not in map), so no Has() guard needed.
  // A req appearing in both finish_req_ids and recompute_req_ids is handled safely.
  for (auto& [name, resource] : resources_) {
    for (const auto& per_dp_vec : schedule_output.finish_req_ids) {
      for (const auto req_id : per_dp_vec) {
        resource->Free(req_id);
      }
    }
    for (const auto& per_dp_vec : schedule_output.recompute_req_ids) {
      for (const auto req_id : per_dp_vec) {
        resource->Free(req_id);
      }
    }
  }

  // ---- Alloc second: ensure every running request owns each registered resource ----
  // Alloc() is idempotent (returns early if already allocated), so no Has() guard needed.
  for (auto& [name, resource] : resources_) {
    for (const auto& req : schedule_output.executor_running_reqs) {
      resource->Alloc(req->req_id);
    }
  }
}

}  // namespace ksana_llm
