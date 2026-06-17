/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/schedule_output.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

namespace ksana_llm {

// Non-template base class for polymorphic storage of heterogeneous resource types.
// Alloc/Free/Has depend only on req_id (int64_t), independent of the resource value
// type, so ProcessScheduleOutput iterates through all registered resources without
// knowing their concrete types.
class InflightResourceInterface {
 public:
  virtual ~InflightResourceInterface() = default;

  // Allocate a resource for the given request. Called in preprocess thread (unique writer).
  virtual void Alloc(int64_t req_id) = 0;

  // Release the resource held by the given request. Called in preprocess thread (unique writer).
  virtual void Free(int64_t req_id) = 0;

  // Check whether the given request already holds this resource.
  virtual bool Has(int64_t req_id) = 0;

  // One-time initialization after the resource pool size is known.
  virtual void Init(const RuntimeConfig& runtime_config) = 0;
};

// Typed intermediate layer that adds type-safe read access (Get) with built-in
// shared_mutex protection. Concrete resource implementations inherit from this
// class and override the pure-virtual DoGetUnderLock.
template <typename T>
class BaseInflightResource : public InflightResourceInterface {
 public:
  // Thread-safe read access. Acquires a shared_lock so that concurrent reads
  // do not block each other, only writes (Alloc/Free) are exclusive.
  T Get(int64_t req_id) {
    auto lock = LockForRead();
    return DoGetUnderLock(req_id);
  }

  // Lock-free read for the execute thread when the scheduler invariant holds.
  //
  // Safety contract: caller must guarantee that no concurrent Alloc/Free can
  // run for req_id at the time of this call.  This is satisfied because
  // finish_req_ids / recompute_req_ids are only populated by the scheduler
  // AFTER the generation result for those requests is returned — which means
  // the execute thread's Forward() for the step that used req_id has already
  // completed before Free(req_id) is ever called from the preprocess thread.
  // See the thread-safety note in ProcessScheduleOutput for details.
  //
  // Do NOT call this method from the preprocess thread or from any context
  // where the above invariant cannot be guaranteed.
  T GetUnderSchedulerInvariant(int64_t req_id) { return DoGetUnderLock(req_id); }

 protected:
  // Called by Get() under shared_lock. Implementations must NOT acquire locks
  // themselves.
  virtual T DoGetUnderLock(int64_t req_id) = 0;

  // RAII helpers for subclasses. Use these in Alloc/Free/Has/Destroy instead
  // of constructing manual lock guards; the returned objects guarantee that
  // the lock is held for the full scope and cannot be silently discarded.
  [[nodiscard]] std::shared_lock<std::shared_mutex> LockForRead() const {
    return std::shared_lock<std::shared_mutex>(mutex_);
  }
  [[nodiscard]] std::unique_lock<std::shared_mutex> LockForWrite() {
    return std::unique_lock<std::shared_mutex>(mutex_);
  }

 private:
  // Each resource instance has its own mutex; the manager does not participate
  // in locking.  Kept private so subclasses can only acquire the lock through
  // LockForRead / LockForWrite, guaranteeing a consistent locking discipline.
  mutable std::shared_mutex mutex_;
};

// Central registry and lifecycle manager for per-request inflight resources.
// Owned by ExecutorRuntime. Registration happens via BaseModel overrides before
// the preprocess/execute threads start. ProcessScheduleOutput is the unified
// entry point for allocation and release, called from the preprocess thread.
class InflightResourceManager {
 public:
  InflightResourceManager() = default;
  ~InflightResourceManager() = default;

  // Register a named resource. Must be called before any ProcessScheduleOutput.
  // All resources share the same req_id space.
  void RegisterResource(const std::string& name, std::shared_ptr<InflightResourceInterface> resource);

  // Unified alloc/free pass driven by one schedule_output.
  // - Alloc: for every request in executor_running_reqs that does not yet hold
  //   each registered resource.
  // - Free:  for every request id in finish_req_ids and recompute_req_ids.
  void ProcessScheduleOutput(const ScheduleOutput& schedule_output);

  // Retrieve a registered resource by name and cast to the requested type T.
  // T must be a concrete subclass of InflightResourceInterface (e.g.
  // CompressorSlotResource). Returns nullptr when no resource is registered
  // under the given name.
  template <typename T>
  T* GetTypedResource(const std::string& name) {
    auto it = resources_.find(name);
    if (it == resources_.end()) {
      return nullptr;
    }
    // First attempt: direct dynamic_cast to the concrete class.
    if (auto* p = dynamic_cast<T*>(it->second.get())) {
      return p;
    }
    // Fallback: cast through the typed intermediate to recover the value.
    KLLM_LOG_ERROR << "Resource '" << name << "' does not match the requested type";
    return nullptr;
  }

 private:
  // Named resources, stored as the non-template base for uniform iteration.
  std::unordered_map<std::string, std::shared_ptr<InflightResourceInterface>> resources_;
};

}  // namespace ksana_llm
