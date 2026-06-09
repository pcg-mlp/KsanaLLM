/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "batch_scheduler/state/batch_state.h"
#include "batch_scheduler/state/scheduler_shared_counter.h"
#include "batch_scheduler/state/scheduler_tick_tok.h"
#include "cache_manager/cache_manager_interface.h"
#include "runtime/infer_request.h"
#include "utils/stop_checker.h"
#include "utils/tokenizer.h"

namespace ksana_llm {

class TransferEngine;
class GlobalCachePutTracker;

class BaseScheduleStrategy {
 public:
  explicit BaseScheduleStrategy(const BatchSchedulerConfig& batch_scheduler_config, const RuntimeConfig& runtime_config)
      : batch_scheduler_config_(batch_scheduler_config), runtime_config_(runtime_config) {}

  virtual void UpdateRunningRequests(const std::vector<std::shared_ptr<InferRequest>>& running_reqs) = 0;

  // Get the next infer reqs that ready to run.
  virtual void Schedule(std::vector<std::shared_ptr<InferRequest>>& waiting_reqs) = 0;

  void SetBatchState(std::shared_ptr<BatchState> batch_state);

  // Set the cache manager instance of scheduler strategy.
  void SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager);

  void SetSharedCounter(std::shared_ptr<SchedulerSharedCounter> scheduler_shared_counter);
  void SetSchedulerTickTok(std::shared_ptr<SchedulerTickTok> scheduler_ticktok);
  void SetDataParaGroupId(size_t dp_group_id);
  void SetTransferEngine(std::shared_ptr<TransferEngine> transfer_engine);
  void SetGlobalCachePutTracker(std::shared_ptr<GlobalCachePutTracker> tracker);

  std::shared_ptr<CacheManagerInterface>& GetCacheManager() { return cache_manager_; }

 protected:
  // The batch state informations, include some queues and mutexes.
  std::shared_ptr<BatchState> batch_state_ = nullptr;

  // Used to manager kv cache block, auto-batching strategy do not use this.
  std::shared_ptr<CacheManagerInterface> cache_manager_ = nullptr;

  // the config and context.
  BatchSchedulerConfig batch_scheduler_config_;

  RuntimeConfig runtime_config_;

  std::shared_ptr<StopChecker> stop_checker_;

  std::shared_ptr<SchedulerSharedCounter> scheduler_shared_counter_ = nullptr;
  std::shared_ptr<SchedulerTickTok> scheduler_ticktok_ = nullptr;

  // Transfer engine for disaggregated prefill-decode mode
  std::shared_ptr<TransferEngine> transfer_engine_ = nullptr;

  // 全局 cache put 追踪器（延迟析构）
  std::shared_ptr<GlobalCachePutTracker> global_cache_put_tracker_ = nullptr;

  // The dp group id.
  size_t dp_group_id_;
};

}  // namespace ksana_llm
