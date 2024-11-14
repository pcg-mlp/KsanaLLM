/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/batch_scheduler/state/batch_state.h"
#include "ksana_llm/cache_manager/cache_manager_interface.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/tokenizer.h"
#include "ksana_llm/utils/stop_checker.h"


namespace ksana_llm {

class BaseScheduleStrategy {
 public:
  BaseScheduleStrategy(const BatchSchedulerConfig &batch_scheduler_config, int tp_num,
                       std::shared_ptr<BatchState> batch_state)
      : batch_state_(batch_state), batch_scheduler_config_(batch_scheduler_config), tp_num_(tp_num) {}

  // Get the next infer reqs that ready to run.
  virtual void Schedule() = 0;

  // Set the cache manager instance of scheduler strategy.
  void SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager);

  void SetTokenizer(std::shared_ptr<Tokenizer> tokenizer);

  std::shared_ptr<CacheManagerInterface>& GetCacheManager() { return cache_manager_; }

 protected:
  // The batch state informations, include some queues and mutexes.
  std::shared_ptr<BatchState> batch_state_ = nullptr;

  // Used to manager kv cache block, auto-batching strategy do not use this.
  std::shared_ptr<CacheManagerInterface> cache_manager_ = nullptr;

  // the config and context.
  BatchSchedulerConfig batch_scheduler_config_;
  int tp_num_;

  // The tokenizer used for encode and decode
  std::shared_ptr<Tokenizer> tokenizer_ = nullptr;

  std::shared_ptr<StopChecker> stop_checker_;
};

}  // namespace ksana_llm
