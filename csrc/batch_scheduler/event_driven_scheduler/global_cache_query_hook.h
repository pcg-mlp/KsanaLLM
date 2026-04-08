/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：全局 cache 查询 Hook。
 * 在 AddRequest 时对请求执行 prefix cache 查找 + block 分配 + 全局 cache 异步查询。
 * ==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/schedule_hook_interface.h"
#include "batch_scheduler/global_cache_fetcher.h"
#include "cache_manager/cache_manager_interface.h"
#include "configure/schedule_config_parser.h"

namespace ksana_llm {

class GlobalCacheQueryHook : public ScheduleHookInterface {
 public:
  // hook_idx: 本 Hook 在 HookManager 中的位置索引，用于构造 AsyncTaskResult
  GlobalCacheQueryHook(std::shared_ptr<GlobalCacheFetcher> fetcher, const BatchSchedulerConfig& config, int hook_idx);
  ~GlobalCacheQueryHook() override = default;

  ReqIdToNextStep Process(const std::vector<InferRequestPtr>& requests, ScheduleContext& context) override;
  ReqIdToNextStep UpdateAsyncResult(const std::vector<AsyncTaskResult>& finished_tasks,
                                    ScheduleContext& context) override;
  void SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback callback) override;

 private:
  // 对单个请求执行 prefix cache 查找 + block 分配。
  // 成功返回 true，失败（如 block 不足）返回 false。
  bool PrepareRequest(InferRequestPtr& req, std::shared_ptr<CacheManagerInterface> cache_manager);

  std::shared_ptr<GlobalCacheFetcher> fetcher_;
  BatchSchedulerConfig config_;
  int hook_idx_ = 0;
  AsyncTaskFinishedCallback async_callback_;
};

}  // namespace ksana_llm
