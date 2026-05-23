/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：全局 cache 查询 Hook 实现。
 * ==============================================================================*/

#include "batch_scheduler/event_driven_scheduler/global_cache_query_hook.h"

#include "batch_scheduler/full_cache_hit_handler.h"
#include "configure/environment.h"
#include "profiler/reporter.h"
#include "utils/logger.h"
#include "utils/singleton.h"
#include "utils/string_utils.h"

namespace ksana_llm {

GlobalCacheQueryHook::GlobalCacheQueryHook(std::shared_ptr<GlobalCacheFetcher> fetcher,
                                           const BatchSchedulerConfig& config, int hook_idx)
    : fetcher_(std::move(fetcher)), config_(config), hook_idx_(hook_idx) {
  KLLM_CHECK(fetcher_ != nullptr);
}

void GlobalCacheQueryHook::SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback callback) {
  async_callback_ = std::move(callback);

  // 注册 GlobalCacheFetcher 的完成回调，将其桥接到事件驱动调度器的 AsyncTaskFinished 机制
  fetcher_->SetCompletionCallback([this](int64_t req_id, bool success) {
    if (!async_callback_) return;

    EventAsyncTaskFinishedInfo info;
    AsyncTaskResult task_result;
    task_result.hook_idx = hook_idx_;
    task_result.task_id = 0;
    task_result.task_ret_state = success ? 0 : 1;
    info.task_results.push_back(task_result);
    async_callback_(info);

    KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: completion callback fired, req_id=" << req_id
                       << ", success=" << success;
  });
}

bool GlobalCacheQueryHook::PrepareRequest(InferRequestPtr& req, std::shared_ptr<CacheManagerInterface> cache_manager) {
  req->cache_manager = cache_manager;
  req->SetPlanningWorkload(req->GetRemainingWorkload());

  size_t seq_token_num = req->GetPlanningSequenceLen();
  ScheduleTaskWorkload planning_workload = req->GetPlanningWorkload();

  // Prefix cache 查找（仅首次 prefill，prefill_start_offset == 0）
  size_t shared_block_num = 0;
  size_t req_step_block_num = 0;
  size_t shared_token_num = 0;

  if (planning_workload.prefill_start_offset == 0) {
    cache_manager->GetRequestPrefixBlockNumber(req->req_id, req->GetPrefillingTokens(), seq_token_num, shared_block_num,
                                               req_step_block_num, shared_token_num);
    if (shared_token_num > 0) {
      REPORT_COUNTER("prefix_cache_hit_req_num", static_cast<size_t>(1));
      REPORT_COUNTER("prefix_cache_hit_token_num", shared_token_num);
      REPORT_COUNTER("prefix_cache_hit_block_num", shared_block_num);
      KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: req_id=" << req->req_id
                         << " find shared_block_num=" << shared_block_num << ", shared_token_num=" << shared_token_num;
    }
  }

  // 计算需要的 block 数量
  req_step_block_num = cache_manager->GetRequestStepBlockNumber(req->req_id, seq_token_num);

  RuntimeConfig runtime_config;
  Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
  req_step_block_num += (runtime_config.mtp_step_num + req->block_token_num - 1) / req->block_token_num;

  KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: req_id=" << req->req_id << ", seq_token_num=" << seq_token_num
                     << ", req_step_block_num=" << req_step_block_num << ", shared_token_num=" << shared_token_num;

  // 分配 block
  Status status = cache_manager->AllocateRequestBlocks(req->req_id, req_step_block_num, req->kv_cache_blocks,
                                                       req->swa_kv_cache_blocks, req->swa_kv_cache_block_idx_offsets);
  if (!status.OK()) {
    KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: AllocateRequestBlocks failed for req_id=" << req->req_id
                       << ", msg=" << status.GetMessage();
    return false;
  }

  // 根据本地 prefix cache 命中更新 workload
  if (shared_token_num > 0) {
    KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: local cache match for req_id=" << req->req_id
                       << ", shared_token_num=" << shared_token_num << ", seq_token_num=" << seq_token_num;

    if (shared_token_num >= seq_token_num) {
      HandleFullPrefixCacheHit(req, planning_workload, seq_token_num);
    } else {
      // 部分命中：调整 offset
      size_t planning_prefill_token_num = planning_workload.prefill_token_num + planning_workload.prefill_start_offset;
      KLLM_CHECK_WITH_INFO(planning_prefill_token_num >= shared_token_num,
                           FormatStr("req_id=%ld planning_total(%zu) < shared(%zu)", req->req_id,
                                     planning_prefill_token_num, shared_token_num));
      planning_workload.prefill_start_offset = shared_token_num;
      planning_workload.prefill_token_num = planning_prefill_token_num - shared_token_num;
      req->SetPlanningWorkload(planning_workload);

      ScheduleTaskWorkload remaining_workload = req->GetRemainingWorkload();
      size_t prefill_token_num = remaining_workload.prefill_token_num + remaining_workload.prefill_start_offset;
      KLLM_CHECK_WITH_INFO(
          prefill_token_num >= shared_token_num,
          FormatStr("req_id=%ld remaining_total(%zu) < shared(%zu)", req->req_id, prefill_token_num, shared_token_num));
      remaining_workload.prefill_start_offset = shared_token_num;
      remaining_workload.prefill_token_num = prefill_token_num - shared_token_num;
      req->SetRemainingWorkload(remaining_workload);
    }
  }

  // 注意：FlexibleCache 更新由 backend 的 ProcessPrefillingQueue 统一完成，
  // 此处不能重复调用 UpdateFlexibleCache，否则 flexible_cached_copy_tasks 会被污染。

  // 设置 cache 命中状态
  req->SetCacheHitStatus(shared_token_num, /*is_first_prefill_step=*/true);

  return true;
}

ReqIdToNextStep GlobalCacheQueryHook::Process(const std::vector<InferRequestPtr>& requests, ScheduleContext& context) {
  ReqIdToNextStep result;
  if (requests.empty()) {
    return result;
  }

  // 对每个请求：prefix cache + block 分配 + 提交全局 cache 查询
  for (auto req : requests) {
    const size_t dp_idx = static_cast<size_t>(req->attn_dp_group_id);
    auto cache_manager = context.GetDpCacheManager(dp_idx);

    bool prepared = PrepareRequest(req, cache_manager);
    if (!prepared) {
      // Block 分配失败，回退到 backend 正常处理（prefix cache + alloc 在 Schedule 中完成）
      KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: prepare failed for req_id=" << req->req_id
                         << ", fallback to backend";
      result[req->req_id] = NextStep{NextStepAction::kContinue, 0};
      continue;
    }

    // 提交到 GlobalCacheFetcher（如果有 IPC 则异步，否则同步完成）
    fetcher_->SubmitRequest(req);
    KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: req_id=" << req->req_id << " submitted to GlobalCacheFetcher";
  }

  // 同步模式（无 IPC）：SubmitRequest 会立即完成，从 completed/rejected 队列取结果。
  // 异步模式（有 IPC）：SubmitRequest 只是提交到线程池，完成通过 CompletionCallback →
  //   UpdateAsyncResult 传递。此处不能调用 TakeCompletedRequests，否则会"偷走"
  //   前一批次异步完成但尚未被 UpdateAsyncResult 消费的请求，导致请求永远卡住。
  if (!fetcher_->IsAsync()) {
    auto completed = fetcher_->TakeCompletedRequests();
    for (auto& req : completed) {
      KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: req_id=" << req->req_id << " completed synchronously";
      result[req->req_id] = NextStep{NextStepAction::kContinue, 0};
    }

    auto rejected = fetcher_->TakeRejectedRequests();
    for (auto& req : rejected) {
      KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: req_id=" << req->req_id << " rejected synchronously";
      result[req->req_id] = NextStep{NextStepAction::kFailAndEnd, 0};
    }
  }

  // 尚未出现在 result 中的请求需要等待异步结果（异步模式下所有已提交的请求都走这里）
  for (auto& req : requests) {
    if (result.find(req->req_id) == result.end()) {
      result[req->req_id] = NextStep{NextStepAction::kWaitAsyncResult, 0};
    }
  }

  return result;
}

ReqIdToNextStep GlobalCacheQueryHook::UpdateAsyncResult(const std::vector<AsyncTaskResult>& finished_tasks,
                                                        ScheduleContext& context) {
  ReqIdToNextStep result;
  if (finished_tasks.empty()) {
    return result;
  }

  // 从 fetcher 取出完成的请求（workload 已由 fetcher 更新）
  auto completed = fetcher_->TakeCompletedRequests();
  for (auto& req : completed) {
    KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: async completed req_id=" << req->req_id;
    result[req->req_id] = NextStep{NextStepAction::kContinue, 0};
  }

  // 被拒绝的请求（如 decode 角色未全命中）
  auto rejected = fetcher_->TakeRejectedRequests();
  for (auto& req : rejected) {
    KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: async rejected req_id=" << req->req_id;
    result[req->req_id] = NextStep{NextStepAction::kFailAndEnd, 0};
  }

  return result;
}

}  // namespace ksana_llm
