/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：全局 cache 查询 Hook 实现。
 * ==============================================================================*/

#include "batch_scheduler/event_driven_scheduler/global_cache_query_hook.h"

#include "configure/environment.h"
#include "profiler/reporter.h"
#include "utils/logger.h"
#include "utils/singleton.h"
#include "utils/string_utils.h"

namespace ksana_llm {

GlobalCacheQueryHook::GlobalCacheQueryHook(std::shared_ptr<GlobalCacheFetcher> fetcher,
                                           const BatchSchedulerConfig& config, int hook_idx,
                                           bool defer_admission_on_block_full)
    : fetcher_(std::move(fetcher)),
      config_(config),
      hook_idx_(hook_idx),
      defer_admission_on_block_full_(defer_admission_on_block_full) {
  KLLM_CHECK(fetcher_ != nullptr);
  max_inflight_reqs_ = 2 * std::max<size_t>(config.max_batch_size, 1);
}

size_t GlobalCacheQueryHook::DeferredCountForTest() const {
  std::lock_guard<std::mutex> lk(pending_mu_);
  return pending_admission_.size();
}

void GlobalCacheQueryHook::OnEvent(const HookEvent& event) {
  switch (event.type) {
    case HookEventType::kBlocksFreed:
      WakeDeferredAdmission();
      break;
    case HookEventType::kRequestExited:
      ReleaseInflightSlot(event.req_id);
      break;
    default:
      break;
  }
}

void GlobalCacheQueryHook::WakeDeferredAdmission() {
  {
    std::lock_guard<std::mutex> lk(pending_mu_);
    if (pending_admission_.empty()) {
      return;
    }
  }
  if (!async_callback_) {
    return;
  }
  // Sentinel: task_id = -1 marks "not a real async task, just a wake signal".
  // UpdateAsyncResult ignores the AsyncTaskResult payload and sweeps its own
  // pending_admission_ queue by re-running PrepareRequest; this is purely a
  // transport to hop the retry onto the EDS thread.
  EventAsyncTaskFinishedInfo info;
  AsyncTaskResult tr;
  tr.hook_idx = hook_idx_;
  tr.task_id = -1;
  tr.task_ret_state = 0;
  info.task_results.push_back(tr);
  async_callback_(info);
}

bool GlobalCacheQueryHook::IsInflightCapExceeded() const {
  if (!defer_admission_on_block_full_) {
    return false;
  }
  std::lock_guard<std::mutex> lk(pending_mu_);
  return in_flight_admitted_.size() >= max_inflight_reqs_;
}

void GlobalCacheQueryHook::ReleaseInflightSlot(int64_t req_id) {
  if (!defer_admission_on_block_full_) {
    return;
  }
  bool erased = false;
  {
    std::lock_guard<std::mutex> lk(pending_mu_);
    erased = in_flight_admitted_.erase(req_id) > 0;
  }
  if (erased) {
    // 释放了一个 in-flight 名额: 唤醒 pending_admission_ 重试 (可能只是 cap 把它们挡住,
    // blocks 其实够)。复用 kBlocksFreed 的 sentinel 投递 (pending 为空时 no-op);
    // 与调用方在批末另发的 kBlocksFreed 事件叠加只是多扫一次, 幂等无害。
    WakeDeferredAdmission();
  }
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
      // Single source of truth: the Request member. Works even when req_ctx is null (pd_v2
      // Prefill). KsanaPythonOutput / StreamingIterator read it back for the access log.
      req->prefix_cache_hit_tokens = static_cast<int>(shared_token_num);
    }
  }

  // 计算需要的 block 数量。mtp_step_num 单独透传给 cache_manager，由 cache_manager 统一处理
  // next-sampler 与 MTP lookahead 的 block 预留。hook fast-path 与 backend fallback 走同一入口，
  // 避免 pd_v2 Decode / Prefill 两侧 block 计数不一致。
  RuntimeConfig runtime_config;
  Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
  req_step_block_num =
      cache_manager->GetRequestStepBlockNumber(req->req_id, seq_token_num, runtime_config.mtp_step_num);

  KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: req_id=" << req->req_id << ", seq_token_num=" << seq_token_num
                     << ", req_step_block_num=" << req_step_block_num << ", shared_token_num=" << shared_token_num;

  // pd_v2 Decode in-flight 并发上限在 Process / UpdateAsyncResult 入口预检。
  Status status = cache_manager->AllocateRequestBlocks(req->req_id, req_step_block_num, req->kv_cache_blocks,
                                                       req->swa_kv_cache_blocks, req->swa_kv_cache_block_idx_offsets,
                                                       req->mamba_kv_cache_block_id);
  if (!status.OK()) {
    KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: AllocateRequestBlocks failed for req_id=" << req->req_id
                       << ", msg=" << status.GetMessage();
    return false;
  }
  // T1 (pd_v2 Decode access log): KV blocks allocated = admission done. The arrive→here gap is
  // the admission / block-quota wait, where a starved Decode pool stalls disaggregated requests.
  StampOnce(req->pd_v2.blocks_allocated_time_us);

  // 准入成功 (本 req 已占住 KV blocks): 计入 in-flight 集合, 直到 kRequestExited 事件释放。
  // 抢占重启会先 ReleaseInflightSlot(erase) 再经本路径重新准入(insert), 故 net 一致; insert
  // 幂等, 对任何重复准入也安全。仅 pd_v2 Decode (defer_admission_on_block_full_) 启用。
  if (defer_admission_on_block_full_) {
    {
      std::lock_guard<std::mutex> lk(pending_mu_);
      in_flight_admitted_.insert(req->req_id);
    }
    // Bind the admission-slot release to InferRequest destruction so the slot
    // is freed on EVERY exit path — including the ones that bypass the EDS
    // terminal handlers' kRequestExited event and would otherwise leak the slot:
    //   - schedule_hook_manager.cpp: a stale async finish whose req already
    //     left req_id_to_req_ptr hits find_preprocessing_req()==nullptr and
    //     `continue`s ("Invalid async finished state");
    //   - HandleStopRequest / HandleHookFailAndEnd early-return on
    //     req->IsStopped() before reaching their kRequestExited event.
    // Under abort / KV-exhaustion churn these leaked in_flight_admitted_ until
    // it pinned at max_inflight_reqs_, permanently wedging admission. The
    // InferRequest is always destroyed exactly once when its last owner drops,
    // so this is the single guaranteed release point. Idempotent with the
    // explicit kRequestExited events (erase of an absent req_id is a no-op),
    // and idempotent across preempt-restart (re-admit re-inserts + re-binds).
    const int64_t admitted_id = req->req_id;
    req->AddOnDestroy([this, admitted_id]() { ReleaseInflightSlot(admitted_id); });
  }

  // 根据本地 prefix cache 命中更新 workload
  if (shared_token_num > 0) {
    KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: local cache match for req_id=" << req->req_id
                       << ", shared_token_num=" << shared_token_num << ", seq_token_num=" << seq_token_num;

    // Prefix 查询硬上限 seq-1，shared_token_num 恒小于 seq_token_num，不再走 FullHit 回退。
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

    if (IsInflightCapExceeded()) {
      KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: admission gate defer (inflight cap) req_id=" << req->req_id;
      {
        std::lock_guard<std::mutex> lk(pending_mu_);
        pending_admission_.push_back(req);
      }
      result[req->req_id] = NextStep{NextStepAction::kWaitAsyncResult, 0};
      continue;
    }

    bool prepared = PrepareRequest(req, cache_manager);
    if (!prepared) {
      if (defer_admission_on_block_full_) {
        // pd_v2 Decode: 本 Hook 是 block 分配的单一所有者, 不能回退到 backend (下游
        // PdV2DecodeHook 假定 req 已经有 block)。挂入 pending_admission_ 等 block 释放后
        // 由 UpdateAsyncResult 重试完整的 PrepareRequest (re-match + alloc)。返回
        // kWaitAsyncResult 让 req 停在 preprocessing_reqs。
        KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: alloc failed for req_id=" << req->req_id
                           << ", deferring admission and waiting for blocks freed";
        {
          std::lock_guard<std::mutex> lk(pending_mu_);
          pending_admission_.push_back(req);
        }
        result[req->req_id] = NextStep{NextStepAction::kWaitAsyncResult, 0};
        continue;
      }
      // V1 / 非 pd_v2: 回退到 backend 正常处理（prefix cache + alloc 在 Schedule 中完成）
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

  // pd_v2 Decode: 先 drain deferred admission。每一次唤醒 (真正的 fetcher 完成回调, 或
  // kBlocksFreed sentinel) 都是一次重试 alloc 的机会。
  if (defer_admission_on_block_full_) {
    std::deque<InferRequestPtr> drained;
    {
      std::lock_guard<std::mutex> lk(pending_mu_);
      drained.swap(pending_admission_);
    }

    // 尽力调度: 某个 req alloc 失败 (pool 仍满) 不阻塞后续 req, 仅把失败的 req 按原序押回
    // pending_admission_ 队尾, 继续尝试 drained 中剩余的 req, 让 pending_admission_ 尽可能被调度。
    for (auto& req : drained) {
      if (IsInflightCapExceeded()) {
        std::lock_guard<std::mutex> lk(pending_mu_);
        pending_admission_.push_back(std::move(req));
        continue;
      }

      const size_t dp_idx = static_cast<size_t>(req->attn_dp_group_id);
      auto cache_manager = context.GetDpCacheManager(dp_idx);

      // 完整重走 prefix-match + alloc。GetRequestPrefixBlockNumber 内置 re-entry 校验
      // 会丢弃在 deferral 窗口里被驱逐/复用的 stale block, 因此重试一定是在干净一致的
      // cached_blocks 上做分配 (这正是旧 PdV2DecodeHook 直接调 AllocateRequestBlocks 时
      // 跳过的校验)。
      bool prepared = PrepareRequest(req, cache_manager);
      if (!prepared) {
        KLLM_LOG_SCHEDULER << "pending admission retry alloc failed, req_id=" << req->req_id;
        std::lock_guard<std::mutex> lk(pending_mu_);
        pending_admission_.push_back(std::move(req));
        continue;
      }

      // alloc 成功: 走和 Process 一样的 fetcher 流程。同步 fetcher 下 SubmitRequest 立即完成,
      // 紧接着下面的 TakeCompletedRequests 会取到它 → kContinue; 异步 fetcher 下它会在后续
      // 某次 UpdateAsyncResult 里通过 TakeCompletedRequests 回来。两种情况下 req 都停在
      // preprocessing_reqs 等待, 这里不主动给 result entry。
      fetcher_->SubmitRequest(req);
      KLLM_LOG_SCHEDULER << "GlobalCacheQueryHook: deferred req_id=" << req->req_id
                         << " admitted (alloc retry success), submitted to GlobalCacheFetcher";
    }
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
