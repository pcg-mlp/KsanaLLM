/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：全局 cache 查询 Hook。
 * 在 AddRequest 时对请求执行 prefix cache 查找 + block 分配 + 全局 cache 异步查询。
 * ==============================================================================*/
#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/schedule_hook_interface.h"
#include "batch_scheduler/global_cache_fetcher.h"
#include "cache_manager/cache_manager_interface.h"
#include "configure/schedule_config_parser.h"

namespace ksana_llm {

class GlobalCacheQueryHook : public ScheduleHookInterface {
 public:
  // hook_idx: 本 Hook 在 HookManager 中的位置索引，用于构造 AsyncTaskResult。
  // defer_admission_on_block_full: 当 AllocateRequestBlocks 因 KV pool 满而失败时的策略。
  //   false (默认, V1 / 非 pd_v2): 返回 kContinue 回退到 backend, 由 ContinuousBatchingBackend
  //     在 Schedule 中自行 prefix-match + alloc, 维持原有 admission 语义。
  //   true (pd_v2 Decode): GlobalCacheQueryHook 是 KV block 分配的单一所有者, 后继
  //     PdV2DecodeHook 假定 req 进来时已经分好 block, 不能回退到 backend。alloc 失败时把 req
  //     挂入本 Hook 自己的 pending_admission_ 队列并返回 kWaitAsyncResult; 等 EDS 释放 block
  //     触发 kBlocksFreed 事件后, UpdateAsyncResult 重新走完整的 PrepareRequest (re-match +
  //     alloc) 重试。这样把 "分配 + 背压" 完整闭环在本 Hook 内, 避免下游 Hook 拿到空 block 列表
  //     或在 stale 的 cached_blocks 上重复分配 (后者会造成 free pool 双 free / cached_blocks
  //     指针重复, 进而 prefix tree 成环)。
  GlobalCacheQueryHook(std::shared_ptr<GlobalCacheFetcher> fetcher, const BatchSchedulerConfig& config, int hook_idx,
                       bool defer_admission_on_block_full = false);
  ~GlobalCacheQueryHook() override = default;

  ReqIdToNextStep Process(const std::vector<InferRequestPtr>& requests, ScheduleContext& context) override;
  ReqIdToNextStep UpdateAsyncResult(const std::vector<AsyncTaskResult>& finished_tasks,
                                    ScheduleContext& context) override;
  void SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback callback) override;

  // 通用事件通知入口: 按事件类型分派到内部处理逻辑, 只处理本 Hook 关心的事件:
  //   - kBlocksFreed  -> WakeDeferredAdmission()
  //   - kRequestExited -> ReleaseInflightSlot(req_id)
  // 其他事件类型忽略。
  void OnEvent(const HookEvent& event) override;

  // For tests: number of requests currently parked waiting for blocks.
  size_t DeferredCountForTest() const;

 private:
  // 对单个请求执行 prefix cache 查找 + block 分配。
  // 成功返回 true，失败（如 block 不足）返回 false。
  bool PrepareRequest(InferRequestPtr& req, std::shared_ptr<CacheManagerInterface> cache_manager);

  // kBlocksFreed 事件处理: EDS 释放 KV block 后 (HandleStopRequest / HandleRecomputeRequest)
  // 唤醒本 Hook 重试 deferred admission。投递一个 sentinel EventAsyncTaskFinishedInfo, 使 EDS
  // 主循环在自己的线程上调度 UpdateAsyncResult, 真正的 alloc 重试发生在那里。pending_admission_
  // 为空或 async_callback_ 尚未接线时为 no-op。
  void WakeDeferredAdmission();

  bool IsInflightCapExceeded() const;

  // kRequestExited 事件处理: pd_v2 Decode 下某个已准入的 req 彻底离场 (HandleStopRequest /
  // HandleHookFailAndEnd / HandlePdV2DecodeRestart 释放其本地 KV blocks) 时被调用, 从
  // in_flight_admitted_ 集合移除该 req_id, 给后续 admission 腾出名额。按 req_id erase, 幂等
  // (不在集合中即 no-op), 因此对重复离场 / 从未准入的 req / 抢占重启 (释放后再经 gate 重新
  // 准入) 都安全, 计数永不泄漏。
  void ReleaseInflightSlot(int64_t req_id);

  std::shared_ptr<GlobalCacheFetcher> fetcher_;
  BatchSchedulerConfig config_;
  int hook_idx_ = 0;
  bool defer_admission_on_block_full_ = false;
  AsyncTaskFinishedCallback async_callback_;

  // pd_v2 Decode in-flight 并发上限 (gated by defer_admission_on_block_full_)。
  // 现象 (warm-cache decode 块池死锁): decode 把每个 admitted req
  // 立刻 ship 去 prefill 并预占整段 input 的 KV 块 (~278), prefill 是 compute-bound
  // (~1-2 req/s), 于是 awaiting-prefill backlog 无界增长 → 预占块把池抽干 → 已返回的
  // decoding req 取 1 个增量块都 ALLOC_FAIL → 吞吐塌缩。block-usable gate (上面那个)
  // 只挡单次 alloc, 不 bound backlog 规模。这里用一个 "已准入未离场" 的 req 计数硬上限
  // 把 in-flight (awaiting-prefill + decoding) 压到 max_inflight_reqs_:
  //   decoding 部分天然 ≤ max_batch_size (batch 槽数), 故 awaiting-prefill backlog
  //   被 bound 到约 max_batch_size, 既能喂饱 decode batch (流水线深度足够), 又给池
  //   留下大量增量块 headroom (~max_batch_size 个 req 的预占 << 整池)。
  // 取 2 * max_batch_size: 1× 给在跑的 decode set, 1× buffer 给在飞 awaiting-prefill,
  // 不引入新配置项 (复用 max_batch_size)。
  size_t max_inflight_reqs_ = 0;
  // 已准入 (AllocateRequestBlocks 成功) 但尚未离场的 req_id 集合。用 set 而非裸计数:
  // ReleaseInflightSlot 按 req_id erase 天然幂等, 对重复离场/抢占重启/从未准入的 req 都安全,
  // 计数不会泄漏成永久卡死。受 pending_mu_ 保护 (与 pending_admission_ 同锁, 临界区极小)。
  std::unordered_set<int64_t> in_flight_admitted_;

  // Deferred admission queue (pd_v2 Decode only, gated by
  // defer_admission_on_block_full_). Holds a strong ref so the request stays
  // alive while parked. FIFO: drained in arrival order on every wake-up.
  mutable std::mutex pending_mu_;
  std::deque<InferRequestPtr> pending_admission_;
};

}  // namespace ksana_llm
