/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "runtime/threadpool.h"

namespace ksana_llm {

class GlobalCacheConnectorInterface;
class BlockAllocatorManager;
class IpcFuncWrapper;
class ExecutorInferRequest;

// 提交时刻定格的 SWA block 列表。
struct SwaBlockSnapshot {
  std::vector<int> block_ids;
  std::vector<size_t> block_idx_offsets;
};

// 在 executor 端管理异步 KV cache put 操作。
// Forward 完成后调用 SubmitBatchPut：仅对已覆盖完整 input 的最后一段 prefill
// 异步上传 KV；完成后通过 IPC 通知 engine。
class GlobalCachePutWorker {
 public:
  GlobalCachePutWorker(std::shared_ptr<GlobalCacheConnectorInterface> connector,
                       std::shared_ptr<BlockAllocatorManager> block_allocator_manager, int dp_group_id, int device_id,
                       std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper, const std::string& role = "",
                       int fetch_thread_num = 4);
  ~GlobalCachePutWorker();

  void Start();
  void Stop();

  // 注入 SWA block allocator 与稀疏选点参数；未调用时不做 SWA put。
  // swa_tail_block_num（N）：尾部密集存储的 block 个数。
  // swa_front_equal_parts：前段等分份数（默认 5）；首尾两段不传，中间 parts-2 份取中位点。
  void SetSwaConfig(std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager, int swa_tail_block_num,
                    int swa_front_equal_parts = 5);

  // Forward 完成后调用，为 batch 中的每个 prefill 请求提交异步 put 任务。
  // Decode 请求会被跳过。
  void SubmitBatchPut(const std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs);

  // Sampling + MTP 完成后调用，为 batch 中的每个 prefill 请求保存生成的 token(s)。
  // 仅在 prefill/standalone 角色下、仅卡 0 保存。保存 generated_tokens + draft_tokens。
  void SubmitFirstTokensPut(const std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs);

 private:
  // 对单个请求执行 KV cache put（在线程池中异步执行）
  // token_ids 为提交时的快照，避免异步执行时被调度器修改。返回实际 BatchPut 成功的 FA block 数。
  int PutRequestKVCache(const std::shared_ptr<ExecutorInferRequest>& req, const std::vector<int>& token_ids);

  // 在 swa_snapshot 的可见集合内做「尾部 N 密存 + 前段等分中位点稀存」选点后上传。
  // SWA block 未压缩、字节数远大于主 KV block，全量存储会让全局 cache 占用翻倍。
  // 返回实际 BatchPut 成功的 SWA block 数。
  int PutRequestSwaCache(const std::shared_ptr<ExecutorInferRequest>& req, const std::vector<int>& token_ids,
                         const SwaBlockSnapshot& swa_snapshot);

  std::shared_ptr<GlobalCacheConnectorInterface> connector_;
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager_;
  std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager_;
  int swa_tail_block_num_ = 4;
  int swa_front_equal_parts_ = 5;
  int dp_group_id_ = 0;
  int device_id_ = 0;
  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_;
  std::string role_;  // "prefill" / "decode" / ""
  int fetch_thread_num_ = 4;

  std::unique_ptr<ThreadPool> thread_pool_;
  std::atomic<bool> terminated_{false};
};

}  // namespace ksana_llm
