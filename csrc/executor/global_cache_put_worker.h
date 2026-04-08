/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "runtime/threadpool.h"

namespace ksana_llm {

class GlobalCacheConnectorInterface;
class BlockAllocatorManager;
class IpcFuncWrapper;
class ExecutorInferRequest;

// 在 executor 端管理异步 KV cache put 操作。
// Forward 完成后调用 SubmitBatchPut，将每个 prefill 请求的 KV cache
// 异步上传到全局存储，完成后通过 IPC 通知 engine。
class GlobalCachePutWorker {
 public:
  GlobalCachePutWorker(std::shared_ptr<GlobalCacheConnectorInterface> connector,
                       std::shared_ptr<BlockAllocatorManager> block_allocator_manager, int dp_group_id, int device_id,
                       std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper, const std::string& role = "");
  ~GlobalCachePutWorker();

  void Start();
  void Stop();

  // Forward 完成后调用，为 batch 中的每个 prefill 请求提交异步 put 任务。
  // Decode 请求会被跳过。
  void SubmitBatchPut(const std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs);

  // Sampling + MTP 完成后调用，为 batch 中的每个 prefill 请求保存生成的 token(s)。
  // 仅在 prefill/standalone 角色下、仅卡 0 保存。保存 generated_tokens + draft_tokens。
  void SubmitFirstTokensPut(const std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs);

 private:
  // 对单个请求执行 KV cache put（在线程池中异步执行）
  // token_ids 为提交时的快照，避免异步执行时被调度器修改
  void PutRequestKVCache(const std::shared_ptr<ExecutorInferRequest>& req, const std::vector<int>& token_ids);

  std::shared_ptr<GlobalCacheConnectorInterface> connector_;
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager_;
  int dp_group_id_ = 0;
  int device_id_ = 0;
  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_;
  std::string role_;  // "prefill" / "decode" / ""

  std::unique_ptr<ThreadPool> thread_pool_;
  std::atomic<bool> terminated_{false};
};

}  // namespace ksana_llm
