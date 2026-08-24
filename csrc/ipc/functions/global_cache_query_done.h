/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "ipc/core/ipc_func_types.h"

namespace ksana_llm {

class GlobalCacheFetcher;

// Executor → Engine: 报告某个设备上全局 cache 查询和拉取完成。
class GlobalCacheQueryDoneRequest : public IpcRequest {
 public:
  virtual ~GlobalCacheQueryDoneRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  int64_t req_id = -1;
  int device_id = -1;
  std::vector<int> matched_ranges;
  bool success = false;
  // 全局 cache 中缓存的生成 token(s)（全命中时有效，含 generated + draft）
  std::vector<int> cached_tokens;
  bool has_cached_tokens = false;
  // 单卡各阶段耗时（us），供 engine 在 fetch_e2e 对齐多卡
  int64_t wait_us = -1;        // enqueue → Call 开始（IPC/long-task 排队）
  int64_t qf_us = -1;          // QueryAndFetch
  int64_t token_us = -1;       // QueryFirstTokens（仅卡 0，其它为 0）
  int64_t report_us = -1;      // ReportGlobalCacheQueryDone Invoke
  int64_t call_us = -1;        // Call 入口 → report 前
  int64_t report_wall_us = 0;  // 发起 report 时的 wall clock，engine 算回报延迟

  // SWA 对齐后的命中 token 数；0 表示无 SWA 对齐结果（未启用 SWA，或没找到合法边界）。
  size_t swa_aligned_hit_len = 0;

  // 实际写入了全局 SWA 内容的 FA block 索引。
  std::vector<size_t> swa_filled_block_idxs;
};

class GlobalCacheQueryDoneResponse : public IpcResponse {
 public:
  virtual ~GlobalCacheQueryDoneResponse() {}
};

// Engine 端的 callable：接收 executor 的完成通知，转发给 GlobalCacheFetcher。
class GlobalCacheQueryDoneCallable : public IpcCallable {
 public:
  virtual ~GlobalCacheQueryDoneCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetGlobalCacheFetcher(std::shared_ptr<GlobalCacheFetcher> fetcher);

 private:
  std::shared_ptr<GlobalCacheFetcher> fetcher_;
};

class GlobalCacheQueryDoneCallback : public IpcCallback {
 public:
  virtual ~GlobalCacheQueryDoneCallback() {}
};

}  // namespace ksana_llm
