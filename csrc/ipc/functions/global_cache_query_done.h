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
