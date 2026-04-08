/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <memory>

#include "ipc/core/ipc_func_types.h"

namespace ksana_llm {

class GlobalCachePutTracker;

// Executor → Engine: 报告某个设备上单个请求的 KV cache put 完成。
class GlobalCachePutDoneRequest : public IpcRequest {
 public:
  virtual ~GlobalCachePutDoneRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  int device_id = -1;
  int64_t req_id = -1;
};

class GlobalCachePutDoneResponse : public IpcResponse {
 public:
  virtual ~GlobalCachePutDoneResponse() {}
};

// Engine 端的 callable：接收 executor 的 put 完成通知，转发给 GlobalCachePutTracker。
class GlobalCachePutDoneCallable : public IpcCallable {
 public:
  virtual ~GlobalCachePutDoneCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetGlobalCachePutTracker(std::shared_ptr<GlobalCachePutTracker> tracker);

 private:
  std::shared_ptr<GlobalCachePutTracker> tracker_;
};

class GlobalCachePutDoneCallback : public IpcCallback {
 public:
  virtual ~GlobalCachePutDoneCallback() {}
};

}  // namespace ksana_llm
