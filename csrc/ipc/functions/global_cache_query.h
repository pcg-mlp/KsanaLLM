/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "global_cache_connector/global_cache_connector_interface.h"
#include "ipc/core/ipc_func_types.h"

namespace ksana_llm {

class IpcFuncWrapper;
class Environment;

// Engine → Executor: 通知 executor 执行全局 cache 查询和拉取。
// 广播到所有 executor，每个 executor 在自己的 GPU 上执行 QueryAndFetch，
// 完成后通过 GlobalCacheQueryDone IPC 通知 engine。
class GlobalCacheQueryRequest : public IpcRequest {
 public:
  virtual ~GlobalCacheQueryRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  int64_t req_id = -1;
  std::vector<int> token_ids;
  size_t local_cached_len = 0;
  int block_size = 0;
  std::vector<int> kv_cache_block_ids;
};

class GlobalCacheQueryResponse : public IpcResponse {
 public:
  virtual ~GlobalCacheQueryResponse() {}
};

// Executor 端的 callable：接收查询请求，执行 Connector 的 QueryAndFetch，
// 然后通过 IPC 报告完成。
class GlobalCacheQueryCallable : public IpcCallable {
 public:
  virtual ~GlobalCacheQueryCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetGlobalCacheConnector(std::shared_ptr<GlobalCacheConnectorInterface> connector);
  void SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager,
                                int dp_group_id);
  // 使用原始指针避免与 IpcFuncWrapper 的循环引用
  void SetIpcFuncWrapper(IpcFuncWrapper* ipc_func_wrapper);
  void SetEnvironment(std::shared_ptr<Environment> env);
  void SetDeviceId(int device_id);
  void SetGlobalCacheRole(const std::string& role, int timeout_ms);

 private:
  // 判断 matched_ranges 是否从 local_cached_len 开始连续覆盖到 total_tokens-1
  static bool IsFullHit(const std::vector<int>& matched_ranges, int total_tokens, int local_cached_len);

  std::shared_ptr<GlobalCacheConnectorInterface> connector_;
  std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager_;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_;
  IpcFuncWrapper* ipc_func_wrapper_ = nullptr;
  std::shared_ptr<Environment> env_;
  int dp_group_id_ = 0;
  int device_id_ = 0;
  std::string role_;  // "prefill" / "decode" / ""
  int timeout_ms_ = 5000;
};

class GlobalCacheQueryCallback : public IpcCallback {
 public:
  virtual ~GlobalCacheQueryCallback() {}
};

}  // namespace ksana_llm
