/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "cache_manager/block_allocator/block_allocator_manager.h"

#include "configure/environment.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/waiter.h"

namespace ksana_llm {

class CreateBlockManagerRequest : public IpcRequest {
 public:
  virtual ~CreateBlockManagerRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  BlockAllocatorManagerConfig block_allocator_manager_config;
};

class CreateBlockManagerResponse : public IpcResponse {
 public:
  virtual ~CreateBlockManagerResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The loading result, succ or failure.
  Status result_status;
};

class CreateBlockManagerCallable : public IpcCallable {
 public:
  virtual ~CreateBlockManagerCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager);

 private:
  std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager_ = nullptr;
};

class CreateBlockManagerCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~CreateBlockManagerCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  // The final result stat of loading weight.
  Status load_status;
};

}  // namespace ksana_llm
