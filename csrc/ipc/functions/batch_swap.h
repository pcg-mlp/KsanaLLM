/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "ipc/core/ipc_func_types.h"

namespace ksana_llm {

// Request sent from engine to executor to trigger batch swap operation.
class BatchSwapRequest : public IpcRequest {
 public:
  virtual ~BatchSwapRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // Device index for the swap operation.
  int index;

  // Whether this is a swap-in operation (true) or swap-out operation (false).
  bool is_swapin;

  // Block ids for the swap operation.
  // For swap-out: host_block_ids are destination, device_block_ids are source.
  // For swap-in: device_block_ids are destination, host_block_ids are source.
  std::vector<int> device_block_ids;
  std::vector<int> host_block_ids;
};

class BatchSwapResponse : public IpcResponse {
 public:
  virtual ~BatchSwapResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // Status of the swap operation.
  Status swap_status;
};

// Callable that executes the batch swap operation on executor side.
class BatchSwapCallable : public IpcCallable {
 public:
  virtual ~BatchSwapCallable();
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager,
                                int dp_group_id);
  void SetBlockAllocatorGroup(std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group);

 private:
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_ = nullptr;
  int dp_group_id_ = -1;
  std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager_;
};

// Empty callback for BatchSwapRequest.
class BatchSwapCallback : public IpcCallback {
 public:
  virtual ~BatchSwapCallback() {}
};

}  // namespace ksana_llm
