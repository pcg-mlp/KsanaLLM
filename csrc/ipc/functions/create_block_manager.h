/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "cache_manager/block_allocator/block_allocator_manager.h"

#include "configure/environment.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "pd_v2/public/pd_v2_connector.h"
#include "utils/waiter.h"

namespace ksana_llm {

class CreateBlockManagerRequest : public IpcRequest {
 public:
  virtual ~CreateBlockManagerRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  BlockAllocatorManagerConfig block_allocator_manager_config;
  // Optional SWA block allocator config; empty map means SWA is disabled.
  BlockAllocatorManagerConfig swa_block_allocator_manager_config;
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
  void SetSwaBlockAllocatorManager(std::shared_ptr<BlockAllocatorManagerInterface> swa_block_allocator_manager);

  // Bound by ksana_executor bootstrap when pd_v2 is enabled. After
  // block_allocator_manager_->Initialize allocates the contiguous KV
  // pool, this callable forwards (base_ptr, total_bytes, block_size,
  // num_blocks, num_layers) to the connector so it can call mooncake
  // registerLocalMemory. No-op when nullptr.
  //
  // `dp_group_id`            — which BlockAllocatorGroup to look up
  //                            (== executor_id / attn_TP). Under
  //                            attn_DP > 1, each executor process owns
  //                            its own group; hardcoding 0 only works
  //                            for attn_DP=1.
  // `device_idx_in_dp_group` — DeviceBlockAllocator index inside that
  //                            group (== executor_id % attn_TP). Under
  //                            attn_DP > 1, this is 0 since each group
  //                            holds one device.
  void SetPdV2Connector(std::shared_ptr<pd_v2::PdV2ConnectorInterface> connector, std::shared_ptr<Environment> env,
                        int dp_group_id, int device_idx_in_dp_group);

 private:
  std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager_ = nullptr;
  // Optional SWA block allocator manager; nullptr means SWA is disabled.
  std::shared_ptr<BlockAllocatorManagerInterface> swa_block_allocator_manager_ = nullptr;
  std::shared_ptr<pd_v2::PdV2ConnectorInterface> pd_v2_connector_ = nullptr;
  std::shared_ptr<Environment> env_for_pd_v2_ = nullptr;
  int dp_group_id_ = 0;
  int device_idx_in_dp_group_ = 0;
};

class CreateBlockManagerCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~CreateBlockManagerCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  // The final result stat of loading weight.
  Status load_status;
};

}  // namespace ksana_llm
