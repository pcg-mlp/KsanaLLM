/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/create_block_manager.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/memory_allocator.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

// Helper: serialize a BlockAllocatorManagerConfig map into the buffer.
static void SerializeBlockAllocatorManagerConfig(const BlockAllocatorManagerConfig& cfg, void* ptr, size_t& bytes) {
  size_t group_size = cfg.size();
  SerializePod<size_t>(group_size, ptr, bytes);
  for (const auto& [dp_id, config] : cfg) {
    SerializePod<int>(dp_id, ptr, bytes);
    SerializeVector<int>(config.devices, ptr, bytes);
    SerializePod<size_t>(config.device_block_num, ptr, bytes);
    SerializePod<size_t>(config.host_block_num, ptr, bytes);
    SerializePod<size_t>(config.block_size, ptr, bytes);
  }
}

// Helper: deserialize a BlockAllocatorManagerConfig map from the buffer.
static void DeserializeBlockAllocatorManagerConfig(BlockAllocatorManagerConfig& cfg, const void* ptr, size_t& offset) {
  size_t group_size = DeserializePod<size_t>(ptr, offset);
  for (size_t i = 0; i < group_size; ++i) {
    int dp_id = DeserializePod<int>(ptr, offset);
    BlockAllocatorGroupConfig config;
    DeserializeVector<int>(config.devices, ptr, offset);
    config.device_block_num = DeserializePod<size_t>(ptr, offset);
    config.host_block_num = DeserializePod<size_t>(ptr, offset);
    config.block_size = DeserializePod<size_t>(ptr, offset);
    cfg[dp_id] = config;
  }
}

Status CreateBlockManagerRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeBlockAllocatorManagerConfig(block_allocator_manager_config, ptr, *bytes);
  SerializeBlockAllocatorManagerConfig(swa_block_allocator_manager_config, ptr, *bytes);
  return Status();
}

Status CreateBlockManagerRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  DeserializeBlockAllocatorManagerConfig(block_allocator_manager_config, ptr, offset);
  DeserializeBlockAllocatorManagerConfig(swa_block_allocator_manager_config, ptr, offset);
  return Status();
}

Status CreateBlockManagerResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(result_status, ptr, *bytes);
  return Status();
}

Status CreateBlockManagerResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  result_status = DeserializeStatus(ptr, offset);
  return Status();
}

void CreateBlockManagerCallable::SetBlockAllocatorManager(
    std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager) {
  block_allocator_manager_ = block_allocator_manager;
}

void CreateBlockManagerCallable::SetSwaBlockAllocatorManager(
    std::shared_ptr<BlockAllocatorManagerInterface> swa_block_allocator_manager) {
  swa_block_allocator_manager_ = swa_block_allocator_manager;
}

Status CreateBlockManagerCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const CreateBlockManagerRequest* real_req = reinterpret_cast<const CreateBlockManagerRequest*>(request);
  block_allocator_manager_->Initialize(real_req->block_allocator_manager_config);
  // Initialize SWA block allocator manager when SWA is enabled and the manager has been registered.
  if (swa_block_allocator_manager_ && !real_req->swa_block_allocator_manager_config.empty()) {
    swa_block_allocator_manager_->Initialize(real_req->swa_block_allocator_manager_config);
  }
  return Status();
}

Status CreateBlockManagerCallback::Call(const IpcResponse* response, IpcContext* context) {
  const CreateBlockManagerResponse* rsp = reinterpret_cast<const CreateBlockManagerResponse*>(response);
  if (!rsp->result_status.OK()) {
    KLLM_LOG_ERROR << "Create block manager failed, info: " << rsp->result_status.GetMessage();
    load_status = rsp->result_status;
  }
  Notify();
  return Status();
}

}  // namespace ksana_llm
