/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/create_block_manager.h"

#include "cache_manager/block_allocator/block_allocator_interface.h"
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
  SerializeBlockAllocatorManagerConfig(mamba_block_allocator_manager_config, ptr, *bytes);
  return Status();
}

Status CreateBlockManagerRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  DeserializeBlockAllocatorManagerConfig(block_allocator_manager_config, ptr, offset);
  DeserializeBlockAllocatorManagerConfig(swa_block_allocator_manager_config, ptr, offset);
  DeserializeBlockAllocatorManagerConfig(mamba_block_allocator_manager_config, ptr, offset);
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

void CreateBlockManagerCallable::SetMambaBlockAllocatorManager(
    std::shared_ptr<BlockAllocatorManagerInterface> mamba_block_allocator_manager) {
  mamba_block_allocator_manager_ = mamba_block_allocator_manager;
}

void CreateBlockManagerCallable::SetPdV2Connector(std::shared_ptr<pd_v2::PdV2ConnectorInterface> connector,
                                                  std::shared_ptr<Environment> env, int dp_group_id,
                                                  int device_idx_in_dp_group) {
  pd_v2_connector_ = std::move(connector);
  env_for_pd_v2_ = std::move(env);
  dp_group_id_ = dp_group_id;
  device_idx_in_dp_group_ = device_idx_in_dp_group;
}

Status CreateBlockManagerCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const CreateBlockManagerRequest* real_req = reinterpret_cast<const CreateBlockManagerRequest*>(request);
  block_allocator_manager_->Initialize(real_req->block_allocator_manager_config);
  // Initialize SWA block allocator manager when SWA is enabled and the manager has been registered.
  if (swa_block_allocator_manager_ && !real_req->swa_block_allocator_manager_config.empty()) {
    swa_block_allocator_manager_->Initialize(real_req->swa_block_allocator_manager_config);
  }
  // Initialize the Mamba state pool allocator manager (after FA/SWA, aligned with their order).
  if (mamba_block_allocator_manager_ && !real_req->mamba_block_allocator_manager_config.empty()) {
    mamba_block_allocator_manager_->Initialize(real_req->mamba_block_allocator_manager_config);
  }

  // pd_v2 hook: after the contiguous KV pool is allocated, register it
  // with Mooncake so peer Prefill engines can RDMA-WRITE into it. Skipped
  // when pd_v2 is disabled (connector pointer left null at bootstrap).
  if (pd_v2_connector_ != nullptr && env_for_pd_v2_ != nullptr) {
    // Pick this executor's own attn_dp BlockAllocatorGroup, then the
    // matching DeviceBlockAllocator inside it. Coordinates were
    // computed at boot from the global TP/DP topology:
    //   dp_group_id            = executor_id / attn_TP
    //   device_idx_in_dp_group = executor_id % attn_TP
    // Hardcoding `(0, 0)` (or `(0, executor_id)`) only works for
    // attn_DP=1 — under attn_DP > 1 each executor process owns a
    // distinct group and the lookup must reflect that.
    auto group = block_allocator_manager_->GetBlockAllocatorGroup(dp_group_id_);
    if (group == nullptr) {
      KLLM_LOG_WARNING << "pd_v2 RegisterKvPool skipped: BlockAllocatorGroup(" << dp_group_id_ << ") is null";
      return Status();
    }
    auto dev_alloc = group->GetDeviceBlockAllocator(device_idx_in_dp_group_);
    if (dev_alloc == nullptr) {
      KLLM_LOG_WARNING << "pd_v2 RegisterKvPool skipped: DeviceBlockAllocator(dp_group_id=" << dp_group_id_
                       << " device_idx_in_dp_group=" << device_idx_in_dp_group_ << ") is null";
      return Status();
    }
    void* base = dev_alloc->GetBlocksBasePtr();
    if (base == nullptr) {
      KLLM_LOG_WARNING << "pd_v2 RegisterKvPool skipped: GetBlocksBasePtr() returned null. "
                       << "Ensure setting.connector.backend=v2 so the contiguous-KV gate fires.";
      return Status();
    }
    const auto& cfg = group->GetBlockAllocatorGroupConfig();
    const size_t num_blocks = cfg.device_block_num;
    const size_t block_size = cfg.block_size;
    // One extra slot is reserved by PreAllocateBlocks (block_num+1)*block_size,
    // include it in the registered region so peers can use any logical id.
    const size_t total_bytes = (num_blocks + 1) * block_size;
    ModelConfig model_config;
    env_for_pd_v2_->GetModelConfig(model_config);
    // pd_v2 注册的 num_layers 需要包含 MTP / nextn 层。KV cache block 布局按
    // num_layer + num_nextn_predict_layers 分配；若只注册 num_layer，Prefill 侧
    // PdV2KvDispatcher 会算错每层 element_size，导致层偏移和 MTP slot 写入错误。
    const size_t num_layers = model_config.num_layer + model_config.num_nextn_predict_layers;
    Status s = pd_v2_connector_->RegisterKvPool(base, total_bytes, block_size, num_blocks, num_layers);
    if (!s.OK()) {
      KLLM_LOG_ERROR << "pd_v2 RegisterKvPool failed: " << s.GetMessage();
      // Don't propagate as a CreateBlockManager error — pd_v2 wiring is
      // orthogonal to local block management. Engine GetKvBufferVa will
      // surface the failure when it tries to read the (still unset) base_va.
    } else {
      KLLM_LOG_INFO << "pd_v2 RegisterKvPool: base=" << base << " total_bytes=" << total_bytes
                    << " block_size=" << block_size << " num_blocks=" << num_blocks << " num_layers=" << num_layers;
    }
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
