/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/batch_swap.h"

#include "ipc/core/ipc_func_serialization.h"

namespace ksana_llm {

// BatchSwapRequest serialization/deserialization
Status BatchSwapRequest::Serialize(void* ptr, size_t* bytes) const {
  size_t offset = 0;
  SerializePod<int>(index, ptr, offset);
  SerializePod<bool>(is_swapin, ptr, offset);
  SerializeVector<int>(device_block_ids, ptr, offset);
  SerializeVector<int>(host_block_ids, ptr, offset);
  *bytes = offset;
  return Status();
}

Status BatchSwapRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  index = DeserializePod<int>(ptr, offset);
  is_swapin = DeserializePod<bool>(ptr, offset);
  DeserializeVector<int>(device_block_ids, ptr, offset);
  DeserializeVector<int>(host_block_ids, ptr, offset);
  return Status();
}

Status BatchSwapResponse::Serialize(void* ptr, size_t* bytes) const {
  size_t offset = 0;
  SerializeStatus(swap_status, ptr, offset);
  *bytes = offset;
  return Status();
}

Status BatchSwapResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  swap_status = DeserializeStatus(ptr, offset);
  return Status();
}

// BatchSwapCallable implementation
BatchSwapCallable::~BatchSwapCallable() = default;

Status BatchSwapCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const BatchSwapRequest* real_req = reinterpret_cast<const BatchSwapRequest*>(request);
  BatchSwapResponse* real_rsp = reinterpret_cast<BatchSwapResponse*>(response);

  if (block_allocator_group_ == nullptr) {
    block_allocator_group_ = block_allocator_manager_->GetBlockAllocatorGroup(dp_group_id_);
  }
  KLLM_LOG_SCHEDULER << "BatchSwapCallable get group: index=" << real_req->index
                     << ", is_swapin=" << real_req->is_swapin << ", device_block_ids=" << real_req->device_block_ids
                     << ", host_block_ids=" << real_req->host_block_ids;
  Status swap_status;
  if (block_allocator_group_) {
    if (real_req->is_swapin) {
      // Swap in: device_block_ids are destination, host_block_ids are source
      swap_status =
          block_allocator_group_->BatchSwapIn(real_req->index, real_req->device_block_ids, real_req->host_block_ids);
    } else {
      // Swap out: host_block_ids are destination, device_block_ids are source
      swap_status =
          block_allocator_group_->BatchSwapOut(real_req->index, real_req->host_block_ids, real_req->device_block_ids);
    }
  } else {
    swap_status = Status(RET_RUNTIME_FAILED, "BatchSwapCallable: block_allocator_group_ is not set.");
  }

  KLLM_LOG_SCHEDULER << "BatchSwapCallable: index=" << real_req->index << ", is_swapin=" << real_req->is_swapin
                     << ", device_block_ids=" << real_req->device_block_ids
                     << ", host_block_ids=" << real_req->host_block_ids << ", status=" << swap_status.GetMessage();

  if (real_rsp) {
    real_rsp->swap_status = swap_status;
  } else {
    KLLM_LOG_WARNING << "BatchSwapCallable: response is null, swap status not returned.";
  }
  return Status();
}

void BatchSwapCallable::SetBlockAllocatorManager(
    std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager, int dp_group_id) {
  block_allocator_manager_ = block_allocator_manager;
  dp_group_id_ = dp_group_id;
}

}  // namespace ksana_llm
