/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/connect_nccl_data_channel.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/memory_allocator.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

Status ConnectNcclDataChannelRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializePod<ncclUniqueId>(nccl_unique_id, ptr, *bytes);
  return Status();
}

Status ConnectNcclDataChannelRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  nccl_unique_id = DeserializePod<ncclUniqueId>(ptr, offset);
  return Status();
}

Status ConnectNcclDataChannelResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(result_status, ptr, *bytes);
  return Status();
}

Status ConnectNcclDataChannelResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  result_status = DeserializeStatus(ptr, offset);
  return Status();
}

void ConnectNcclDataChannelCallable::SetDistributedDataCoordinator(
    std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator) {
  distributed_data_coordinator_ = distributed_data_coordinator;
}

Status ConnectNcclDataChannelCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const ConnectNcclDataChannelRequest* real_req = reinterpret_cast<const ConnectNcclDataChannelRequest*>(request);

  STATUS_CHECK_RETURN(distributed_data_coordinator_->InitializeNcclCluster(real_req->nccl_unique_id));

  return Status();
}

Status ConnectNcclDataChannelCallback::Call(const IpcResponse* response, IpcContext* context) {
  const ConnectNcclDataChannelResponse* rsp = reinterpret_cast<const ConnectNcclDataChannelResponse*>(response);
  if (!rsp->result_status.OK()) {
    KLLM_LOG_ERROR << "Init nccl data channel error: " << rsp->result_status.GetMessage();
    status = rsp->result_status;
  }
  Notify();
  return Status();
}

}  // namespace ksana_llm
