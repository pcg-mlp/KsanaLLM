/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/connect_tcp_data_channel.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/memory_allocator.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

Status ConnectTcpDataChannelRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializePod<HostPort>(executor_downstream_host_port, ptr, *bytes);
  return Status();
}

Status ConnectTcpDataChannelRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  executor_downstream_host_port = DeserializePod<HostPort>(ptr, offset);
  return Status();
}

Status ConnectTcpDataChannelResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(status, ptr, *bytes);
  return Status();
}

Status ConnectTcpDataChannelResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  status = DeserializeStatus(ptr, offset);
  return Status();
}

void ConnectTcpDataChannelCallable::SetDistributedDataCoordinator(
    std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator) {
  distributed_data_coordinator_ = distributed_data_coordinator;
}

Status ConnectTcpDataChannelCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const ConnectTcpDataChannelRequest* real_req = reinterpret_cast<const ConnectTcpDataChannelRequest*>(request);
  ConnectTcpDataChannelResponse* real_rsp = reinterpret_cast<ConnectTcpDataChannelResponse*>(response);

  HostPort host_port = real_req->executor_downstream_host_port;

  STATUS_CHECK_RETURN(real_rsp->status = distributed_data_coordinator_->Connect(host_port.host, host_port.port));

  return Status();
}

Status ConnectTcpDataChannelCallback::Call(const IpcResponse* response, IpcContext* context) {
  const ConnectTcpDataChannelResponse* rsp = reinterpret_cast<const ConnectTcpDataChannelResponse*>(response);
  if (!rsp->status.OK()) {
    KLLM_LOG_ERROR << "Connect tcp data channel error: " << rsp->status.GetMessage();
    status = rsp->status;
  }
  Notify();
  return Status();
}

}  // namespace ksana_llm
