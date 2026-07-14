/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/listen_tcp_data_channel.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/memory_allocator.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

Status ListenTcpDataChannelRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  return Status();
}

Status ListenTcpDataChannelRequest::Deserialize(const void* ptr) { return Status(); }

Status ListenTcpDataChannelResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(status, ptr, *bytes);
  SerializePod<HostPort>(executor_host_port, ptr, *bytes);
  return Status();
}

Status ListenTcpDataChannelResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  status = DeserializeStatus(ptr, offset);
  executor_host_port = DeserializePod<HostPort>(ptr, offset);
  return Status();
}

void ListenTcpDataChannelCallable::SetDistributedDataCoordinator(
    std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator) {
  distributed_data_coordinator_ = distributed_data_coordinator;
}

Status ListenTcpDataChannelCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  uint16_t port;
  std::string host;
  distributed_data_coordinator_->Listen(host, port);

  ListenTcpDataChannelResponse* rsp = reinterpret_cast<ListenTcpDataChannelResponse*>(response);
  rsp->executor_host_port.port = port;
  memcpy(rsp->executor_host_port.host, host.c_str(), host.size());

  return Status();
}

Status ListenTcpDataChannelCallback::Call(const IpcResponse* response, IpcContext* context) {
  const ListenTcpDataChannelResponse* rsp = reinterpret_cast<const ListenTcpDataChannelResponse*>(response);
  if (!rsp->status.OK()) {
    KLLM_LOG_ERROR << "Listen tcp data channel error: " << rsp->status.GetMessage();
    status = rsp->status;
  }

  int executor_id = context->to_rank;
  if (executor_host_ports.size() <= executor_id) {
    status = Status(RET_RUNTIME_FAILED, fmt::format("Invalid executor_id {}.", executor_id));
  } else {
    executor_host_ports[executor_id] = rsp->executor_host_port;
  }

  Notify();
  return Status();
}

}  // namespace ksana_llm
