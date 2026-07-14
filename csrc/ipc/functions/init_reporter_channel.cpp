/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/init_reporter_channel.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/memory_allocator.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

Status InitReporterChannelResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(result_status, ptr, *bytes);
  return Status();
}

Status InitReporterChannelResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  result_status = DeserializeStatus(ptr, offset);
  return Status();
}

void InitReporterChannelCallable::SetDistributedControlCoordinator(
    std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator) {
  distributed_control_coordinator_ = distributed_control_coordinator;
}

Status InitReporterChannelCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const InitReporterChannelRequest* real_req = reinterpret_cast<const InitReporterChannelRequest*>(request);

  STATUS_CHECK_RETURN(distributed_control_coordinator_->InitializeResultReporter());

  return Status();
}

Status InitReporterChannelCallback::Call(const IpcResponse* response, IpcContext* context) {
  const InitReporterChannelResponse* rsp = reinterpret_cast<const InitReporterChannelResponse*>(response);
  if (!rsp->result_status.OK()) {
    KLLM_LOG_ERROR << "Init result reporter channel error: " << rsp->result_status.GetMessage();
    status = rsp->result_status;
  }
  Notify();
  return Status();
}

}  // namespace ksana_llm
