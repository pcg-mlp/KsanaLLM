/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/dispatch_schdule_output.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"

namespace ksana_llm {

Status DispatchScheduleOutputRequest::Serialize(void* ptr, size_t* bytes) const {
  if (schedule_output) {
    return ScheduleOutputParser::SerializeScheduleOutput(schedule_output.get(), ptr, bytes);
  }

  if (serialized_packet) {
    memcpy(ptr, serialized_packet->body, serialized_packet->size);
    *bytes = serialized_packet->size;
  }

  return Status();
}

Status DispatchScheduleOutputRequest::Deserialize(const void* ptr) {
  schedule_output = std::make_shared<ScheduleOutput>();
  return ScheduleOutputParser::DeserializeScheduleOutput(ptr, schedule_output.get());
}

Status DispatchScheduleOutputCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const DispatchScheduleOutputRequest* real_req = reinterpret_cast<const DispatchScheduleOutputRequest*>(request);
  schedule_output_queue_->Put(real_req->schedule_output);
  return Status();
}

void DispatchScheduleOutputCallable::SetRecvQueue(
    std::shared_ptr<BlockingQueue<std::shared_ptr<ScheduleOutput>>> schedule_output_queue) {
  schedule_output_queue_ = schedule_output_queue;
}

}  // namespace ksana_llm
