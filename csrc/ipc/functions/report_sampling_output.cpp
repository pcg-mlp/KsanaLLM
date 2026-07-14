/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/report_sampling_output.h"

#include "ipc/core/ipc_func_serialization.h"

namespace ksana_llm {

Status ReportSamplingOutputRequest::Serialize(void* ptr, size_t* bytes) const {
  return SamplingOutputParser::SerializeSamplingOutput(sampling_output.get(), ptr, bytes);
}

Status ReportSamplingOutputRequest::Deserialize(const void* ptr) {
  sampling_output = std::make_shared<SamplingOutput>();
  return SamplingOutputParser::DeserializeSamplingOutput(ptr, sampling_output.get());
}

Status ReportSamplingOutputCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const ReportSamplingOutputRequest* real_req = reinterpret_cast<const ReportSamplingOutputRequest*>(request);
  sampling_output_queue_->Put(real_req->sampling_output);
  return Status();
}

void ReportSamplingOutputCallable::SetRecvQueue(
    std::shared_ptr<BlockingQueue<std::shared_ptr<SamplingOutput>>> schedule_output_queue) {
  sampling_output_queue_ = schedule_output_queue;
}

}  // namespace ksana_llm
