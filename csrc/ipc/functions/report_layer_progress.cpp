/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/report_layer_progress.h"

#include "ipc/core/ipc_func_serialization.h"

namespace ksana_llm {

Status ReportLayerProgressRequest::Serialize(void* ptr, size_t* bytes) const {
  size_t offset = 0;
  SerializePod<int>(device_id, ptr, offset);
  SerializePod<int>(layer_index, ptr, offset);
  *bytes = offset;
  return Status();
}

Status ReportLayerProgressRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  device_id = DeserializePod<int>(ptr, offset);
  layer_index = DeserializePod<int>(ptr, offset);
  return Status();
}

Status ReportLayerProgressCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const ReportLayerProgressRequest* real_req = reinterpret_cast<const ReportLayerProgressRequest*>(request);

  LayerProgressInfo info(real_req->device_id, real_req->layer_index);
  KLLM_LOG_DEBUG << "device_id: " << info.device_id << ", layer_index: " << info.layer_index;
  layer_progress_queue_->Put(info);

  return Status();
}

void ReportLayerProgressCallable::SetRecvQueue(std::shared_ptr<BlockingQueue<LayerProgressInfo>> queue) {
  layer_progress_queue_ = queue;
}

}  // namespace ksana_llm