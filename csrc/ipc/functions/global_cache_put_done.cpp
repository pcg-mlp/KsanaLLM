/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/global_cache_put_done.h"

#include "batch_scheduler/global_cache_put_tracker.h"
#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"

namespace ksana_llm {

Status GlobalCachePutDoneRequest::Serialize(void* ptr, size_t* bytes) const {
  size_t offset = 0;
  SerializePod<int>(device_id, ptr, offset);
  SerializePod<int64_t>(req_id, ptr, offset);
  *bytes = offset;
  return Status();
}

Status GlobalCachePutDoneRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  device_id = DeserializePod<int>(ptr, offset);
  req_id = DeserializePod<int64_t>(ptr, offset);
  return Status();
}

Status GlobalCachePutDoneCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const GlobalCachePutDoneRequest* real_req = reinterpret_cast<const GlobalCachePutDoneRequest*>(request);

  KLLM_LOG_SCHEDULER << "received put done notification, device_id=" << real_req->device_id
                     << ", req_id=" << real_req->req_id;

  if (tracker_) {
    tracker_->OnDevicePutDone(real_req->device_id, real_req->req_id);
  } else {
    KLLM_LOG_WARNING << "tracker not set";
  }

  return Status();
}

void GlobalCachePutDoneCallable::SetGlobalCachePutTracker(std::shared_ptr<GlobalCachePutTracker> tracker) {
  tracker_ = tracker;
}

}  // namespace ksana_llm
