/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/init_device_communication.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

Status InitDeviceCommunicationRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
#ifdef ENABLE_CUDA
  SerializeArray<char>(nccl_uid.internal, NCCL_UNIQUE_ID_BYTES, ptr, *bytes);
#elif defined(ENABLE_KUNLUN)
  SerializeArray<char>(bkcl_uid.id, 128, ptr, *bytes);
#endif

  return Status();
}

Status InitDeviceCommunicationRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
#ifdef ENABLE_CUDA
  DeserializeArray<char>(nccl_uid.internal, ptr, offset);
#elif defined(ENABLE_KUNLUN)
  DeserializeArray<char>(bkcl_uid.id, ptr, offset);
#endif

  return Status();
}

void InitDeviceCommunicationCallable::SetContext(std::shared_ptr<Context> context) { context_ = context; }

Status InitDeviceCommunicationCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const InitDeviceCommunicationRequest* real_req = reinterpret_cast<const InitDeviceCommunicationRequest*>(request);
#ifdef ENABLE_CUDA
  context_->device->ext->InitNcclParam(real_req->nccl_uid);
#elif defined(ENABLE_KUNLUN)
  context_->device->ext->SetBKCLUniqueId(real_req->bkcl_uid);
  int tp_size = context_->global->GetTensorParallelSize();
  int device_id = context_->device->GetDeviceId();
  context_->device->ext->InitBkcl(device_id, tp_size);
#endif
  return Status();
}

Status InitDeviceCommunicationCallback::Call(const IpcResponse* response, IpcContext* context) {
  Notify();
  return Status();
}

}  // namespace ksana_llm
