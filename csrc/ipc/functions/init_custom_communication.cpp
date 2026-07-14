/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/init_custom_communication.h"

#include <cstring>

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
// Check whether the IPC handle is valid
static bool VerifyIpcHandle(const cudaIpcMemHandle_t& ipc_handle) {
  static cudaIpcMemHandle_t none{};
  return std::memcmp(&ipc_handle, &none, sizeof(cudaIpcMemHandle_t)) != 0;
}
#endif

Status InitCustomCommunicationRequest::Serialize(void* ptr, size_t* bytes) const {
#ifdef ENABLE_CUDA
  SerializeInit(*bytes);
  for (size_t i = 0; i < kCustomIpcHandleCount; i++) {
    SerializeVector<cudaIpcMemHandle_t>(ipc_handles[i], ptr, *bytes);
  }
#endif

  return Status();
}

Status InitCustomCommunicationRequest::Deserialize(const void* ptr) {
#ifdef ENABLE_CUDA
  size_t offset = 0;
  for (size_t i = 0; i < kCustomIpcHandleCount; i++) {
    DeserializeVector<cudaIpcMemHandle_t>(ipc_handles[i], ptr, offset);
  }
#endif

  return Status();
}

Status InitCustomCommunicationResponse::Serialize(void* ptr, size_t* bytes) const {
#ifdef ENABLE_CUDA
  SerializeInit(*bytes);
  for (size_t i = 0; i < kCustomIpcHandleCount; i++) {
    SerializePod<cudaIpcMemHandle_t>(ipc_handles[i], ptr, *bytes);
  }
#endif

  return Status();
}

Status InitCustomCommunicationResponse::Deserialize(const void* ptr) {
#ifdef ENABLE_CUDA
  size_t offset = 0;
  for (size_t i = 0; i < kCustomIpcHandleCount; i++) {
    ipc_handles[i] = DeserializePod<cudaIpcMemHandle_t>(ptr, offset);
  }
#endif

  return Status();
}

void InitCustomCommunicationCallable::SetContext(std::shared_ptr<Context> context) { context_ = context; }

Status InitCustomCommunicationCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  if (!context_->device->ext->IsP2PAccessSupported()) {
    return Status();
  }

#ifdef ENABLE_CUDA
  const InitCustomCommunicationRequest* real_req = reinterpret_cast<const InitCustomCommunicationRequest*>(request);
  InitCustomCommunicationResponse* real_rsp = reinterpret_cast<InitCustomCommunicationResponse*>(response);
  const int node_device_num = context_->global->GetTensorParallelSize();
  if (real_req->ipc_handles.front().empty()) {
    // Pull IPC handles from each executor via the response
    const int rank = context_->device->GetDeviceId();
    std::array<void*, kCustomIpcHandleCount> pointers{
        context_->device->ext->GetCustomAllReduceSignals()[rank],
        context_->device->ext->GetCustomAllReduceInputs()[rank],
        context_->device->ext->GetTrtAllReduceBuffers()[rank],
        context_->device->ext->GetTrtAllReduceBuffers()[node_device_num + rank],
        context_->device->ext->GetTrtAllReduceBuffers()[2 * node_device_num + rank],
        context_->device->ext->GetTrtAllReduceFlags()[rank],
        context_->device->ext->GetTrtAllReduceWorkspaces()[rank]};
    if (context_->device->ext->IsMulticastSupported()) {
      // When using multicast, the reduce input does not need to be exported
      pointers[1] = nullptr;
    }
    KLLM_CHECK(pointers.size() == kCustomIpcHandleCount);
    for (size_t i = 0; i < kCustomIpcHandleCount; i++) {
      if (pointers[i] != nullptr) {
        CUDA_CHECK(cudaIpcGetMemHandle(&real_rsp->ipc_handles[i], pointers[i]));
      }
    }
  } else {
    // Push IPC handles to each executor via the request
    for (int rank = 0; rank < node_device_num; rank++) {
      if (rank != context_->device->GetDeviceId()) {
        std::array<void**, kCustomIpcHandleCount> pointers{
            context_->device->ext->GetCustomAllReduceSignals() + rank,
            context_->device->ext->GetCustomAllReduceInputs() + rank,
            context_->device->ext->GetTrtAllReduceBuffers().data() + rank,
            context_->device->ext->GetTrtAllReduceBuffers().data() + node_device_num + rank,
            context_->device->ext->GetTrtAllReduceBuffers().data() + 2 * node_device_num + rank,
            context_->device->ext->GetTrtAllReduceFlags().data() + rank,
            context_->device->ext->GetTrtAllReduceWorkspaces().data() + rank};
        KLLM_CHECK(pointers.size() == kCustomIpcHandleCount);
        for (size_t i = 0; i < kCustomIpcHandleCount; i++) {
          if (VerifyIpcHandle(real_req->ipc_handles[i][rank])) {
            CUDA_CHECK(
                cudaIpcOpenMemHandle(pointers[i], real_req->ipc_handles[i][rank], cudaIpcMemLazyEnablePeerAccess));
          }
        }
      }
    }
    for (size_t i = 0; i < kCustomIpcHandleCount; i++) {
      real_rsp->ipc_handles[i] = cudaIpcMemHandle_t{};
    }
  }
#endif
  return Status();
}

#ifdef ENABLE_CUDA
void InitCustomCommunicationCallback::SetCustomCommunicationIpcHandles(std::vector<cudaIpcMemHandle_t>* ipc_handles) {
  ipc_handles_ = ipc_handles;
}
#endif

Status InitCustomCommunicationCallback::Call(const IpcResponse* response, IpcContext* context) {
#ifdef ENABLE_CUDA
  const InitCustomCommunicationResponse* real_rsp = reinterpret_cast<const InitCustomCommunicationResponse*>(response);
  for (size_t i = 0; i < kCustomIpcHandleCount; i++) {
    if (VerifyIpcHandle(real_rsp->ipc_handles[i])) {
      // Set ipc handle based on response
      ipc_handles_[i][context->to_rank] = real_rsp->ipc_handles[i];
    }
  }
#endif
  Notify();
  return Status();
}

}  // namespace ksana_llm
