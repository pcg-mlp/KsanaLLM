/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/init_multicast_communication.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/status.h"
#ifdef ENABLE_CUDA
#  include "device/nvidia/cuda_utils.h"
#endif

namespace ksana_llm {

Status InitMulticastCommunicationRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializePod<uint64_t>(unique_op_id, ptr, *bytes);
  SerializePod<int>(broadcast_id, ptr, *bytes);

  return Status();
}

Status InitMulticastCommunicationRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  unique_op_id = DeserializePod<uint64_t>(ptr, offset);
  broadcast_id = DeserializePod<int>(ptr, offset);

  return Status();
}

void InitMulticastCommunicationCallable::SetContext(std::shared_ptr<Context> context) { context_ = context; }

Status InitMulticastCommunicationCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  if (!context_->device->ext->IsMulticastSupported()) {
    return Status();
  }

#ifdef ENABLE_CUDA
  const InitMulticastCommunicationRequest* real_req =
      reinterpret_cast<const InitMulticastCommunicationRequest*>(request);
  auto nvls_mcast_memory = NvlsMcastMemory::GetInstance();
  auto& nccl_ipc_socket = nvls_mcast_memory->GetNcclIpcSocket();
  const int rank = context_->device->GetDeviceId();
  if (real_req->unique_op_id >= 0) {
    // Initialize IPC socket — use the persistent abort_flag from NvlsMcastMemory
    // (the old code used a stack-local abort_flag that became a dangling pointer)
    nvls_mcast_memory->GetAbortFlag() = 0;
    nccl_ipc_socket = NcclIpcSocketInit(rank, real_req->unique_op_id, &nvls_mcast_memory->GetAbortFlag());
  } else {
    auto& ipc_handles = nvls_mcast_memory->GetIpcHandles();
    const int node_device_num = context_->global->GetTensorParallelSize();
    KLLM_CHECK(ipc_handles.size() == static_cast<size_t>(node_device_num + 1));
    KLLM_CHECK_WITH_INFO(nccl_ipc_socket != nullptr,
                         "IPC socket is null — socket initialization may have been skipped "
                         "(check unique_op_id sign overflow).");
    if (real_req->broadcast_id == rank || (real_req->broadcast_id == node_device_num && rank == 0)) {
      const int send_id = real_req->broadcast_id == node_device_num ? 0 : rank;
      // Broadcast IPC handle file descriptor
      for (int peer = 0; peer < node_device_num; peer++) {
        if (peer != send_id) {
          NcclIpcSocketSendFd(nccl_ipc_socket, ipc_handles[real_req->broadcast_id].fd, peer);
        }
      }
    } else if (real_req->broadcast_id < 0) {
      // Receive IPC handle file descriptor (must after the broadcast completes)
      if (-real_req->broadcast_id - 1 != rank && (real_req->broadcast_id != -node_device_num - 1 || rank != 0)) {
        const int recv_id = -real_req->broadcast_id - 1;
        ipc_handles[recv_id].fd = NcclIpcSocketRecvFd(nccl_ipc_socket);
      }
    } else if (real_req->broadcast_id == node_device_num + 1) {
      nvls_mcast_memory->InitMcastMemory();
      // Overwrite the custom allreduce input handle with the peer unicast address
      void** peer_input_handles = context_->device->ext->GetCustomAllReduceInputs();
      for (int peer = 0; peer < node_device_num; peer++) {
        peer_input_handles[peer] = reinterpret_cast<void*>(nvls_mcast_memory->GetNvlsHandle()->peer_uc_vas[peer]);
      }
      // Close IPC socket
      NcclIpcSocketClose(nccl_ipc_socket);
    }
  }
#endif
  return Status();
}

Status InitMulticastCommunicationCallback::Call(const IpcResponse* response, IpcContext* context) {
  Notify();
  return Status();
}

}  // namespace ksana_llm
