/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/nccl_operations.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

#ifdef ENABLE_CUDA
#  include "executor/nccl_executor_handler.h"
#endif

namespace ksana_llm {

// ==================== ConnectorInitNcclCommGroup ====================

Status ConnectorInitNcclCommGroupRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeString(group_key, ptr, *bytes);
#ifdef ENABLE_CUDA
  SerializePod<ncclUniqueId>(comm_id, ptr, *bytes);
#endif
  SerializePod<int>(rank, ptr, *bytes);
  SerializePod<int>(world_size, ptr, *bytes);
  return Status();
}

Status ConnectorInitNcclCommGroupRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  group_key = DeserializeString(ptr, offset);
#ifdef ENABLE_CUDA
  comm_id = DeserializePod<ncclUniqueId>(ptr, offset);
#endif
  rank = DeserializePod<int>(ptr, offset);
  world_size = DeserializePod<int>(ptr, offset);
  return Status();
}

Status ConnectorInitNcclCommGroupResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(result_status, ptr, *bytes);
  return Status();
}

Status ConnectorInitNcclCommGroupResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  result_status = DeserializeStatus(ptr, offset);
  return Status();
}

#ifdef ENABLE_CUDA
void ConnectorInitNcclCommGroupCallable::SetNcclExecutorHandler(std::shared_ptr<NcclExecutorHandler> handler) {
  nccl_handler_ = handler;
}
#endif

Status ConnectorInitNcclCommGroupCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
#ifdef ENABLE_CUDA
  const ConnectorInitNcclCommGroupRequest* req = reinterpret_cast<const ConnectorInitNcclCommGroupRequest*>(request);
  ConnectorInitNcclCommGroupResponse* rsp = reinterpret_cast<ConnectorInitNcclCommGroupResponse*>(response);
  KLLM_LOG_INFO << "Connector init NCCL comm group: " << req->group_key;
  if (!nccl_handler_) {
    rsp->result_status = Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL handler not set");
    return rsp->result_status;
  }

  rsp->result_status = nccl_handler_->InitializeCommGroup(req->group_key, req->comm_id, req->rank, req->world_size);
  return Status();
#else
  return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "CUDA not enabled");
#endif
}

Status ConnectorInitNcclCommGroupCallback::Call(const IpcResponse* response, IpcContext* context) {
  const ConnectorInitNcclCommGroupResponse* rsp = reinterpret_cast<const ConnectorInitNcclCommGroupResponse*>(response);
  if (!rsp->result_status.OK()) {
    KLLM_LOG_ERROR << "Connector init NCCL comm group error: " << rsp->result_status.GetMessage();
    status = rsp->result_status;
  }
  Notify();
  return Status();
}

// ==================== ConnectorNcclSendData ====================

Status ConnectorNcclSendDataRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeString(group_key, ptr, *bytes);
  SerializePod<int>(peer_rank, ptr, *bytes);
  SerializeVector<uint8_t>(data, ptr, *bytes);
  SerializeEnum<DataType>(dtype, ptr, *bytes);
  return Status();
}

Status ConnectorNcclSendDataRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  group_key = DeserializeString(ptr, offset);
  peer_rank = DeserializePod<int>(ptr, offset);
  DeserializeVector<uint8_t>(data, ptr, offset);
  dtype = DeserializeEnum<DataType>(ptr, offset);
  return Status();
}

Status ConnectorNcclSendDataResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(result_status, ptr, *bytes);
  return Status();
}

Status ConnectorNcclSendDataResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  result_status = DeserializeStatus(ptr, offset);
  return Status();
}

#ifdef ENABLE_CUDA
void ConnectorNcclSendDataCallable::SetNcclExecutorHandler(std::shared_ptr<NcclExecutorHandler> handler) {
  nccl_handler_ = handler;
}
#endif

Status ConnectorNcclSendDataCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
#ifdef ENABLE_CUDA
  const ConnectorNcclSendDataRequest* req = reinterpret_cast<const ConnectorNcclSendDataRequest*>(request);
  ConnectorNcclSendDataResponse* rsp = reinterpret_cast<ConnectorNcclSendDataResponse*>(response);

  if (!nccl_handler_) {
    rsp->result_status = Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL handler not set");
    return rsp->result_status;
  }

  // NcclExecutorHandler::Send内部会处理GPU内存分配、数据拷贝和stream管理
  rsp->result_status =
      nccl_handler_->Send(req->group_key, req->peer_rank, req->data.data(), req->data.size(), req->dtype);

  return Status();
#else
  return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "CUDA not enabled");
#endif
}

Status ConnectorNcclSendDataCallback::Call(const IpcResponse* response, IpcContext* context) {
  const ConnectorNcclSendDataResponse* rsp = reinterpret_cast<const ConnectorNcclSendDataResponse*>(response);
  if (!rsp->result_status.OK()) {
    KLLM_LOG_ERROR << "Connector NCCL send data error: " << rsp->result_status.GetMessage();
    status = rsp->result_status;
  }
  Notify();
  return Status();
}

// ==================== ConnectorNcclRecvData ====================

Status ConnectorNcclRecvDataRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeString(group_key, ptr, *bytes);
  SerializePod<int>(peer_rank, ptr, *bytes);
  SerializePod<size_t>(count, ptr, *bytes);
  SerializeEnum<DataType>(dtype, ptr, *bytes);
  return Status();
}

Status ConnectorNcclRecvDataRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  group_key = DeserializeString(ptr, offset);
  peer_rank = DeserializePod<int>(ptr, offset);
  count = DeserializePod<size_t>(ptr, offset);
  dtype = DeserializeEnum<DataType>(ptr, offset);
  return Status();
}

Status ConnectorNcclRecvDataResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(result_status, ptr, *bytes);
  SerializeVector<uint8_t>(data, ptr, *bytes);
  return Status();
}

Status ConnectorNcclRecvDataResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  result_status = DeserializeStatus(ptr, offset);
  DeserializeVector<uint8_t>(data, ptr, offset);
  return Status();
}

#ifdef ENABLE_CUDA
void ConnectorNcclRecvDataCallable::SetNcclExecutorHandler(std::shared_ptr<NcclExecutorHandler> handler) {
  nccl_handler_ = handler;
}
#endif

Status ConnectorNcclRecvDataCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
#ifdef ENABLE_CUDA
  const ConnectorNcclRecvDataRequest* req = reinterpret_cast<const ConnectorNcclRecvDataRequest*>(request);
  ConnectorNcclRecvDataResponse* rsp = reinterpret_cast<ConnectorNcclRecvDataResponse*>(response);

  if (!nccl_handler_) {
    rsp->result_status = Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL handler not set");
    return rsp->result_status;
  }

  // 准备接收缓冲区
  rsp->data.resize(req->count);

  // NcclExecutorHandler::Recv内部会处理GPU内存分配、数据拷贝和stream管理
  rsp->result_status = nccl_handler_->Recv(req->group_key, req->peer_rank, rsp->data.data(), req->count, req->dtype);

  return Status();
#else
  return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "CUDA not enabled");
#endif
}

Status ConnectorNcclRecvDataCallback::Call(const IpcResponse* response, IpcContext* context) {
  const ConnectorNcclRecvDataResponse* rsp = reinterpret_cast<const ConnectorNcclRecvDataResponse*>(response);
  if (!rsp->result_status.OK()) {
    KLLM_LOG_ERROR << "Connector NCCL recv data error: " << rsp->result_status.GetMessage();
    status = rsp->result_status;
  } else {
    data = rsp->data;
  }
  Notify();
  return Status();
}

// ==================== ConnectorNcclSendBlocks ====================

Status ConnectorNcclSendBlocksRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeString(group_key, ptr, *bytes);
  SerializePod<int>(peer_rank, ptr, *bytes);
  SerializeVector<int>(block_ids, ptr, *bytes);
  SerializeVector<int64_t>(offsets, ptr, *bytes);
  SerializeVector<size_t>(counts, ptr, *bytes);
  SerializeEnum<DataType>(dtype, ptr, *bytes);
  return Status();
}

Status ConnectorNcclSendBlocksRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  group_key = DeserializeString(ptr, offset);
  peer_rank = DeserializePod<int>(ptr, offset);
  DeserializeVector<int>(block_ids, ptr, offset);
  DeserializeVector<int64_t>(offsets, ptr, offset);
  DeserializeVector<size_t>(counts, ptr, offset);
  dtype = DeserializeEnum<DataType>(ptr, offset);
  return Status();
}

Status ConnectorNcclSendBlocksResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(result_status, ptr, *bytes);
  return Status();
}

Status ConnectorNcclSendBlocksResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  result_status = DeserializeStatus(ptr, offset);
  return Status();
}

#ifdef ENABLE_CUDA
void ConnectorNcclSendBlocksCallable::SetNcclExecutorHandler(std::shared_ptr<NcclExecutorHandler> handler) {
  nccl_handler_ = handler;
}
#endif

Status ConnectorNcclSendBlocksCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
#ifdef ENABLE_CUDA
  const ConnectorNcclSendBlocksRequest* req = reinterpret_cast<const ConnectorNcclSendBlocksRequest*>(request);
  ConnectorNcclSendBlocksResponse* rsp = reinterpret_cast<ConnectorNcclSendBlocksResponse*>(response);

  if (!nccl_handler_) {
    rsp->result_status = Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL handler not set");
    return rsp->result_status;
  }

  // NcclExecutorHandler::SendGroup内部会根据block_ids和offsets解析GPU指针并处理stream管理
  rsp->result_status =
      nccl_handler_->SendGroup(req->group_key, req->peer_rank, req->block_ids, req->offsets, req->counts, req->dtype);

  return Status();
#else
  return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "CUDA not enabled");
#endif
}

Status ConnectorNcclSendBlocksCallback::Call(const IpcResponse* response, IpcContext* context) {
  const ConnectorNcclSendBlocksResponse* rsp = reinterpret_cast<const ConnectorNcclSendBlocksResponse*>(response);
  if (!rsp->result_status.OK()) {
    KLLM_LOG_ERROR << "Connector NCCL send blocks error: " << rsp->result_status.GetMessage();
    status = rsp->result_status;
  }
  Notify();
  return Status();
}

// ==================== ConnectorNcclRecvBlocks ====================

Status ConnectorNcclRecvBlocksRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeString(group_key, ptr, *bytes);
  SerializePod<int>(peer_rank, ptr, *bytes);
  SerializeVector<int>(block_ids, ptr, *bytes);
  SerializeVector<int64_t>(offsets, ptr, *bytes);
  SerializeVector<size_t>(counts, ptr, *bytes);
  SerializeEnum<DataType>(dtype, ptr, *bytes);
  return Status();
}

Status ConnectorNcclRecvBlocksRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  group_key = DeserializeString(ptr, offset);
  peer_rank = DeserializePod<int>(ptr, offset);
  DeserializeVector<int>(block_ids, ptr, offset);
  DeserializeVector<int64_t>(offsets, ptr, offset);
  DeserializeVector<size_t>(counts, ptr, offset);
  dtype = DeserializeEnum<DataType>(ptr, offset);
  return Status();
}

Status ConnectorNcclRecvBlocksResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(result_status, ptr, *bytes);
  return Status();
}

Status ConnectorNcclRecvBlocksResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  result_status = DeserializeStatus(ptr, offset);
  return Status();
}

#ifdef ENABLE_CUDA
void ConnectorNcclRecvBlocksCallable::SetNcclExecutorHandler(std::shared_ptr<NcclExecutorHandler> handler) {
  nccl_handler_ = handler;
}
#endif

Status ConnectorNcclRecvBlocksCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
#ifdef ENABLE_CUDA
  const ConnectorNcclRecvBlocksRequest* req = reinterpret_cast<const ConnectorNcclRecvBlocksRequest*>(request);
  ConnectorNcclRecvBlocksResponse* rsp = reinterpret_cast<ConnectorNcclRecvBlocksResponse*>(response);

  if (!nccl_handler_) {
    rsp->result_status = Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL handler not set");
    return rsp->result_status;
  }

  // NcclExecutorHandler::RecvGroup内部会根据block_ids和offsets解析GPU指针并处理stream管理
  rsp->result_status =
      nccl_handler_->RecvGroup(req->group_key, req->peer_rank, req->block_ids, req->offsets, req->counts, req->dtype);

  return Status();
#else
  return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "CUDA not enabled");
#endif
}

Status ConnectorNcclRecvBlocksCallback::Call(const IpcResponse* response, IpcContext* context) {
  const ConnectorNcclRecvBlocksResponse* rsp = reinterpret_cast<const ConnectorNcclRecvBlocksResponse*>(response);
  if (!rsp->result_status.OK()) {
    KLLM_LOG_ERROR << "Connector NCCL recv blocks error: " << rsp->result_status.GetMessage();
    status = rsp->result_status;
  }
  Notify();
  return Status();
}

// ==================== ConnectorGetNcclCommInfo ====================

Status ConnectorGetNcclCommInfoRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeString(group_key, ptr, *bytes);
  return Status();
}

Status ConnectorGetNcclCommInfoRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  group_key = DeserializeString(ptr, offset);
  return Status();
}

Status ConnectorGetNcclCommInfoResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(result_status, ptr, *bytes);
  SerializePod<int>(rank, ptr, *bytes);
  SerializePod<int>(world_size, ptr, *bytes);
  return Status();
}

Status ConnectorGetNcclCommInfoResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  result_status = DeserializeStatus(ptr, offset);
  rank = DeserializePod<int>(ptr, offset);
  world_size = DeserializePod<int>(ptr, offset);
  return Status();
}

#ifdef ENABLE_CUDA
void ConnectorGetNcclCommInfoCallable::SetNcclExecutorHandler(std::shared_ptr<NcclExecutorHandler> handler) {
  nccl_handler_ = handler;
}
#endif

Status ConnectorGetNcclCommInfoCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
#ifdef ENABLE_CUDA
  const ConnectorGetNcclCommInfoRequest* req = reinterpret_cast<const ConnectorGetNcclCommInfoRequest*>(request);
  ConnectorGetNcclCommInfoResponse* rsp = reinterpret_cast<ConnectorGetNcclCommInfoResponse*>(response);

  if (!nccl_handler_) {
    rsp->result_status = Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL handler not set");
    return rsp->result_status;
  }

  rsp->result_status = nccl_handler_->GetCommGroupInfo(req->group_key, rsp->rank, rsp->world_size);
  return Status();
#else
  return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "CUDA not enabled");
#endif
}

Status ConnectorGetNcclCommInfoCallback::Call(const IpcResponse* response, IpcContext* context) {
  const ConnectorGetNcclCommInfoResponse* rsp = reinterpret_cast<const ConnectorGetNcclCommInfoResponse*>(response);
  if (!rsp->result_status.OK()) {
    KLLM_LOG_ERROR << "Connector get NCCL comm info error: " << rsp->result_status.GetMessage();
    status = rsp->result_status;
  } else {
    rank = rsp->rank;
    world_size = rsp->world_size;
  }
  Notify();
  return Status();
}

}  // namespace ksana_llm