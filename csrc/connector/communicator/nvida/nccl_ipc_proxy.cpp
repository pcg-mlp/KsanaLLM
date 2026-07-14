/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "connector/communicator/nvida/nccl_ipc_proxy.h"

#ifdef ENABLE_CUDA
#  include "ipc/core/ipc_func_utils.h"
#  include "ipc/functions/nccl_operations.h"
#  include "utils/logger.h"

namespace ksana_llm {

NcclIpcProxy::NcclIpcProxy(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper) : ipc_func_wrapper_(ipc_func_wrapper) {
  KLLM_LOG_INFO << "NcclIpcProxy created";
}

NcclIpcProxy::~NcclIpcProxy() { KLLM_LOG_INFO << "NcclIpcProxy destroyed"; }

Status NcclIpcProxy::InitializeCommGroup(const std::string& group_key, const ncclUniqueId& comm_id, int rank,
                                         int world_size, int executor_id) {
  if (!ipc_func_wrapper_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "IpcFuncWrapper not set");
  }

  KLLM_LOG_INFO << "NcclIpcProxy::InitializeCommGroup - group_key: " << group_key << ", rank: " << rank
                << ", world_size: " << world_size << ", executor_id: " << executor_id;

  return ipc_func_wrapper_->ConnectorInitNcclCommGroup(group_key, comm_id, rank, world_size, executor_id);
}

Status NcclIpcProxy::Send(const std::string& group_key, int peer_rank, const std::vector<uint8_t>& data, DataType dtype,
                          int executor_id) {
  if (!ipc_func_wrapper_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "IpcFuncWrapper not set");
  }

  KLLM_LOG_DEBUG << "NcclIpcProxy::Send - group_key: " << group_key << ", peer_rank: " << peer_rank
                 << ", data_size: " << data.size() << ", executor_id: " << executor_id;

  return ipc_func_wrapper_->ConnectorNcclSendData(group_key, peer_rank, data, dtype, executor_id);
}

Status NcclIpcProxy::Recv(const std::string& group_key, int peer_rank, size_t count, DataType dtype, int executor_id,
                          std::vector<uint8_t>& data) {
  if (!ipc_func_wrapper_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "IpcFuncWrapper not set");
  }

  KLLM_LOG_DEBUG << "NcclIpcProxy::Recv - group_key: " << group_key << ", peer_rank: " << peer_rank
                 << ", count: " << count << ", executor_id: " << executor_id;

  return ipc_func_wrapper_->ConnectorNcclRecvData(group_key, peer_rank, count, dtype, executor_id, data);
}

Status NcclIpcProxy::SendGroup(const std::string& group_key, int peer_rank, const std::vector<int>& block_ids,
                               const std::vector<int64_t>& offsets, const std::vector<size_t>& counts, DataType dtype,
                               int executor_id) {
  if (!ipc_func_wrapper_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "IpcFuncWrapper not set");
  }

  KLLM_LOG_DEBUG << "NcclIpcProxy::SendGroup - group_key: " << group_key << ", peer_rank: " << peer_rank
                 << ", buffer_count: " << block_ids.size() << ", executor_id: " << executor_id;

  return ipc_func_wrapper_->ConnectorNcclSendBlocks(group_key, peer_rank, block_ids, offsets, counts, dtype,
                                                    executor_id);
}

Status NcclIpcProxy::RecvGroup(const std::string& group_key, int peer_rank, const std::vector<int>& block_ids,
                               const std::vector<int64_t>& offsets, const std::vector<size_t>& counts, DataType dtype,
                               int executor_id) {
  if (!ipc_func_wrapper_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "IpcFuncWrapper not set");
  }

  KLLM_LOG_DEBUG << "NcclIpcProxy::RecvGroup - group_key: " << group_key << ", peer_rank: " << peer_rank
                 << ", buffer_count: " << block_ids.size() << ", executor_id: " << executor_id;

  return ipc_func_wrapper_->ConnectorNcclRecvBlocks(group_key, peer_rank, block_ids, offsets, counts, dtype,
                                                    executor_id);
}

Status NcclIpcProxy::GetCommGroupInfo(const std::string& group_key, int executor_id, int& rank, int& world_size) {
  if (!ipc_func_wrapper_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "IpcFuncWrapper not set");
  }

  KLLM_LOG_DEBUG << "NcclIpcProxy::GetCommGroupInfo - group_key: " << group_key << ", executor_id: " << executor_id;

  return ipc_func_wrapper_->ConnectorGetNcclCommInfo(group_key, executor_id, rank, world_size);
}

}  // namespace ksana_llm

#endif  // ENABLE_CUDA