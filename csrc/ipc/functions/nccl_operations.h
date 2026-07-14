/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#  include <cuda_runtime.h>
#  include <nccl.h>
#endif

#include <memory>
#include <string>
#include <vector>

#include "connector/communicator/communicator.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/status.h"
#include "utils/waiter.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
class NcclExecutorHandler;
#endif

// ==================== ConnectorInitNcclCommGroup ====================

class ConnectorInitNcclCommGroupRequest : public IpcRequest {
 public:
  virtual ~ConnectorInitNcclCommGroupRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  std::string group_key;
#ifdef ENABLE_CUDA
  ncclUniqueId comm_id;
#endif
  int rank;
  int world_size;
};

class ConnectorInitNcclCommGroupResponse : public IpcResponse {
 public:
  virtual ~ConnectorInitNcclCommGroupResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  Status result_status;
};

class ConnectorInitNcclCommGroupCallable : public IpcCallable {
 public:
  virtual ~ConnectorInitNcclCommGroupCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

#ifdef ENABLE_CUDA
  void SetNcclExecutorHandler(std::shared_ptr<NcclExecutorHandler> handler);
#endif

 private:
#ifdef ENABLE_CUDA
  std::shared_ptr<NcclExecutorHandler> nccl_handler_ = nullptr;
#endif
};

class ConnectorInitNcclCommGroupCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~ConnectorInitNcclCommGroupCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  Status status;
};

// ==================== ConnectorNcclSendData ====================

class ConnectorNcclSendDataRequest : public IpcRequest {
 public:
  virtual ~ConnectorNcclSendDataRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  std::string group_key;
  int peer_rank;
  std::vector<uint8_t> data;  // 实际数据内容（从engine的host内存传输）
  DataType dtype;
};

class ConnectorNcclSendDataResponse : public IpcResponse {
 public:
  virtual ~ConnectorNcclSendDataResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  Status result_status;
};

class ConnectorNcclSendDataCallable : public IpcCallable {
 public:
  virtual ~ConnectorNcclSendDataCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

#ifdef ENABLE_CUDA
  void SetNcclExecutorHandler(std::shared_ptr<NcclExecutorHandler> handler);
#endif

 private:
#ifdef ENABLE_CUDA
  std::shared_ptr<NcclExecutorHandler> nccl_handler_ = nullptr;
#endif
};

class ConnectorNcclSendDataCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~ConnectorNcclSendDataCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  Status status;
};

// ==================== ConnectorNcclRecvData ====================

class ConnectorNcclRecvDataRequest : public IpcRequest {
 public:
  virtual ~ConnectorNcclRecvDataRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  std::string group_key;
  int peer_rank;
  size_t count;  // 接收数据的大小
  DataType dtype;
};

class ConnectorNcclRecvDataResponse : public IpcResponse {
 public:
  virtual ~ConnectorNcclRecvDataResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  Status result_status;
  std::vector<uint8_t> data;  // 接收到的数据内容（传回engine的host内存）
};

class ConnectorNcclRecvDataCallable : public IpcCallable {
 public:
  virtual ~ConnectorNcclRecvDataCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

#ifdef ENABLE_CUDA
  void SetNcclExecutorHandler(std::shared_ptr<NcclExecutorHandler> handler);
#endif

 private:
#ifdef ENABLE_CUDA
  std::shared_ptr<NcclExecutorHandler> nccl_handler_ = nullptr;
#endif
};

class ConnectorNcclRecvDataCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~ConnectorNcclRecvDataCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  Status status;
  std::vector<uint8_t> data;  // 接收到的数据
};

// ==================== ConnectorNcclSendBlocks ====================

class ConnectorNcclSendBlocksRequest : public IpcRequest {
 public:
  virtual ~ConnectorNcclSendBlocksRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  std::string group_key;
  int peer_rank;
  std::vector<int> block_ids;    // Block IDs instead of GPU pointers
  std::vector<int64_t> offsets;  // Offsets within blocks
  std::vector<size_t> counts;
  DataType dtype;
};

class ConnectorNcclSendBlocksResponse : public IpcResponse {
 public:
  virtual ~ConnectorNcclSendBlocksResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  Status result_status;
};

class ConnectorNcclSendBlocksCallable : public IpcCallable {
 public:
  virtual ~ConnectorNcclSendBlocksCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

#ifdef ENABLE_CUDA
  void SetNcclExecutorHandler(std::shared_ptr<NcclExecutorHandler> handler);
#endif

 private:
#ifdef ENABLE_CUDA
  std::shared_ptr<NcclExecutorHandler> nccl_handler_ = nullptr;
#endif
};

class ConnectorNcclSendBlocksCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~ConnectorNcclSendBlocksCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  Status status;
};

// ==================== ConnectorNcclRecvBlocks ====================

class ConnectorNcclRecvBlocksRequest : public IpcRequest {
 public:
  virtual ~ConnectorNcclRecvBlocksRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  std::string group_key;
  int peer_rank;
  std::vector<int> block_ids;    // Block IDs instead of GPU pointers
  std::vector<int64_t> offsets;  // Offsets within blocks
  std::vector<size_t> counts;
  DataType dtype;
};

class ConnectorNcclRecvBlocksResponse : public IpcResponse {
 public:
  virtual ~ConnectorNcclRecvBlocksResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  Status result_status;
};

class ConnectorNcclRecvBlocksCallable : public IpcCallable {
 public:
  virtual ~ConnectorNcclRecvBlocksCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

#ifdef ENABLE_CUDA
  void SetNcclExecutorHandler(std::shared_ptr<NcclExecutorHandler> handler);
#endif

 private:
#ifdef ENABLE_CUDA
  std::shared_ptr<NcclExecutorHandler> nccl_handler_ = nullptr;
#endif
};

class ConnectorNcclRecvBlocksCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~ConnectorNcclRecvBlocksCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  Status status;
};

// ==================== ConnectorGetNcclCommInfo ====================

class ConnectorGetNcclCommInfoRequest : public IpcRequest {
 public:
  virtual ~ConnectorGetNcclCommInfoRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  std::string group_key;
};

class ConnectorGetNcclCommInfoResponse : public IpcResponse {
 public:
  virtual ~ConnectorGetNcclCommInfoResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  Status result_status;
  int rank;
  int world_size;
};

class ConnectorGetNcclCommInfoCallable : public IpcCallable {
 public:
  virtual ~ConnectorGetNcclCommInfoCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

#ifdef ENABLE_CUDA
  void SetNcclExecutorHandler(std::shared_ptr<NcclExecutorHandler> handler);
#endif

 private:
#ifdef ENABLE_CUDA
  std::shared_ptr<NcclExecutorHandler> nccl_handler_ = nullptr;
#endif
};

class ConnectorGetNcclCommInfoCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~ConnectorGetNcclCommInfoCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  Status status;
  int rank;
  int world_size;
};

}  // namespace ksana_llm