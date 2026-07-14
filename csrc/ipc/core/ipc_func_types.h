/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include "utils/status.h"

#define SHARED_MEMORY_NAME "/ksana_llm_shm"

#define CHANNEL_SLOT_NUM (2)
#define MAX_EXECUTOR_NUM (8)
#define CHANNEL_META_SIZE (1024 * 1024)
// NOTE(jinxcwu) 多模态embedding数据比较大，设置成256M
#define CHANNEL_DATA_SIZE (1024 * 1024 * 256)

namespace ksana_llm {

enum IpcRole { IPC_ROLE_ENGINE, IPC_ROLE_EXECUTOR };

// The context for a ipc invocation.
struct IpcContext {
  // The unique id for a ipc function.
  int func_id;

  // The unique sequence for a execution.
  size_t exec_seq;

  // The rank of from and to process, -1 means all, not support sync mode in this case.
  int from_rank;
  int to_rank;

  // Whether the invocation is asynchronous.
  bool is_async = true;

  // Whether it is a response packet.
  bool is_response = false;

  // Whehter it is a one-way invocation, no waiting and skip result if it is true.
  bool is_oneway = false;

  // Whether it is a heavy invoke, that should execute in a separate threadpool.
  bool is_heavy = false;

  // whether all executors share the same callable object.
  bool share_callable = false;

  // whether all executors share the same callback object.
  bool share_callback = false;
};

// ipc request.
class IpcRequest {
 public:
  virtual ~IpcRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const;
  virtual Status Deserialize(const void* ptr);

  // Current rank and role.
  int rank;
  IpcRole role;
};

// ipc response.
class IpcResponse {
 public:
  virtual ~IpcResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const;
  virtual Status Deserialize(const void* ptr);

  // Current rank and role.
  int rank;
  IpcRole role;
};

// ipc func define.
class IpcCallable {
 public:
  virtual ~IpcCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context);
};

// ipc callback.
class IpcCallback {
 public:
  virtual ~IpcCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context);
};

// The information of a ipc function.
struct IpcFuncInfo {
  std::shared_ptr<IpcRequest> request;
  std::shared_ptr<IpcResponse> response;
  std::shared_ptr<IpcCallable> callable;
  std::shared_ptr<IpcCallback> callback;
  std::shared_ptr<std::mutex> mutex;
};

// The serialized ipc packet, request or response, include contex.
struct IpcPacket {
  // in bytes, invalid packet if be zero.
  size_t size = 0;

  // including context part.
  char data[CHANNEL_DATA_SIZE];
};

// Used to weakup waiting rpc function.
struct IpcNotifier {
  explicit IpcNotifier(IpcResponse* ipc_response) { response = ipc_response; }

  std::mutex mutex;
  std::condition_variable cv;

  // The response pointer.
  IpcResponse* response = nullptr;

  // Whether rpc function is finished.
  bool finished = false;
};

}  // namespace ksana_llm
