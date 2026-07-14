/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_process_channel_manager.h"
#include "utils/blocking_queue.h"
#include "utils/id_generator.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

class IpcFuncProxy {
 public:
  IpcFuncProxy(size_t executor_num, bool is_engine, int executor_id, bool initialize_device = true);
  ~IpcFuncProxy();

  // Unlink shared memory after all executors attached.
  void UnlinkIpcResource();

  // Invoke the ipc function.
  Status Invoke(const IpcRequest* request, IpcResponse* response, IpcContext* context);

  // Set role of ipc, engine or executor, the index is executor id, not used for engine.
  void SetProcessRole(IpcRole ipc_role, size_t executor_num, int executor_id = -1);

  // Set function info of a ipc function.
  template <typename CallableType, typename RequestType, typename ResponseType, typename CallbackType>
  void SetFuncInfo(int func_id) {
    if (ipc_shared_func_infos_.find(func_id) != ipc_shared_func_infos_.end()) {
      KLLM_THROW(fmt::format("SetFuncInfo failed, the func id {} is already existed.", func_id));
    }

    ipc_func_infos_[func_id].resize(ipc_executor_num_);
    for (int executor_id = -1; executor_id < ipc_executor_num_; ++executor_id) {
      IpcFuncInfo ipc_func_info;
      ipc_func_info.callable = std::make_shared<CallableType>();
      ipc_func_info.request = std::make_shared<RequestType>();
      ipc_func_info.response = std::make_shared<ResponseType>();
      ipc_func_info.callback = std::make_shared<CallbackType>();
      ipc_func_info.mutex = std::make_shared<std::mutex>();

      if (executor_id == -1) {
        ipc_shared_func_infos_[func_id] = ipc_func_info;
      } else {
        ipc_func_infos_[func_id][executor_id] = ipc_func_info;
      }
    }
  }

  // Initialize or destroy the function proxy.
  void Initialize();
  void Destroy();

  template <typename CallableType>
  std::shared_ptr<CallableType> GetIpcFuncCallable(int func_id) {
    IpcFuncInfo& ipc_func_info = GetIpcFuncInfo(func_id, ipc_executor_id_);
    return std::dynamic_pointer_cast<CallableType>(ipc_func_info.callable);
  }

  template <typename CallableType>
  std::shared_ptr<CallableType> GetIpcFuncCallable(int func_id, int executor_id) {
    IpcFuncInfo& ipc_func_info = GetIpcFuncInfo(func_id, executor_id);
    return std::dynamic_pointer_cast<CallableType>(ipc_func_info.callable);
  }

  template <typename CallbackType>
  std::shared_ptr<CallbackType> GetIpcFuncCallback(int func_id) {
    IpcFuncInfo& ipc_func_info = GetIpcFuncInfo(func_id, ipc_executor_id_);
    return std::dynamic_pointer_cast<CallbackType>(ipc_func_info.callback);
  }

  template <typename CallbackType>
  std::shared_ptr<CallbackType> GetIpcFuncCallback(int func_id, int executor_id) {
    IpcFuncInfo& ipc_func_info = GetIpcFuncInfo(func_id, executor_id);
    return std::dynamic_pointer_cast<CallbackType>(ipc_func_info.callback);
  }

  template <typename CallableType>
  std::shared_ptr<CallableType> GetSharedIpcFuncCallable(int func_id) {
    IpcFuncInfo& ipc_func_info = GetIpcFuncInfo(func_id, -1);
    return std::dynamic_pointer_cast<CallableType>(ipc_func_info.callable);
  }

  template <typename CallbackType>
  std::shared_ptr<CallbackType> GetSharedIpcFuncCallback(int func_id) {
    IpcFuncInfo& ipc_func_info = GetIpcFuncInfo(func_id, -1);
    return std::dynamic_pointer_cast<CallbackType>(ipc_func_info.callback);
  }

 private:
  void SendIpcRequest(const IpcRequest* request, IpcContext* context);
  void SendIpcResponse(const IpcResponse* response, IpcContext* context);

  // Execute all packets, including received and callback, executor_id with -1 for broadcast.
  Status ExecuteIpcPackets(int executor_id, std::shared_ptr<IpcProcessChannel> ipc_channel);

  // Get ipc function info, executor_id with -1 for broadcast.
  IpcFuncInfo& GetIpcFuncInfo(int func_id, int executor_id);

  // Get notifier through execute sequence.
  std::shared_ptr<IpcNotifier> GetIpcNotifier(size_t exec_seq);

  // Waiting and notify ipc function.
  Status WaitingUntilIpcFuncFinished(std::shared_ptr<IpcNotifier>& ipc_notifier);
  Status NotifyWaitingIpcFunc(std::shared_ptr<IpcNotifier>& ipc_notifier);

  // Append packet to task queeu.
  void AppendLongTaskQueue(int executor_id, void* serialized_packet, size_t bytes);

  // Fetch task from queue, and execute it.
  Status ExecuteIpcLongTasks();

  // Fast mode, access shared memory and static func_info directly.
  void ExecuteIpcPacketsImpl(int executor_id, int current_rank, IpcContext* context_ptr, void* data_ptr);

 private:
  // An ipc task engity, executor id identify which channel if from.
  struct IpcTask {
    int executor_id = -1;
    void* packet = nullptr;
  };

 private:
  size_t executor_num_ = 0;

  // engine or executor.
  IpcRole ipc_role_ = IpcRole::IPC_ROLE_ENGINE;

  // -1 for engine.
  int ipc_executor_id_ = -1;
  int ipc_executor_num_ = 0;

  // Used to generate unique id.
  IdGenerator id_generator_;

  // Whether the ipc function should be stopped.
  bool stopped_ = false;

  // Whether Destroy() has been called (for idempotent shutdown).
  bool destroyed_ = false;

  // For test only, whether initialize device context.
  bool initialize_device_ = true;

  // The parser objects of all ipc functions.
  std::unordered_map<int, std::vector<IpcFuncInfo>> ipc_func_infos_;
  std::unordered_map<int, IpcFuncInfo> ipc_shared_func_infos_;

  // The channel manager.
  std::shared_ptr<IpcProcessChannelManager> ipc_channel_manager_ = nullptr;

  // The execute thread
  std::vector<std::unique_ptr<std::thread>> ipc_exec_threads_;

  // The threads used to execute heavy long task only.
  std::vector<std::unique_ptr<std::thread>> ipc_long_task_exec_threads_;

  // The serialized packet that need to executed.
  BlockingQueue<IpcTask> ipc_long_task_queue_;

  // The broadcast execute thread, only for executor.
  std::unique_ptr<std::thread> ipc_broadcast_exec_thread_ = nullptr;

  // exec seq id to notifier.
  std::unordered_map<size_t, std::shared_ptr<IpcNotifier>> ipc_notifiers_;

  // Protect the notifiers.
  std::mutex mutex_;
};

}  // namespace ksana_llm
