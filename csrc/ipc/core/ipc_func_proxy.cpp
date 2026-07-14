/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/core/ipc_func_proxy.h"

#include <cstdlib>
#include <cstring>

#include "device/device_utils.h"
#include "distributed/packet_type.h"
#include "ipc/core/ipc_func_types.h"

namespace ksana_llm {

// The thread num of ipc long task.
const size_t LONG_TASK_THREAD_NUM = 4;

IpcFuncProxy::IpcFuncProxy(size_t executor_num, bool is_engine, int executor_id, bool initialize_device) {
  executor_num_ = executor_num;
  initialize_device_ = initialize_device;
  ipc_channel_manager_ = std::make_shared<IpcProcessChannelManager>(executor_num_, executor_id, is_engine);

  if (is_engine) {
    SetProcessRole(IpcRole::IPC_ROLE_ENGINE, executor_num_, executor_id);
  } else {
    SetProcessRole(IpcRole::IPC_ROLE_EXECUTOR, executor_num_, executor_id);
  }

  Initialize();
}

IpcFuncProxy::~IpcFuncProxy() { Destroy(); }

void IpcFuncProxy::UnlinkIpcResource() { ipc_channel_manager_->UnlinkIpcResource(); }

void IpcFuncProxy::SetProcessRole(IpcRole ipc_role, size_t executor_num, int executor_id) {
  ipc_role_ = ipc_role;
  ipc_executor_num_ = executor_num;
  ipc_executor_id_ = executor_id;

  size_t channel_size = (ipc_role_ == IpcRole::IPC_ROLE_ENGINE) ? executor_num_ : 1;
}

void IpcFuncProxy::Initialize() {
  ipc_channel_manager_->Initialize();

  ipc_exec_threads_.resize(ipc_executor_num_);
  if (ipc_role_ == IpcRole::IPC_ROLE_ENGINE) {
    for (size_t executor_id = 0; executor_id < ipc_executor_num_; ++executor_id) {
      ipc_exec_threads_[executor_id] =
          std::unique_ptr<std::thread>(new std::thread(&IpcFuncProxy::ExecuteIpcPackets, this, executor_id,
                                                       ipc_channel_manager_->GetExecutor2EngineChannel(executor_id)));
    }
  } else {
    ipc_exec_threads_[ipc_executor_id_] = std::unique_ptr<std::thread>(
        new std::thread(&IpcFuncProxy::ExecuteIpcPackets, this, ipc_executor_id_,
                        ipc_channel_manager_->GetEngine2ExecutorChannel(ipc_executor_id_)));
    ipc_broadcast_exec_thread_ = std::unique_ptr<std::thread>(
        new std::thread(&IpcFuncProxy::ExecuteIpcPackets, this, -1, ipc_channel_manager_->GetBroadcastChannel()));
  }

  ipc_long_task_exec_threads_.resize(LONG_TASK_THREAD_NUM);
  for (size_t i = 0; i < LONG_TASK_THREAD_NUM; ++i) {
    ipc_long_task_exec_threads_[i] =
        std::unique_ptr<std::thread>(new std::thread(&IpcFuncProxy::ExecuteIpcLongTasks, this));
  }
}

void IpcFuncProxy::Destroy() {
  if (destroyed_) {
    return;
  }
  destroyed_ = true;
  stopped_ = true;

  // broadcast.
  ipc_channel_manager_->GetBroadcastChannel()->Stop();

  if (ipc_role_ == IpcRole::IPC_ROLE_ENGINE) {
    for (size_t executor_id = 0; executor_id < ipc_executor_num_; ++executor_id) {
      ipc_channel_manager_->GetEngine2ExecutorChannel(executor_id)->Stop();
      ipc_channel_manager_->GetExecutor2EngineChannel(executor_id)->Stop();

      if (ipc_exec_threads_[executor_id]->joinable()) {
        ipc_exec_threads_[executor_id]->join();
      }
    }
  } else {
    if (ipc_broadcast_exec_thread_->joinable()) {
      ipc_broadcast_exec_thread_->join();
    }

    if (ipc_exec_threads_[ipc_executor_id_]->joinable()) {
      ipc_exec_threads_[ipc_executor_id_]->join();
    }

    ipc_channel_manager_->GetEngine2ExecutorChannel(ipc_executor_id_)->Stop();
    ipc_channel_manager_->GetExecutor2EngineChannel(ipc_executor_id_)->Stop();
  }

  ipc_long_task_queue_.Stop();
  for (size_t i = 0; i < LONG_TASK_THREAD_NUM; ++i) {
    if (ipc_long_task_exec_threads_[i] && ipc_long_task_exec_threads_[i]->joinable()) {
      ipc_long_task_exec_threads_[i]->join();
    }
  }
}

Status IpcFuncProxy::Invoke(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  context->exec_seq = id_generator_.Gen();
  if (context->from_rank >= ipc_executor_num_ || context->to_rank >= ipc_executor_num_) {
    return Status(RET_INVALID_ARGUMENT,
                  fmt::format("IpcFuncProxy error, wrong executor {} or {}.", context->from_rank, context->to_rank));
  }

  if (context->to_rank < 0 && !context->is_async) {
    return Status(RET_INVALID_ARGUMENT, "IpcFuncProxy error, sync mode support only one callee.");
  }

  SendIpcRequest(request, context);

  // If in sync mode, wait until the response received.
  if (!context->is_async) {
    if (!response) {
      return Status(RET_INVALID_ARGUMENT, "IpcFuncProxy error, response can not be null in sync mode.");
    }

    std::shared_ptr<IpcNotifier> ipc_notifier = std::make_shared<IpcNotifier>(response);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      ipc_notifiers_[context->exec_seq] = ipc_notifier;
    }

    WaitingUntilIpcFuncFinished(ipc_notifier);

    // Clear notifier when finished.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ipc_notifiers_.erase(context->exec_seq);
    }
  }

  return Status();
}

Status IpcFuncProxy::WaitingUntilIpcFuncFinished(std::shared_ptr<IpcNotifier>& ipc_notifier) {
  std::unique_lock<std::mutex> lock(ipc_notifier->mutex);
  ipc_notifier->cv.wait(lock, [this, &ipc_notifier] { return ipc_notifier->finished; });

  return Status();
}

Status IpcFuncProxy::NotifyWaitingIpcFunc(std::shared_ptr<IpcNotifier>& ipc_notifier) {
  std::unique_lock<std::mutex> lock(ipc_notifier->mutex);
  ipc_notifier->finished = true;
  ipc_notifier->cv.notify_all();

  return Status();
}

void IpcFuncProxy::SendIpcRequest(const IpcRequest* request, IpcContext* context) {
  std::shared_ptr<IpcProcessChannel> ipc_channel = nullptr;
  if (ipc_role_ == IpcRole::IPC_ROLE_ENGINE) {
    if (context->to_rank < 0) {
      ipc_channel = ipc_channel_manager_->GetBroadcastChannel();
    } else {
      ipc_channel = ipc_channel_manager_->GetEngine2ExecutorChannel(context->to_rank);
    }
  } else {
    ipc_channel = ipc_channel_manager_->GetExecutor2EngineChannel(context->from_rank);
  }

  if (!ipc_channel->WaitWritableAndLock()) {
    KLLM_LOG_WARNING << "SendIpcRequest error, channel stopped, break..";
    return;
  }

  size_t* size_ptr = reinterpret_cast<size_t*>(ipc_channel->GetWritePtr());
  IpcContext* context_ptr = reinterpret_cast<IpcContext*>(size_ptr + 1);
  void* data_ptr = reinterpret_cast<void*>(context_ptr + 1);

  memcpy(context_ptr, context, sizeof(IpcContext));
  request->Serialize(data_ptr, size_ptr);
  *size_ptr += sizeof(IpcContext);
  context_ptr->is_response = false;

  if (!ipc_channel->WriteAndUnlock()) {
    KLLM_LOG_WARNING << "SendIpcRequest error, channel stopped, break..";
    return;
  }
}

void IpcFuncProxy::SendIpcResponse(const IpcResponse* response, IpcContext* context) {
  std::shared_ptr<IpcProcessChannel> ipc_channel = nullptr;

  int to_rank = (context->to_rank >= 0) ? context->to_rank : ipc_executor_id_;
  if (ipc_role_ == IpcRole::IPC_ROLE_ENGINE) {
    ipc_channel = ipc_channel_manager_->GetEngine2ExecutorChannel(context->from_rank);
  } else {
    ipc_channel = ipc_channel_manager_->GetExecutor2EngineChannel(to_rank);
  }

  if (!ipc_channel->WaitWritableAndLock()) {
    KLLM_LOG_WARNING << "SendIpcResponse error, channel stopped, break..";
    return;
  }

  size_t* size_ptr = reinterpret_cast<size_t*>(ipc_channel->GetWritePtr());
  IpcContext* context_ptr = reinterpret_cast<IpcContext*>(size_ptr + 1);
  void* data_ptr = reinterpret_cast<void*>(context_ptr + 1);

  memcpy(context_ptr, context, sizeof(IpcContext));
  response->Serialize(data_ptr, size_ptr);
  *size_ptr += sizeof(IpcContext);
  context_ptr->is_response = true;

  // Reset to_rank to real, so the broadcast caller knows who reply it.
  context_ptr->to_rank = to_rank;

  if (!ipc_channel->WriteAndUnlock()) {
    KLLM_LOG_WARNING << "SendIpcResponse error, channel stopped, break..";
    return;
  }
}

void IpcFuncProxy::ExecuteIpcPacketsImpl(int executor_id, int current_rank, IpcContext* context_ptr, void* data_ptr) {
  // Deserialize and execution.
  if (!context_ptr->is_response) {
    int func_info_id = context_ptr->share_callable ? -1 : executor_id;
    IpcFuncInfo& ipc_func_info = GetIpcFuncInfo(context_ptr->func_id, func_info_id);

    // Heavy tasks are executed by multiple threads concurrently, so we must lock
    // to protect shared request/response objects even when share_callable is false.
    // Normal tasks are executed serially in a single thread, so no lock is needed.
    bool need_lock = context_ptr->share_callable || context_ptr->is_heavy;

    if (need_lock) {
      ipc_func_info.mutex->lock();
    }

    // This will change ipc_func_info.request state, so it is not thread-safe.
    ipc_func_info.request->Deserialize(data_ptr);
    ipc_func_info.request->role = ipc_role_;
    ipc_func_info.request->rank = current_rank;
    ipc_func_info.callable->Call(ipc_func_info.request.get(), ipc_func_info.response.get(), context_ptr);

    // Send response only if not in oneway invocation.
    if (!context_ptr->is_oneway) {
      SendIpcResponse(ipc_func_info.response.get(), context_ptr);
    }

    if (need_lock) {
      ipc_func_info.mutex->unlock();
    }
  } else {
    int func_info_id = context_ptr->share_callback ? -1 : executor_id;
    IpcFuncInfo& ipc_func_info = GetIpcFuncInfo(context_ptr->func_id, func_info_id);

    // async mode, callback.
    if (context_ptr->is_async) {
      // Heavy task callbacks may also be executed concurrently
      bool need_lock = context_ptr->share_callback || context_ptr->is_heavy;

      if (need_lock) {
        ipc_func_info.mutex->lock();
      }

      ipc_func_info.response->Deserialize(data_ptr);
      ipc_func_info.response->role = ipc_role_;
      ipc_func_info.response->rank = current_rank;
      ipc_func_info.callback->Call(ipc_func_info.response.get(), context_ptr);

      if (need_lock) {
        ipc_func_info.mutex->unlock();
      }

    } else {
      // sync mode, weakup waiting caller.
      std::shared_ptr<IpcNotifier> ipc_notifier = GetIpcNotifier(context_ptr->exec_seq);
      ipc_notifier->response->Deserialize(data_ptr);
      ipc_notifier->response->role = ipc_role_;
      ipc_notifier->response->rank = current_rank;
      NotifyWaitingIpcFunc(ipc_notifier);
    }
  }
}

void IpcFuncProxy::AppendLongTaskQueue(int executor_id, void* serialized_packet, size_t bytes) {
  IpcTask ipc_task;
  ipc_task.executor_id = executor_id;
  ipc_task.packet = malloc(bytes);
  if (!ipc_task.packet) {
    KLLM_THROW(fmt::format("AppendLongTaskQueue error, malloc {} bytes error.", bytes));
  }

  memcpy(ipc_task.packet, serialized_packet, bytes);
  ipc_long_task_queue_.Put(ipc_task);
}

Status IpcFuncProxy::ExecuteIpcLongTasks() {
  int current_rank = 0;
  if (ipc_role_ == IpcRole::IPC_ROLE_EXECUTOR) {
    if (initialize_device_) {
      SetDevice(ipc_executor_id_);
    }
    current_rank = ipc_executor_id_;
  }

  while (!stopped_) {
    IpcTask ipc_task = ipc_long_task_queue_.Get();
    void* packet = ipc_task.packet;
    if (!packet) {
      KLLM_LOG_INFO << "ExecuteIpcLongTasks stopped.";
      break;
    }

    IpcContext* context_ptr = reinterpret_cast<IpcContext*>(packet);
    void* data_ptr = reinterpret_cast<void*>(context_ptr + 1);

    int executor_id = ipc_task.executor_id;
    ExecuteIpcPacketsImpl(executor_id, current_rank, context_ptr, data_ptr);
    free(packet);
  }
  return Status();
}

Status IpcFuncProxy::ExecuteIpcPackets(int executor_id, std::shared_ptr<IpcProcessChannel> ipc_channel) {
  int current_rank = 0;
  if (ipc_role_ == IpcRole::IPC_ROLE_EXECUTOR) {
    if (initialize_device_) {
      SetDevice(ipc_executor_id_);
    }
    current_rank = ipc_executor_id_;
  }

  while (!stopped_) {
    if (!ipc_channel->WaitReadableAndLock()) {
      KLLM_LOG_WARNING << "ExecuteIpcPackets error, executor " << executor_id << " channel stopped, break..";
      break;
    }

    // The memory is shared by multiple executors, DO NOT modify anything here.
    size_t* size_ptr = reinterpret_cast<size_t*>(ipc_channel->GetReadPtr());
    IpcContext* context_ptr = reinterpret_cast<IpcContext*>(size_ptr + 1);
    void* data_ptr = reinterpret_cast<void*>(context_ptr + 1);

    if (!context_ptr->is_heavy) {
      ExecuteIpcPacketsImpl(executor_id, current_rank, context_ptr, data_ptr);
    } else {
      // The value of size_ptr is not including the sizeof(size_t) bytes itself.
      AppendLongTaskQueue(executor_id, context_ptr, *size_ptr);
    }

    if (!ipc_channel->ReadAndUnlock()) {
      KLLM_LOG_WARNING << "ExecuteIpcPackets error, channel stopped, break..";
      break;
    }
  }
  return Status();
}

IpcFuncInfo& IpcFuncProxy::GetIpcFuncInfo(int func_id, int executor_id) {
  if (ipc_func_infos_.find(func_id) == ipc_func_infos_.end()) {
    KLLM_THROW(fmt::format("GetIpcFuncInfo error, the func id {} is not existed.", func_id));
  }

  if (executor_id < 0) {
    return ipc_shared_func_infos_[func_id];
  }
  return ipc_func_infos_[func_id][executor_id];
}

std::shared_ptr<IpcNotifier> IpcFuncProxy::GetIpcNotifier(size_t exec_seq) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (ipc_notifiers_.find(exec_seq) == ipc_notifiers_.end()) {
    KLLM_THROW(fmt::format("GetIpcNotifier error, the exec seq {} is not existed.", exec_seq));
  }
  return ipc_notifiers_[exec_seq];
}

}  // namespace ksana_llm
