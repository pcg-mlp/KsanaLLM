/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/core/ipc_func_proxy.h"

#include <cstdlib>

#include <algorithm>
#include <cstring>
#include <sstream>

#include "device/device_utils.h"
#include "distributed/packet_type.h"
#include "ipc/core/ipc_func_types.h"
#include "profiler/profile_event.h"
#include "utils/logger.h"

namespace ksana_llm {

IpcFuncProxy::IpcFuncProxy(size_t executor_num, bool is_engine, int executor_id, bool initialize_device,
                           size_t long_task_thread_num) {
  executor_num_ = executor_num;
  initialize_device_ = initialize_device;
  long_task_thread_num_ = std::max<size_t>(1, long_task_thread_num);
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

  ipc_long_task_exec_threads_.resize(long_task_thread_num_);
  for (size_t i = 0; i < long_task_thread_num_; ++i) {
    ipc_long_task_exec_threads_[i] =
        std::unique_ptr<std::thread>(new std::thread(&IpcFuncProxy::ExecuteIpcLongTasks, this));
  }
  KLLM_LOG_INFO << "long_task_thread_num=" << long_task_thread_num_
                << ", role=" << (ipc_role_ == IpcRole::IPC_ROLE_ENGINE ? "engine" : "executor")
                << ", executor_id=" << ipc_executor_id_;
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
    // 必须先 Stop 再 join：收包线程堵在 WaitReadableAndLock，未 Stop 时 join 会永久等待。
    ipc_channel_manager_->GetEngine2ExecutorChannel(ipc_executor_id_)->Stop();
    ipc_channel_manager_->GetExecutor2EngineChannel(ipc_executor_id_)->Stop();

    if (ipc_broadcast_exec_thread_->joinable()) {
      ipc_broadcast_exec_thread_->join();
    }

    if (ipc_exec_threads_[ipc_executor_id_]->joinable()) {
      ipc_exec_threads_[ipc_executor_id_]->join();
    }
  }

  ipc_long_task_queue_.Stop();
  for (size_t i = 0; i < long_task_thread_num_; ++i) {
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

  // Sync-mode notifier MUST be registered before SendIpcRequest. Under heavy
  // load the executor round-trip can complete before this thread would reach
  // a post-send registration block, leaving GetIpcNotifier with an empty map.
  std::shared_ptr<IpcNotifier> ipc_notifier;
  if (!context->is_async) {
    if (!response) {
      return Status(RET_INVALID_ARGUMENT, "IpcFuncProxy error, response can not be null in sync mode.");
    }
    ipc_notifier = std::make_shared<IpcNotifier>(response);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ipc_notifiers_[context->exec_seq] = ipc_notifier;
    }
  }

  if (!SendIpcRequest(request, context)) {
    if (ipc_notifier) {
      NotifyWaitingIpcFunc(ipc_notifier);
      std::lock_guard<std::mutex> lock(mutex_);
      ipc_notifiers_.erase(context->exec_seq);
    }
    return Status(RET_RUNTIME_FAILED, "SendIpcRequest failed, channel stopped");
  }

  // If in sync mode, wait until the response received.
  if (!context->is_async) {
    Status wait_status = WaitingUntilIpcFuncFinished(ipc_notifier);

    // Clear notifier when finished.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ipc_notifiers_.erase(context->exec_seq);
    }
    return wait_status;
  }

  return Status();
}

Status IpcFuncProxy::WaitingUntilIpcFuncFinished(std::shared_ptr<IpcNotifier>& ipc_notifier) {
  PROFILE_EVENT_SCOPE(ipc_sync_wait, "Ipc.WaitingUntilIpcFuncFinished");
  std::unique_lock<std::mutex> lock(ipc_notifier->mutex);
  // 仅等 finished；发送失败须 NotifyWaitingIpcFunc，对端不得在回包前 Destroy。
  ipc_notifier->cv.wait(lock, [&ipc_notifier] { return ipc_notifier->finished; });

  return Status();
}

Status IpcFuncProxy::NotifyWaitingIpcFunc(std::shared_ptr<IpcNotifier>& ipc_notifier) {
  std::unique_lock<std::mutex> lock(ipc_notifier->mutex);
  ipc_notifier->finished = true;
  ipc_notifier->cv.notify_all();

  return Status();
}

bool IpcFuncProxy::SendIpcRequest(const IpcRequest* request, IpcContext* context) {
  PROFILE_EVENT_SCOPE(ipc_send_request_total, "Ipc.SendIpcRequest.Total");
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

  {
    PROFILE_EVENT_SCOPE(ipc_send_request_wait, "Ipc.SendIpcRequest.WaitWritable");
    if (!ipc_channel->WaitWritableAndLock()) {
      KLLM_LOG_WARNING << "SendIpcRequest error, channel stopped, break..";
      return false;
    }
  }

  size_t* size_ptr = reinterpret_cast<size_t*>(ipc_channel->GetWritePtr());
  IpcContext* context_ptr = reinterpret_cast<IpcContext*>(size_ptr + 1);
  void* data_ptr = reinterpret_cast<void*>(context_ptr + 1);

  memcpy(context_ptr, context, sizeof(IpcContext));
  {
    PROFILE_EVENT_SCOPE(ipc_send_request_serialize, "Ipc.SendIpcRequest.Serialize");
    request->Serialize(data_ptr, size_ptr);
  }
  *size_ptr += sizeof(IpcContext);
  context_ptr->is_response = false;

  {
    PROFILE_EVENT_SCOPE(ipc_send_request_notify, "Ipc.SendIpcRequest.WriteAndUnlock");
    if (!ipc_channel->WriteAndUnlock()) {
      KLLM_LOG_WARNING << "SendIpcRequest error, channel stopped, break..";
      return false;
    }
  }
  return true;
}

void IpcFuncProxy::SendIpcResponse(const IpcResponse* response, IpcContext* context) {
  PROFILE_EVENT_SCOPE(ipc_send_response_total, "Ipc.SendIpcResponse.Total");
  std::shared_ptr<IpcProcessChannel> ipc_channel = nullptr;

  int to_rank = (context->to_rank >= 0) ? context->to_rank : ipc_executor_id_;
  if (ipc_role_ == IpcRole::IPC_ROLE_ENGINE) {
    ipc_channel = ipc_channel_manager_->GetEngine2ExecutorChannel(context->from_rank);
  } else {
    ipc_channel = ipc_channel_manager_->GetExecutor2EngineChannel(to_rank);
  }

  {
    PROFILE_EVENT_SCOPE(ipc_send_response_wait, "Ipc.SendIpcResponse.WaitWritable");
    if (!ipc_channel->WaitWritableAndLock()) {
      KLLM_LOG_WARNING << "SendIpcResponse error, channel stopped, break..";
      return;
    }
  }

  size_t* size_ptr = reinterpret_cast<size_t*>(ipc_channel->GetWritePtr());
  IpcContext* context_ptr = reinterpret_cast<IpcContext*>(size_ptr + 1);
  void* data_ptr = reinterpret_cast<void*>(context_ptr + 1);

  memcpy(context_ptr, context, sizeof(IpcContext));
  {
    PROFILE_EVENT_SCOPE(ipc_send_response_serialize, "Ipc.SendIpcResponse.Serialize");
    response->Serialize(data_ptr, size_ptr);
  }
  *size_ptr += sizeof(IpcContext);
  context_ptr->is_response = true;

  // Reset to_rank to real, so the broadcast caller knows who reply it.
  context_ptr->to_rank = to_rank;

  {
    PROFILE_EVENT_SCOPE(ipc_send_response_notify, "Ipc.SendIpcResponse.WriteAndUnlock");
    if (!ipc_channel->WriteAndUnlock()) {
      KLLM_LOG_WARNING << "SendIpcResponse error, channel stopped, break..";
      return;
    }
  }
}

void IpcFuncProxy::ExecuteIpcPacketsImpl(int executor_id, int current_rank, IpcContext* context_ptr, void* data_ptr) {
  PROFILE_EVENT_SCOPE(ipc_execute_packet, "Ipc.ExecuteIpcPacketsImpl");
  // Deserialize and execution.
  if (!context_ptr->is_response) {
    int func_info_id = context_ptr->share_callable ? -1 : executor_id;
    IpcFuncInfo& ipc_func_info = GetIpcFuncInfo(context_ptr->func_id, func_info_id);

    // share_callable：多线程/多路共用同一 request，必须加锁。
    // heavy 且非 share：每次执行私有 request/response，不加锁，允许多 long-task 线程并行 Call。
    // 其它（非 heavy、单 reader）：沿用共享 request，无需锁。
    const bool use_private_packet = context_ptr->is_heavy && !context_ptr->share_callable &&
                                    ipc_func_info.create_request && ipc_func_info.create_response;
    const bool need_lock = context_ptr->share_callable || (context_ptr->is_heavy && !use_private_packet);

    std::shared_ptr<IpcRequest> private_request;
    std::shared_ptr<IpcResponse> private_response;
    IpcRequest* request_ptr = ipc_func_info.request.get();
    IpcResponse* response_ptr = ipc_func_info.response.get();

    if (need_lock) {
      ipc_func_info.mutex->lock();
    } else if (use_private_packet) {
      private_request = ipc_func_info.create_request();
      private_response = ipc_func_info.create_response();
      request_ptr = private_request.get();
      response_ptr = private_response.get();
    }

    {
      PROFILE_EVENT_SCOPE(ipc_request_deserialize, "Ipc.ExecuteIpcPackets.RequestDeserialize");
      request_ptr->Deserialize(data_ptr);
    }
    request_ptr->role = ipc_role_;
    request_ptr->rank = current_rank;

    {
      PROFILE_EVENT_SCOPE(ipc_request_call, "Ipc.ExecuteIpcPackets.RequestCall");
      ipc_func_info.callable->Call(request_ptr, response_ptr, context_ptr);
    }

    if (!context_ptr->is_oneway) {
      SendIpcResponse(response_ptr, context_ptr);
    }

    if (need_lock) {
      ipc_func_info.mutex->unlock();
    }
  } else {
    int func_info_id = context_ptr->share_callback ? -1 : executor_id;
    IpcFuncInfo& ipc_func_info = GetIpcFuncInfo(context_ptr->func_id, func_info_id);

    if (context_ptr->is_async) {
      bool need_lock = context_ptr->share_callback || context_ptr->is_heavy;

      if (need_lock) {
        ipc_func_info.mutex->lock();
      }

      {
        PROFILE_EVENT_SCOPE(ipc_response_deserialize, "Ipc.ExecuteIpcPackets.ResponseDeserialize");
        ipc_func_info.response->Deserialize(data_ptr);
      }
      ipc_func_info.response->role = ipc_role_;
      ipc_func_info.response->rank = current_rank;
      {
        PROFILE_EVENT_SCOPE(ipc_response_callback, "Ipc.ExecuteIpcPackets.ResponseCallback");
        ipc_func_info.callback->Call(ipc_func_info.response.get(), context_ptr);
      }

      if (need_lock) {
        ipc_func_info.mutex->unlock();
      }

    } else {
      std::shared_ptr<IpcNotifier> ipc_notifier = GetIpcNotifier(context_ptr->exec_seq);
      {
        PROFILE_EVENT_SCOPE(ipc_response_deserialize_sync, "Ipc.ExecuteIpcPackets.ResponseDeserializeSync");
        ipc_notifier->response->Deserialize(data_ptr);
      }
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
    IpcTask ipc_task;
    {
      PROFILE_EVENT_SCOPE(ipc_long_wait, "Ipc.ExecuteIpcLongTasks.QueueGet");
      ipc_task = ipc_long_task_queue_.Get();
    }
    void* packet = ipc_task.packet;
    if (!packet) {
      KLLM_LOG_INFO << "ExecuteIpcLongTasks stopped.";
      break;
    }

    PROFILE_EVENT_SCOPE(ipc_long_iter, "Ipc.ExecuteIpcLongTasks.iter");
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
    {
      PROFILE_EVENT_SCOPE(ipc_recv_wait, "Ipc.ExecuteIpcPackets.WaitReadable");
      if (!ipc_channel->WaitReadableAndLock()) {
        KLLM_LOG_WARNING << "ExecuteIpcPackets error, executor " << executor_id << " channel stopped, break..";
        break;
      }
    }

    PROFILE_EVENT_SCOPE(ipc_recv_iter, "Ipc.ExecuteIpcPackets.iter");
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
    std::ostringstream oss;
    for (const auto& kv : ipc_notifiers_) {
      oss << kv.first << ",";
    }
    KLLM_LOG_ERROR << "[ipc-trace] GetIpcNotifier missing seq=" << exec_seq << ", role=" << ipc_role_
                   << ", map_size=" << ipc_notifiers_.size() << ", live_keys=[" << oss.str() << "]";
    KLLM_THROW(fmt::format("GetIpcNotifier error, the exec seq {} is not existed.", exec_seq));
  }
  return ipc_notifiers_[exec_seq];
}

}  // namespace ksana_llm
