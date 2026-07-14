/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ipc/core/ipc_process_channel.h"

namespace ksana_llm {

// Manage ipc invocations between processes.
class IpcProcessChannelManager {
 public:
  IpcProcessChannelManager(size_t executor_num, int executor_id, bool is_engine);
  ~IpcProcessChannelManager();

  // Initialize ipc resource.
  void Initialize();

  // Unlink shared memory after all executors attached.
  void UnlinkIpcResource();

  // Get process channel.
  std::shared_ptr<IpcProcessChannel>& GetEngine2ExecutorChannel(int device_id);
  std::shared_ptr<IpcProcessChannel>& GetExecutor2EngineChannel(int device_id);
  std::shared_ptr<IpcProcessChannel>& GetBroadcastChannel();

 private:
  // Get total buffer size for ipc.
  size_t GetIpcBufferSize() const;

  // Create global shared memory, used for ipc communication.
  void InitializeIpcMemory();

  // set data and meta ptrs for every ipc channel.
  void InitializeIpcPtrs();

  // Create all ipc channel.
  void InitializeIpcChannels();

 private:
  // The shared memory address
  int shm_fd_ = -1;
  uint8_t* shm_ptr_ = nullptr;
  size_t bytes_ = 0;

  std::string shared_memory_name_;

  // The num of child processes.
  size_t executor_num_ = 0;
  int executor_id_ = -1;

  // we should create or just attach ipc resource.
  bool is_engine_ = false;

 private:
  // engine to executor, the index is executor id.
  std::vector<std::shared_ptr<IpcProcessChannel>> engine2executor_channels_;

  // executor to engine executor, the index is executor id.
  std::vector<std::shared_ptr<IpcProcessChannel>> executor2engine_channels_;

  // The base pointer for meta buffer of every executor.
  std::vector<uint8_t*> engine2executor_meta_ptrs_;
  std::vector<uint8_t*> executor2engine_meta_ptrs_;

  // The base pointer for send and recv buffers of every executor.
  std::vector<uint8_t*> engine2executor_data_ptrs_;
  std::vector<uint8_t*> executor2engine_data_ptrs_;

  // For broadcast handle.
  uint8_t* broadcast_meta_ptr_ = nullptr;
  uint8_t* broadcast_data_ptr_ = nullptr;

  std::shared_ptr<IpcProcessChannel> broadcast_channel_ = nullptr;
};

}  // namespace ksana_llm
