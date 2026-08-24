/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <pthread.h>

#include <atomic>
#include <mutex>
#include <vector>

#include "ipc/core/ipc_func_types.h"

namespace ksana_llm {

// A process channel, used to do some synchronize between processes.
class alignas(64) IpcProcessChannel {
 public:
  struct SharedData {
    pthread_mutex_t mutex;
    pthread_rwlock_t rwlock;

    pthread_cond_t read_cond;
    pthread_cond_t write_cond;

    // The item number in channel now.
    int used_slot_num = 0;

    // Whether the channel should stopped.
    volatile bool stopped = false;

    int write_offset = 0;
    int read_offset[MAX_EXECUTOR_NUM];
    int slot_read_write_flags[CHANNEL_SLOT_NUM][MAX_EXECUTOR_NUM];
  };

 public:
  IpcProcessChannel(uint8_t* meta_ptr, uint8_t* data_ptr, size_t executor_num, int executor_id, bool is_broadcast,
                    bool is_engine);
  ~IpcProcessChannel();

  // Wait until the channel is writable, return false if channel stopped.
  bool WaitWritableAndLock();

  // Wait until the channle is readable, return false if channel stopped.
  bool WaitReadableAndLock();

  // Write to channel. the readable will return true, return false if channel stopped
  bool WriteAndUnlock();

  // Read from channel, the writable will return true, return false if channel stopped
  bool ReadAndUnlock();

  // Get memory address of a channel.
  void* GetWritePtr();
  void* GetReadPtr();

  // Stop the process channel.
  void Stop();

 private:
  // Lock & unlock with error check.
  void Lock();
  void Unlock();

  bool CheckSlotReadFinish(int slot_id);
  bool CheckSlotReadable(int slot_id);
  void SetSlotWriteFinish(int slot_id);
  void SetSlotReadFinish(int slot_id);

 private:
  // The meta and data memory pointers.
  SharedData* meta_ptr_ = nullptr;
  uint8_t* data_ptr_ = nullptr;

  int executor_id_;
  size_t executor_num_;

  bool is_broadcast_ = false;
  std::vector<int> read_finished_flags_;
  std::vector<int> write_finished_flags_;

  // we should create or just attach ipc resource.
  bool is_engine_ = false;

  // 本进程内写端串行化：WaitWritable 在 Unlock(mutex) 与 wrlock 之间存在 TOCTOU，
  // 多线程并发写会把 used_slot_num 推过 CHANNEL_SLOT_NUM，覆盖未读 slot。
  // 同向 channel 通常只有一个写进程，进程内 mutex 即可关闭该窗口。
  std::mutex local_write_mutex_;
};

}  // namespace ksana_llm
