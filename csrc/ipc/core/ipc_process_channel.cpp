/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/core/ipc_process_channel.h"

#include <pthread.h>

#include "utils/logger.h"

namespace ksana_llm {

IpcProcessChannel::IpcProcessChannel(uint8_t* meta_ptr, uint8_t* data_ptr, size_t executor_num, int executor_id,
                                     bool is_broadcast, bool is_engine) {
  meta_ptr_ = reinterpret_cast<SharedData*>(meta_ptr);
  data_ptr_ = data_ptr;

  is_engine_ = is_engine;
  executor_id_ = executor_id;
  executor_num_ = executor_num;
  is_broadcast_ = is_broadcast;
  read_finished_flags_.resize(executor_num_, 2);
  write_finished_flags_.resize(executor_num_, 1);

  if (executor_num_ >= 64) {
    throw std::runtime_error(fmt::format("Support at most 64 executors, got {}.", executor_num_));
  }

  if (!is_engine_ && (executor_id_ < 0 || executor_id_ >= executor_num_)) {
    throw std::runtime_error(fmt::format("Invalid executor_id: {}", executor_id_));
  }

  if (is_engine_) {
    // Inplace construction.
    new (meta_ptr_) SharedData();
    memset(meta_ptr_->read_offset, 0, MAX_EXECUTOR_NUM * sizeof(int));
    memset(meta_ptr_->slot_read_write_flags, 0, CHANNEL_SLOT_NUM * MAX_EXECUTOR_NUM * sizeof(int));

    // ipc mutex.
    pthread_mutexattr_t mutex_attr;
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
    if (pthread_mutex_init(&meta_ptr_->mutex, &mutex_attr) != 0) {
      KLLM_THROW("Create process channel failed, pthread_mutex_init error.");
    }

    // ipc rwlock
    pthread_rwlockattr_t rwlock_attr;
    pthread_rwlockattr_init(&rwlock_attr);
    pthread_rwlockattr_setpshared(&rwlock_attr, PTHREAD_PROCESS_SHARED);
    if (pthread_rwlock_init(&meta_ptr_->rwlock, &rwlock_attr) != 0) {
      KLLM_THROW("Create process channel failed, pthread_rwlock_init error.");
    }

    // ipc condition variable.
    pthread_condattr_t cond_attr;
    pthread_condattr_init(&cond_attr);
    pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
    if (pthread_cond_init(&meta_ptr_->read_cond, &cond_attr) != 0 ||
        pthread_cond_init(&meta_ptr_->write_cond, &cond_attr) != 0) {
      KLLM_THROW("Create process channel failed, pthread_cond_init error.");
    }

    pthread_mutexattr_destroy(&mutex_attr);
    pthread_condattr_destroy(&cond_attr);
  }
}

IpcProcessChannel::~IpcProcessChannel() {
  if (is_engine_ && meta_ptr_) {
    pthread_mutex_destroy(&meta_ptr_->mutex);
    pthread_rwlock_destroy(&meta_ptr_->rwlock);
    pthread_cond_destroy(&meta_ptr_->read_cond);
    pthread_cond_destroy(&meta_ptr_->write_cond);
  }
}

bool IpcProcessChannel::CheckSlotReadFinish(int slot_id) {
  return !memcmp(meta_ptr_->slot_read_write_flags[slot_id], read_finished_flags_.data(), executor_num_ * sizeof(int));
}

bool IpcProcessChannel::CheckSlotReadable(int slot_id) {
  return meta_ptr_->slot_read_write_flags[slot_id][executor_id_] == 1;
}

void IpcProcessChannel::SetSlotWriteFinish(int slot_id) {
  memcpy(meta_ptr_->slot_read_write_flags[slot_id], write_finished_flags_.data(), executor_num_ * sizeof(int));
}

void IpcProcessChannel::SetSlotReadFinish(int slot_id) { meta_ptr_->slot_read_write_flags[slot_id][executor_id_] = 2; }

bool IpcProcessChannel::WaitWritableAndLock() {
  // Loop because predicate must be re-validated after we own the rwlock:
  // multiple writers can pass the `used_slot_num < CHANNEL_SLOT_NUM` check
  // while holding the mutex, all release the mutex, then serialize on the
  // rwlock. Without a re-check the trailing writers will write into already-
  // occupied slots and over-fill the ring (observed: seq 155648 lost,
  // seq 155392 read twice on Engine2Executor[2] under concurrent BatchSwap).
  while (true) {
    Lock();
    // If in broadcast mode, wait util all executor finish reading.
    while (meta_ptr_->used_slot_num >= CHANNEL_SLOT_NUM && !meta_ptr_->stopped) {
      pthread_cond_wait(&meta_ptr_->write_cond, &meta_ptr_->mutex);
    }
    if (meta_ptr_->stopped) {
      Unlock();
      return false;
    }
    Unlock();

    if (pthread_rwlock_wrlock(&meta_ptr_->rwlock) != 0) {
      KLLM_THROW("WaitWritableAndLock error, pthread_rwlock_wrlock failed.");
    }

    // Re-check predicate under mutex while holding wrlock to close the
    // TOCTOU window. Lock order mutex→rwlock is consistent with
    // WriteAndUnlock / ReadAndUnlock (rwlock-held → take mutex).
    Lock();
    if (meta_ptr_->used_slot_num < CHANNEL_SLOT_NUM && !meta_ptr_->stopped) {
      Unlock();
      // Note: rwlock is not released, the caller owns it.
      return true;
    }
    bool was_stopped = meta_ptr_->stopped;
    Unlock();
    pthread_rwlock_unlock(&meta_ptr_->rwlock);
    if (was_stopped) {
      return false;
    }
    // Predicate no longer holds; loop to wait on write_cond again.
  }
}

bool IpcProcessChannel::WaitReadableAndLock() {
  Lock();

  while (
      (meta_ptr_->used_slot_num <= 0 || (is_broadcast_ && !CheckSlotReadable(meta_ptr_->read_offset[executor_id_]))) &&
      !meta_ptr_->stopped) {
    pthread_cond_wait(&meta_ptr_->read_cond, &meta_ptr_->mutex);
  }

  if (meta_ptr_->stopped) {
    Unlock();
    return false;
  }

  Unlock();
  pthread_rwlock_rdlock(&meta_ptr_->rwlock);

  // Note: the lock is not released.
  return true;
}

bool IpcProcessChannel::WriteAndUnlock() {
  Lock();

  // The caller must hold the lock
  if (meta_ptr_->stopped) {
    Unlock();
    pthread_rwlock_unlock(&meta_ptr_->rwlock);
    return false;
  }

  SetSlotWriteFinish(meta_ptr_->write_offset);
  meta_ptr_->used_slot_num++;

  // advance write offset.
  if (++meta_ptr_->write_offset == CHANNEL_SLOT_NUM) {
    meta_ptr_->write_offset = 0;
  }

  // Notify reader.
  if (is_broadcast_) {
    pthread_cond_broadcast(&meta_ptr_->read_cond);
  } else {
    pthread_cond_signal(&meta_ptr_->read_cond);
  }

  Unlock();
  pthread_rwlock_unlock(&meta_ptr_->rwlock);
  return true;
}

bool IpcProcessChannel::ReadAndUnlock() {
  Lock();

  // The caller must hold the lock
  if (meta_ptr_->stopped) {
    Unlock();
    pthread_rwlock_unlock(&meta_ptr_->rwlock);
    return false;
  }

  // advance read offset.
  int current_read_slot = meta_ptr_->read_offset[executor_id_];
  if (++meta_ptr_->read_offset[executor_id_] == CHANNEL_SLOT_NUM) {
    meta_ptr_->read_offset[executor_id_] = 0;
  }

  if (is_broadcast_) {
    SetSlotReadFinish(current_read_slot);
    if (!CheckSlotReadFinish(current_read_slot)) {
      Unlock();
      pthread_rwlock_unlock(&meta_ptr_->rwlock);
      return true;
    }
  }

  meta_ptr_->used_slot_num--;

  // Notify writer.
  pthread_cond_signal(&meta_ptr_->write_cond);

  Unlock();
  pthread_rwlock_unlock(&meta_ptr_->rwlock);
  return true;
}

void* IpcProcessChannel::GetWritePtr() { return data_ptr_ + (meta_ptr_->write_offset * CHANNEL_DATA_SIZE); }

void* IpcProcessChannel::GetReadPtr() { return data_ptr_ + (meta_ptr_->read_offset[executor_id_] * CHANNEL_DATA_SIZE); }

void IpcProcessChannel::Stop() {
  // Wake up all blocking waiting.
  Lock();
  meta_ptr_->stopped = true;
  if (is_broadcast_) {
    pthread_cond_broadcast(&meta_ptr_->read_cond);
  } else {
    pthread_cond_signal(&meta_ptr_->read_cond);
  }
  pthread_cond_signal(&meta_ptr_->write_cond);
  Unlock();
}

void IpcProcessChannel::Lock() {
  if (pthread_mutex_lock(&meta_ptr_->mutex) != 0) {
    KLLM_THROW("Process channel Lock failed, pthread_mutex_lock error.");
  }
}

void IpcProcessChannel::Unlock() {
  if (pthread_mutex_unlock(&meta_ptr_->mutex) != 0) {
    KLLM_THROW("Process channel Unlock failed, pthread_mutex_unlock error.");
  }
}

}  // namespace ksana_llm
