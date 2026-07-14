/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/core/ipc_process_channel_manager.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

#include "ipc/core/ipc_func_types.h"
#include "utils/logger.h"

namespace ksana_llm {

IpcProcessChannelManager::IpcProcessChannelManager(size_t executor_num, int executor_id, bool is_engine) {
  if (executor_num == 0 || executor_num >= 64) {
    KLLM_THROW(fmt::format("Invalid executor_num: {}, must be in range [1, 63]", executor_num));
  }

  if (!is_engine && (executor_id < 0 || executor_id >= executor_num)) {
    KLLM_THROW(fmt::format("Invalid executor_id: {}, must be in range [0, {})", executor_id, executor_num));
  }

  is_engine_ = is_engine;
  executor_id_ = executor_id;
  executor_num_ = executor_num;

  // get shared memory name from environment if existed.
  const char* ksana_ipc_name = std::getenv("KSANA_IPC_NAME");
  if (ksana_ipc_name) {
    shared_memory_name_ = "/";
    shared_memory_name_ += ksana_ipc_name;
  } else {
    pid_t engine_pid = is_engine ? getpid() : getppid();
    shared_memory_name_ = std::string(SHARED_MEMORY_NAME) + "_" + std::to_string(engine_pid);
  }
}

IpcProcessChannelManager::~IpcProcessChannelManager() {
  // Must destruct channels first, it's meta_ptr will be invalid after shm_fd_ is closed.
  engine2executor_channels_.clear();
  executor2engine_channels_.clear();
  broadcast_channel_.reset();

  if (shm_ptr_ != nullptr) {
    if (munmap(shm_ptr_, bytes_) == -1) {
      KLLM_LOG_ERROR << "~IpcProcessChannelManager error, munmap failed.";
    }
    shm_ptr_ = nullptr;
  }

  UnlinkIpcResource();

  if (shm_fd_ >= 0) {
    if (close(shm_fd_) == -1) {
      KLLM_LOG_ERROR << "~IpcProcessChannelManager error, close shm_fd failed.";
    }
    shm_fd_ = -1;
  }
}

void IpcProcessChannelManager::UnlinkIpcResource() {
  // skip error, succ only once if invoked multiple times.
  if (is_engine_) {
    shm_unlink(shared_memory_name_.c_str());
  }
}

void IpcProcessChannelManager::Initialize() {
  try {
    InitializeIpcMemory();
    InitializeIpcPtrs();
    InitializeIpcChannels();
  } catch (...) {
    if (shm_ptr_ != nullptr) {
      munmap(shm_ptr_, bytes_);
      shm_ptr_ = nullptr;
    }
    if (shm_fd_ >= 0) {
      close(shm_fd_);
      shm_fd_ = -1;
    }
    if (is_engine_) {
      shm_unlink(shared_memory_name_.c_str());
    }
    throw;
  }
}

size_t IpcProcessChannelManager::GetIpcBufferSize() const {
  // x2 for send and recv buffers, +1 for broadcast,
  return (executor_num_ + 1) * 2 * (CHANNEL_META_SIZE + CHANNEL_DATA_SIZE * CHANNEL_SLOT_NUM);
}

void IpcProcessChannelManager::InitializeIpcMemory() {
  bytes_ = GetIpcBufferSize();
  if (is_engine_) {
    // Remove old shared memory if existed.
    int fd = shm_open(shared_memory_name_.c_str(), O_RDONLY, 0);
    if (fd >= 0) {
      shm_unlink(shared_memory_name_.c_str());
      close(fd);
    }

    shm_fd_ = shm_open(shared_memory_name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
    if (shm_fd_ == -1) {
      KLLM_THROW(fmt::format("Create shared memory {} failed, shmopen error.", shared_memory_name_.c_str()));
    }

    if (ftruncate(shm_fd_, bytes_) == -1) {
      KLLM_THROW(fmt::format("Create shared memory {} failed, ftruncate error.", shared_memory_name_.c_str()));
    }
  } else {
    shm_fd_ = shm_open(shared_memory_name_.c_str(), O_RDWR, 0666);
    if (shm_fd_ == -1) {
      KLLM_THROW(fmt::format("Attach shared memory {} failed, shmopen error.", shared_memory_name_.c_str()));
    }
  }

  shm_ptr_ = reinterpret_cast<uint8_t*>(mmap(NULL, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0));
  if (shm_ptr_ == MAP_FAILED) {
    KLLM_THROW(fmt::format("Create shared memory {} failed, mmap error.", shared_memory_name_.c_str()));
  }

  KLLM_LOG_INFO << "Get shared memory " << shared_memory_name_ << "(" << bytes_
                << " bytes) with addr: " << reinterpret_cast<void*>(shm_ptr_) << ", pid: " << getpid();
}

void IpcProcessChannelManager::InitializeIpcPtrs() {
  engine2executor_meta_ptrs_.reserve(executor_num_);
  executor2engine_meta_ptrs_.reserve(executor_num_);
  engine2executor_data_ptrs_.reserve(executor_num_);
  executor2engine_data_ptrs_.reserve(executor_num_);

  // The shared memory layout:
  // +-----------------------------------------------------+-----------------------------------------------------+
  // |                   channel meta                      |                       channel data                  |
  // +-----+-----------------------+-----------------------+-----+-----------------------+-----------------------+
  // |broad|   engine_to_executor  |  executor_to_engine   |broad|   engine_to_executor  |  executor_to_engine   |
  // |-cast|         meta          |         meta          |-cast|         data          |          data         |
  // +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
  // | 0-N |  0  |  1  | ... |  N  |  0  |  1  | ... |  N  | 0-N |  0  |  1  | ... |  N  |  0  |  1  | ... |  N  |
  // +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
  broadcast_meta_ptr_ = shm_ptr_;
  broadcast_data_ptr_ = shm_ptr_ + ((1 + (executor_num_ * 2)) * CHANNEL_META_SIZE);

  uint8_t* engine2executor_meta_base_ptr = broadcast_meta_ptr_ + CHANNEL_META_SIZE;
  uint8_t* executor2engine_meta_base_ptr = engine2executor_meta_base_ptr + (executor_num_ * CHANNEL_META_SIZE);

  size_t channl_data_size = CHANNEL_DATA_SIZE * CHANNEL_SLOT_NUM;
  uint8_t* engine2executor_data_base_ptr = broadcast_data_ptr_ + channl_data_size;
  uint8_t* executor2engine_data_base_ptr = engine2executor_data_base_ptr + (executor_num_ * channl_data_size);

  for (size_t executor_idx = 0; executor_idx < executor_num_; ++executor_idx) {
    engine2executor_meta_ptrs_.push_back(engine2executor_meta_base_ptr + (executor_idx * CHANNEL_META_SIZE));
    executor2engine_meta_ptrs_.push_back(executor2engine_meta_base_ptr + (executor_idx * CHANNEL_META_SIZE));
    engine2executor_data_ptrs_.push_back(engine2executor_data_base_ptr + (executor_idx * channl_data_size));
    executor2engine_data_ptrs_.push_back(executor2engine_data_base_ptr + (executor_idx * channl_data_size));
  }
}

void IpcProcessChannelManager::InitializeIpcChannels() {
  broadcast_channel_ = std::make_shared<IpcProcessChannel>(broadcast_meta_ptr_, broadcast_data_ptr_, executor_num_,
                                                           executor_id_, true, is_engine_);

  engine2executor_channels_.resize(executor_num_);
  executor2engine_channels_.resize(executor_num_);
  if (is_engine_) {
    // For engine, create channels between engine and all executors.
    for (size_t executor_id = 0; executor_id < executor_num_; ++executor_id) {
      engine2executor_channels_[executor_id] = std::make_shared<IpcProcessChannel>(
          engine2executor_meta_ptrs_[executor_id], engine2executor_data_ptrs_[executor_id], executor_num_, executor_id,
          false, is_engine_);
      executor2engine_channels_[executor_id] = std::make_shared<IpcProcessChannel>(
          executor2engine_meta_ptrs_[executor_id], executor2engine_data_ptrs_[executor_id], executor_num_, executor_id,
          false, is_engine_);
    }
  } else {
    // For executor, just create channels between engine and current executor, leave nullptr for other executors.
    engine2executor_channels_[executor_id_] = std::make_shared<IpcProcessChannel>(
        engine2executor_meta_ptrs_[executor_id_], engine2executor_data_ptrs_[executor_id_], executor_num_, executor_id_,
        false, is_engine_);
    executor2engine_channels_[executor_id_] = std::make_shared<IpcProcessChannel>(
        executor2engine_meta_ptrs_[executor_id_], executor2engine_data_ptrs_[executor_id_], executor_num_, executor_id_,
        false, is_engine_);
  }
}

std::shared_ptr<IpcProcessChannel>& IpcProcessChannelManager::GetEngine2ExecutorChannel(int device_id) {
  if (device_id < 0 || device_id >= executor_num_) {
    KLLM_THROW(fmt::format("Invalid device_id: {}, executor_num: {}", device_id, executor_num_));
  }

  if (!engine2executor_channels_[device_id]) {
    KLLM_THROW(fmt::format("Channel for device {} is not initialized", device_id));
  }

  return engine2executor_channels_[device_id];
}

std::shared_ptr<IpcProcessChannel>& IpcProcessChannelManager::GetExecutor2EngineChannel(int device_id) {
  if (device_id < 0 || device_id >= executor_num_) {
    KLLM_THROW(fmt::format("Invalid device_id: {}, executor_num: {}", device_id, executor_num_));
  }

  if (!executor2engine_channels_[device_id]) {
    KLLM_THROW(fmt::format("Channel for device {} is not initialized", device_id));
  }

  return executor2engine_channels_[device_id];
}

std::shared_ptr<IpcProcessChannel>& IpcProcessChannelManager::GetBroadcastChannel() { return broadcast_channel_; }

}  // namespace ksana_llm
