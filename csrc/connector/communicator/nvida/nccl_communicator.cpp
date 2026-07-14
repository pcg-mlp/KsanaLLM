/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "connector/communicator/nvida/nccl_communicator.h"
#include <numeric>
#include "base64.hpp"

namespace ksana_llm {
#ifdef ENABLE_CUDA
NcclCommunicator::~NcclCommunicator() { Shutdown(); }

void NcclCommunicator::DoSetReceiveCallback(const ReceiveCallback& callback) { receive_callback_ = callback; }

void NcclCommunicator::Shutdown() {
  // Stop the thread pool if it exists
  if (init_thread_pool_) {
    init_thread_pool_->Stop();
    init_thread_pool_.reset();
  }

  // Close all NCCL resources
  for (auto& [group_key, comm_group] : comm_groups_) {
    KLLM_LOG_INFO << "Shutting down NCCL group: " << group_key;
    // The NcclCommGroup destructor will handle resource cleanup
  }
  comm_groups_.clear();

  KLLM_LOG_INFO << "NcclCommunicator shutdown complete";
}

Status NcclCommunicator::Initialize() {
  // NCCL通信器的初始化
  KLLM_LOG_INFO << "NcclCommunicator initialized successfully";
  return Status();
}

NcclCommunicator::NcclRankInfo NcclCommunicator::CalcNcclRanks(GroupRole group_role, int node_rank, int world_size,
                                                               int device_count, int dev_id) {
  NcclRankInfo info{};
  // int global_world_size = 2 * world_size * device_count; // unused, remove warning
  int role = (group_role == GroupRole::PREFILL) ? 0 : 1;
  int peer_role = 1 - role;
  info.cur_rank = role * world_size + node_rank * device_count + dev_id;
  info.peer_rank = peer_role * world_size + node_rank * device_count + dev_id;
  info.send_rank = info.peer_rank;
  info.recv_rank = info.peer_rank;
  KLLM_LOG_DEBUG << "NCCL ranks: " << "cur_rank=" << info.cur_rank << ", peer_rank=" << info.peer_rank
                 << ", send_rank=" << info.send_rank << ", recv_rank=" << info.recv_rank;
  return info;
}

// Initialize NCCL communicators on executors via IPC
Status NcclCommunicator::InitializeNcclCommsOnExecutors(const std::string& group_key, const ncclUniqueId& comm_id) {
  if (!nccl_ipc_proxy_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL IPC proxy not set");
  }

  auto comm_group = GetCommunicatorGroup(group_key);
  if (!comm_group) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
  }

  // 计算全局 world size（两组）
  const int global_world_size = 2 * world_size_;

  // 预分配 device_ranks_ 空间，避免多线程写入时的竞争
  comm_group->device_ranks_.resize(device_count_);

  // 存储每个设备的状态结果
  std::vector<Status> device_statuses(device_count_);

  // 创建并启动线程池（如果尚未创建）
  if (!init_thread_pool_) {
    init_thread_pool_ = std::make_unique<ThreadPool>(device_count_);
    init_thread_pool_->Start();
  }

  // 使用 ThreadPool 并行初始化每个设备的 NCCL 通信器，防止串行初始化时卡住
  std::vector<std::future<void>> futures;
  futures.reserve(device_count_);

  for (int dev_id = 0; dev_id < device_count_; ++dev_id) {
    futures.emplace_back(init_thread_pool_->Submit(
        [this, &group_key, &comm_id, &comm_group, &device_statuses, global_world_size, dev_id]() {
          KLLM_LOG_DEBUG << "Initializing NCCL communicator on executor for device " << dev_id;
          NcclRankInfo ranks = CalcNcclRanks(config_.group_role, node_rank_, world_size_, device_count_, dev_id);

          // Store rank information in comm_group (每个线程写入不同的索引位置，无需加锁)
          comm_group->device_ranks_[dev_id] = ranks.cur_rank;

          // Initialize NCCL communicator on executor via IPC
          int executor_id = GetExecutorIdForDevice(dev_id);
          Status status =
              nccl_ipc_proxy_->InitializeCommGroup(group_key, comm_id, ranks.cur_rank, global_world_size, executor_id);

          if (!status.OK()) {
            KLLM_LOG_ERROR << "Failed to initialize NCCL communicator on executor " << executor_id << ": "
                           << status.GetMessage();
          }
          device_statuses[dev_id] = status;
        }));
  }

  // 等待所有任务完成
  for (auto& future : futures) {
    future.get();
  }

  // 检查所有设备的执行结果
  for (int dev_id = 0; dev_id < device_count_; ++dev_id) {
    if (!device_statuses[dev_id].OK()) {
      return device_statuses[dev_id];
    }
  }

  comm_group->world_size_ = global_world_size;
  KLLM_LOG_INFO << "Initialized NCCL communicators for group " << group_key << " on " << device_count_ << " executors";

  return Status();
}

bool NcclCommunicator::CheckCommGroupAvailable(const std::string& group_key) {
  {
    std::lock_guard<std::mutex> lock(comm_group_mutex_);
    auto existing_it = comm_groups_.find(group_key);
    if (existing_it != comm_groups_.end() && existing_it->second) {
      if (existing_it->second->IsActive()) {
        KLLM_LOG_DEBUG << "Communication group " << group_key << " is already active";
        return true;
      } else {
        comm_groups_.erase(group_key);
        return false;
      }
    }
    return false;
  }
}

void NcclCommunicator::CreateNcclCommGroup(const std::string& group_key, const std::string& encoded_value) {
  // Decode the NCCL ID from base64
  std::string decoded_value = base64::from_base64(encoded_value);
  if (decoded_value.size() != sizeof(ncclUniqueId)) {
    KLLM_LOG_ERROR << "Invalid NCCL ID size for group " << group_key;
    return;
  }
  ncclUniqueId comm_id;
  memset(&comm_id, 0, sizeof(ncclUniqueId));
  memcpy(&comm_id, decoded_value.data(), sizeof(ncclUniqueId));

  // Create a new communication group
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->comm_id = comm_id;
  comm_group->comm_id_str = encoded_value;

  // Store the new communication group
  {
    std::lock_guard<std::mutex> lock(comm_group_mutex_);
    comm_groups_[group_key] = std::move(comm_group);
  }
}


Status NcclCommunicator::ProcessHeartbeatData(
    const std::unordered_map<std::string, std::string>& comm_group_to_id,
    const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>& comm_group_to_address) {
  if (comm_group_to_id.empty()) {
    KLLM_LOG_DEBUG << "No communication groups provided in heartbeat response";
    return Status();
  }

  // 处理每个通信组
  for (const auto& [group_key, encoded_value] : comm_group_to_id) {
    // 如果通信ID为空且是prefill角色的主节点，则需要注册一个新的ID
    std::string actual_encoded_value = encoded_value;
    if (encoded_value.empty()) {
      if (config_.group_role == GroupRole::PREFILL && node_rank_ == 0) {
        // 生成新的NCCL ID
        ncclUniqueId nccl_id;
        memset(&nccl_id, 0, sizeof(ncclUniqueId));
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        std::string to_id_bytes(reinterpret_cast<const char*>(&nccl_id), sizeof(ncclUniqueId));
        actual_encoded_value = base64::to_base64(to_id_bytes);

        // 通过回调发送新的ID
        if (send_comm_id_callback_) {
          auto status = send_comm_id_callback_(group_key, actual_encoded_value);
          if (!status.OK()) {
            KLLM_LOG_ERROR << "Failed to send NCCL ID through callback: " << status.GetMessage();
            continue;
          }
        } else {
          KLLM_LOG_ERROR << "No send_comm_id_callback_ set, cannot register NCCL ID for group: " << group_key;
          continue;
        }
      } else {
        // 非主节点或非prefill角色不处理空ID组
        continue;
      }
    }

    // 检查并创建通信组
    if (!CheckCommGroupAvailable(group_key)) {
      KLLM_LOG_DEBUG << "Communication group " << group_key << " is not available";
      CreateNcclCommGroup(group_key, actual_encoded_value);
    }

    // 如果通信组存在但尚未激活，则初始化NCCL通信器
    auto it = comm_groups_.find(group_key);
    if (it != comm_groups_.end()) {
      auto& comm_group = it->second;
      if (comm_group && !comm_group->IsActive()) {
        comm_group->device_count_ = device_count_;
        Status status = InitializeNcclCommsOnExecutors(group_key, comm_group->comm_id);
        if (!status.OK()) {
          KLLM_LOG_ERROR << "Failed to initialize NCCL communicators for group " << group_key << ": "
                         << status.GetMessage();
          continue;
        }
        KLLM_LOG_INFO << "NCCL communication group " << group_key << " initialized";
      }
    }
  }

  return Status();
}

bool NcclCommunicator::IsConnectionReady(const std::string& group_key, int dev_id) const {
  // 使用 const_cast 来解决常量方法中使用非常量互斥锁的问题
  std::lock_guard<std::mutex> lock(*const_cast<std::mutex*>(&comm_group_mutex_));
  auto it = comm_groups_.find(group_key);
  if (it == comm_groups_.end() || !it->second) {
    return false;
  }
  // Check if the communication group is active and device is valid
  return it->second->IsActive() && dev_id >= 0 && dev_id < it->second->device_count_;
}

// Communicator 接口实现 - 发送数据
Status NcclCommunicator::Send(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id,
                              const void* buf, size_t count, DataType dtype) {
  return Send(group_key, src_dev_id, dst_dev_id, nullptr, buf, count, dtype);
}

// Communicator 接口实现 - 接收数据
Status NcclCommunicator::Recv(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id, void* buf,
                              size_t count, DataType dtype) {
  return Recv(group_key, src_dev_id, dst_dev_id, nullptr, buf, count, dtype);
}

// Communicator 接口实现 - 批量发送数据
Status NcclCommunicator::SendGroup(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id,
                                   const std::vector<int>& block_ids, const std::vector<int64_t>& offsets,
                                   const std::vector<size_t>& counts, DataType dtype) {
  return SendGroup(group_key, src_dev_id, dst_dev_id, nullptr, block_ids, offsets, counts, dtype);
}

// Communicator 接口实现 - 批量接收数据
Status NcclCommunicator::RecvGroup(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id,
                                   const std::vector<int>& block_ids, const std::vector<int64_t>& offsets,
                                   const std::vector<size_t>& counts, DataType dtype) {
  return RecvGroup(group_key, src_dev_id, dst_dev_id, nullptr, block_ids, offsets, counts, dtype);
}

Status NcclCommunicator::Send(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream,
                              const void* buf, size_t count, DataType dtype) {
  // TODO(zakwang): 暂未查清楚卡住原因，保持双方卡号一致
  local_dev_id = peer_dev_id;
  KLLM_LOG_DEBUG << "[NCCL] Sending data to group_key=" << group_key << ", local_dev_id=" << local_dev_id
                 << ", peer_dev_id=" << peer_dev_id << ", count=" << count;

  if (!nccl_ipc_proxy_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL IPC proxy not set");
  }

  // 将数据内容复制到vector中进行传输
  // buf指向的是主机内存（engine进程可以访问）
  std::vector<uint8_t> data(count);
  std::memcpy(data.data(), buf, count);

  // 将整个Send逻辑转发到executor端处理
  int executor_id = GetExecutorIdForDevice(local_dev_id);

  return nccl_ipc_proxy_->Send(group_key, peer_dev_id, data, dtype, executor_id);
}

Status NcclCommunicator::Recv(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream,
                              void* buf, size_t count, DataType dtype) {
  // TODO(zakwang): 暂未查清楚卡住原因，保持双方卡号一致
  peer_dev_id = local_dev_id;
  KLLM_LOG_DEBUG << "[NCCL] Receiving data from group_key=" << group_key << ", local_dev_id=" << local_dev_id
                 << ", peer_dev_id=" << peer_dev_id << ", count=" << count;

  if (!nccl_ipc_proxy_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL IPC proxy not set");
  }

  // 将整个Recv逻辑转发到executor端处理
  int executor_id = GetExecutorIdForDevice(local_dev_id);

  std::vector<uint8_t> data;
  Status status = nccl_ipc_proxy_->Recv(group_key, peer_dev_id, count, dtype, executor_id, data);

  if (status.OK() && !data.empty()) {
    // 将接收到的数据复制到buf指向的主机内存
    std::memcpy(buf, data.data(), std::min(count, data.size()));
  }

  return status;
}

Status NcclCommunicator::SendGroup(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream,
                                   const std::vector<int>& block_ids, const std::vector<int64_t>& offsets,
                                   const std::vector<size_t>& counts, DataType dtype) {
  // TODO(zakwang): 暂未查清楚卡住原因，保持双方卡号一致
  local_dev_id = peer_dev_id;
  KLLM_LOG_DEBUG << "[NCCL] SendGroup starting for group_key=" << group_key << ", local_dev_id=" << local_dev_id
                 << ", peer_dev_id=" << peer_dev_id << ", buffer_count=" << block_ids.size();

  if (block_ids.empty()) {
    return Status();  // 无数据需要发送，直接返回成功
  }

  if (block_ids.size() != offsets.size() || block_ids.size() != counts.size()) {
    KLLM_LOG_ERROR << "[NCCL] SendGroup parameter mismatch: block_ids size " << block_ids.size() << ", offsets size "
                   << offsets.size() << ", counts size " << counts.size();
    return Status(RetCode::RET_INVALID_ARGUMENT, "block_ids, offsets and counts vectors must have the same size");
  }

  if (!nccl_ipc_proxy_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL IPC proxy not set");
  }

  // 将整个SendGroup逻辑转发到executor端处理
  int executor_id = GetExecutorIdForDevice(local_dev_id);

  return nccl_ipc_proxy_->SendGroup(group_key, peer_dev_id, block_ids, offsets, counts, dtype, executor_id);
}

Status NcclCommunicator::RecvGroup(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream,
                                   const std::vector<int>& block_ids, const std::vector<int64_t>& offsets,
                                   const std::vector<size_t>& counts, DataType dtype) {
  // TODO(zakwang): 暂未查清楚卡住原因，保持双方卡号一致
  peer_dev_id = local_dev_id;
  KLLM_LOG_DEBUG << "[NCCL] RecvGroup starting for group_key=" << group_key << ", local_dev_id=" << local_dev_id
                 << ", peer_dev_id=" << peer_dev_id << ", buffer_count=" << block_ids.size();

  if (block_ids.size() != offsets.size() || block_ids.size() != counts.size()) {
    KLLM_LOG_ERROR << "[NCCL] RecvGroup parameter mismatch: block_ids size " << block_ids.size() << ", offsets size "
                   << offsets.size() << ", counts size " << counts.size();
    return Status(RetCode::RET_INVALID_ARGUMENT, "block_ids, offsets and counts vectors must have the same size");
  }

  if (!nccl_ipc_proxy_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL IPC proxy not set");
  }

  // 将整个RecvGroup逻辑转发到executor端处理
  int executor_id = GetExecutorIdForDevice(local_dev_id);

  return nccl_ipc_proxy_->RecvGroup(group_key, peer_dev_id, block_ids, offsets, counts, dtype, executor_id);
}

// 获取通信组指针（只读，不转移所有权）
NcclCommGroup* NcclCommunicator::GetCommunicatorGroup(const std::string& group_key) {
  std::lock_guard<std::mutex> lock(comm_group_mutex_);
  auto it = comm_groups_.find(group_key);
  return it != comm_groups_.end() ? it->second.get() : nullptr;
}
#endif  // ENABLE_CUDA
}  // namespace ksana_llm