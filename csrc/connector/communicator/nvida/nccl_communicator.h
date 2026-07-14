/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#ifdef ENABLE_CUDA
#  include <nccl.h>
#endif
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include "connector/communicator/communicator.h"
#include "runtime/threadpool.h"
#include "utils/logger.h"
#ifdef ENABLE_CUDA
#  include "device/nvidia/cuda_utils.h"
#  include "device/nvidia/nccl_utils.h"
#  include "connector/communicator/nvida/nccl_ipc_proxy.h"
#endif
#include "utils/ret_code.h"
#include "utils/status.h"

// TODO(zakkwang): 将communicator相关功能移到executor中实现
namespace ksana_llm {
#ifdef ENABLE_CUDA

// Simplified communication group structure for engine side
// Actual NCCL resources are managed on executor side
struct NcclCommGroup {
  /** @brief NCCL unique ID for establishing communication */
  ncclUniqueId comm_id;

  /** @brief Communication ID string (base64 encoded) */
  std::string comm_id_str;

  /** @brief Address tuples for this group */
  std::vector<std::tuple<int, int, std::string>> address_tuples;

  /** @brief Number of devices in this communication group */
  int device_count_ = 0;

  /** @brief World size for this communication group */
  int world_size_ = 0;

  /** @brief Rank information for each device */
  std::vector<int> device_ranks_;

  NcclCommGroup() = default;
  ~NcclCommGroup() = default;

  // 禁止拷贝
  NcclCommGroup(const NcclCommGroup&) = delete;
  NcclCommGroup& operator=(const NcclCommGroup&) = delete;

  bool IsActive() const {
    // Check if comm_id is initialized
    return device_count_ > 0 && world_size_ > 0 && !device_ranks_.empty();
  }
};

class NcclCommunicator : public Communicator {
 public:
  explicit NcclCommunicator(const ksana_llm::ConnectorConfig& config) : Communicator(config) {}
  ~NcclCommunicator() override;

  void Shutdown() override;

  Status Initialize() override;

  Status ProcessHeartbeatData(const std::unordered_map<std::string, std::string>& comm_group_to_id,
                              const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>&
                                  comm_group_to_address) override;

  // Initialize NCCL communicators on executors via IPC
  Status InitializeNcclCommsOnExecutors(const std::string& group_key, const ncclUniqueId& comm_id);

  // Communicator 接口实现
  Status Send(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id, const void* buf,
              size_t count, DataType dtype) override;
  Status Recv(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id, void* buf, size_t count,
              DataType dtype) override;
  Status SendGroup(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id,
                   const std::vector<int>& block_ids, const std::vector<int64_t>& offsets,
                   const std::vector<size_t>& counts, DataType dtype) override;
  Status RecvGroup(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id,
                   const std::vector<int>& block_ids, const std::vector<int64_t>& offsets,
                   const std::vector<size_t>& counts, DataType dtype) override;

  // Communicator 接口实现
  Status Send(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream, const void* buf,
              size_t count, DataType dtype);
  Status Recv(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream, void* buf,
              size_t count, DataType dtype);
  Status SendGroup(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream,
                   const std::vector<int>& block_ids, const std::vector<int64_t>& offsets,
                   const std::vector<size_t>& counts, DataType dtype);
  Status RecvGroup(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream,
                   const std::vector<int>& block_ids, const std::vector<int64_t>& offsets,
                   const std::vector<size_t>& counts, DataType dtype);

  struct NcclRankInfo {
    int cur_rank;
    int peer_rank;
    int send_rank;
    int recv_rank;
  };
  NcclRankInfo CalcNcclRanks(GroupRole group_role, int node_rank, int world_size, int device_count, int dev_id);

  // 在 NcclCommunicator 增加测试专用 comm_group 插入方法
  void SetCommGroupForTest(const std::string& group_key, std::unique_ptr<NcclCommGroup> group) {
    std::lock_guard<std::mutex> lock(comm_group_mutex_);
    comm_groups_[group_key] = std::move(group);
  }

  void SetSendCommIdCallback(std::function<Status(const std::string&, const std::string&)> callback) {
    send_comm_id_callback_ = std::move(callback);
  }

  // Set IPC proxy for communicating with executors
  void SetNcclIpcProxy(std::shared_ptr<NcclIpcProxy> nccl_ipc_proxy) { nccl_ipc_proxy_ = nccl_ipc_proxy; }

  /**
   * @brief Protected implementation of SetReceiveCallback (implements Communicator interface)
   *
   * @param callback Function to call when data is received
   */
  void DoSetReceiveCallback(const ReceiveCallback& callback) override;

  bool CheckCommGroupAvailable(const std::string& group_key);

  void CreateNcclCommGroup(const std::string& group_key, const std::string& encoded_value);
  bool IsConnectionReady(const std::string& group_key, int dev_id) const;
  NcclCommGroup* GetCommunicatorGroup(const std::string& group_key);

  // Get executor ID for a given device ID
  int GetExecutorIdForDevice(int device_id) const { return device_id; }

 private:
  std::mutex comm_group_mutex_;
  std::unordered_map<std::string, std::unique_ptr<NcclCommGroup>> comm_groups_;

  // 回调函数，用于发送通信ID到协调器
  std::function<Status(const std::string&, const std::string&)> send_comm_id_callback_;

  /** @brief Callback for received messages (modern C++ style) */
  ReceiveCallback receive_callback_;
  void* recv_user_data_ = nullptr;

  // IPC proxy for communicating with executors
  std::shared_ptr<NcclIpcProxy> nccl_ipc_proxy_ = nullptr;

  // ThreadPool for parallel NCCL initialization
  std::unique_ptr<ThreadPool> init_thread_pool_ = nullptr;
};

#endif  // ENABLE_CUDA
}  // namespace ksana_llm
