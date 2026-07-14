/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#  include <cuda_runtime.h>
#  include <nccl.h>
#  include <memory>
#  include <mutex>
#  include <string>
#  include <unordered_map>
#  include <vector>

#  include "configure/schedule_config_parser.h"
#  include "connector/communicator/nvida/nccl_communicator.h"
#  include "utils/blocking_queue.h"
#  include "utils/logger.h"
#  include "utils/status.h"

namespace ksana_llm {

class BlockAllocatorManager;
class BlockAllocatorInterface;

/**
 * @brief Executor端的NCCL操作处理器
 *
 * 该类在executor进程中运行，负责实际的NCCL通信操作
 * 每个executor只管理一张GPU卡
 */
class NcclExecutorHandler {
 public:
  NcclExecutorHandler(int device_id, ConnectorConfig connector_config);
  ~NcclExecutorHandler();

  /**
   * @brief 设置BlockAllocatorManager用于解析block_id到GPU指针
   *
   * @param block_allocator_manager BlockAllocatorManager实例
   * @param rank 当前executor的rank
   * @param runtime_config 运行时配置
   */
  void SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager, int rank,
                                const RuntimeConfig& runtime_config);

  /**
   * @brief 初始化NCCL通信组
   *
   * @param group_key 通信组标识
   * @param comm_id NCCL唯一ID
   * @param rank 当前rank
   * @param world_size 总的rank数量
   * @return Status
   */
  Status InitializeCommGroup(const std::string& group_key, const ncclUniqueId& comm_id, int rank, int world_size);

  /**
   * @brief 发送数据（从host内存）
   *
   * @param group_key 通信组标识
   * @param peer_rank 目标rank
   * @param host_buf host内存数据缓冲区
   * @param count 字节数
   * @param dtype 数据类型
   * @return Status
   */
  Status Send(const std::string& group_key, int peer_rank, const void* host_buf, size_t count, DataType dtype);

  /**
   * @brief 接收数据（到host内存）
   *
   * @param group_key 通信组标识
   * @param peer_rank 源rank
   * @param host_buf host内存数据缓冲区
   * @param count 字节数
   * @param dtype 数据类型
   * @return Status
   */
  Status Recv(const std::string& group_key, int peer_rank, void* host_buf, size_t count, DataType dtype);

  /**
   * @brief 批量发送数据（使用block_id和offset）
   *
   * @param group_key 通信组标识
   * @param peer_rank 目标rank
   * @param block_ids block ID列表
   * @param offsets block内偏移量列表（字节）
   * @param counts 元素数量列表
   * @param dtype 数据类型
   * @return Status
   */
  Status SendGroup(const std::string& group_key, int peer_rank, const std::vector<int>& block_ids,
                   const std::vector<int64_t>& offsets, const std::vector<size_t>& counts, DataType dtype);

  /**
   * @brief 批量接收数据（使用block_id和offset）
   *
   * @param group_key 通信组标识
   * @param peer_rank 源rank
   * @param block_ids block ID列表
   * @param offsets block内偏移量列表（字节）
   * @param counts 元素数量列表
   * @param dtype 数据类型
   * @return Status
   */
  Status RecvGroup(const std::string& group_key, int peer_rank, const std::vector<int>& block_ids,
                   const std::vector<int64_t>& offsets, const std::vector<size_t>& counts, DataType dtype);

  /**
   * @brief 获取通信组信息（用于查询）
   *
   * @param group_key 通信组标识
   * @param rank 输出：当前rank
   * @param world_size 输出：总rank数
   * @return Status
   */
  Status GetCommGroupInfo(const std::string& group_key, int& rank, int& world_size);

 private:
  struct CommGroupInfo {
    ncclComm_t comm = nullptr;
    cudaStream_t send_stream = nullptr;
    cudaStream_t recv_stream = nullptr;
    int rank = -1;
    int world_size = -1;
  };

  /**
   * @brief 根据block_id和offset解析GPU指针
   *
   * @param block_ids block ID列表
   * @param offsets 偏移量列表
   * @param ptrs 输出：GPU指针列表
   * @return Status
   */
  Status ResolveBlockPtrs(const std::vector<int>& block_ids, const std::vector<int64_t>& offsets,
                          std::vector<void*>& ptrs);

  /**
   * @brief 延迟初始化device_allocator_（第一次使用时）
   * @return Status
   */
  Status EnsureDeviceAllocatorInitialized();

  int device_id_;
  ConnectorConfig connector_config_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<CommGroupInfo>> comm_groups_;

  // Lazy initialization members
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager_;
  int rank_ = -1;
  int attn_tensor_parallel_size_ = -1;
  std::shared_ptr<BlockAllocatorInterface> device_allocator_;

  std::unique_ptr<BlockingQueue<void*>> gpu_buffer_pool_;
};

}  // namespace ksana_llm

#endif  // ENABLE_CUDA