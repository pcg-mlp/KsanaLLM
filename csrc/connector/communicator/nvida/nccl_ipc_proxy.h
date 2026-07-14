/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#  include <cuda_runtime.h>
#  include <nccl.h>
#endif

#include <memory>
#include <string>
#include <vector>

#include "connector/communicator/communicator.h"
#include "ipc/ipc_func_wrapper.h"
#include "utils/status.h"

// TODO(zakkwang): 将communicator相关功能移到executor中实现
namespace ksana_llm {

/**
 * @brief Engine端的NCCL IPC代理类
 *
 * 该类在engine进程中运行，通过IPC将NCCL操作转发到executor进程执行
 * 替代原有NcclCommunicator中直接操作GPU的部分
 */
class NcclIpcProxy {
 public:
  explicit NcclIpcProxy(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper);
  virtual ~NcclIpcProxy();

  /**
   * @brief 初始化NCCL通信组（通过IPC）
   *
   * @param group_key 通信组标识
   * @param comm_id NCCL唯一ID
   * @param rank 当前rank
   * @param world_size 总的rank数量
   * @param executor_id 目标executor ID
   * @return Status
   */
  virtual Status InitializeCommGroup(const std::string& group_key, const ncclUniqueId& comm_id, int rank,
                                     int world_size, int executor_id);

  /**
   * @brief 发送数据（通过IPC）- 传输数据内容而非GPU指针
   *
   * @param group_key 通信组标识
   * @param peer_rank 目标rank
   * @param data 要发送的数据内容
   * @param dtype 数据类型
   * @param executor_id 目标executor ID
   * @return Status
   */
  virtual Status Send(const std::string& group_key, int peer_rank, const std::vector<uint8_t>& data, DataType dtype,
                      int executor_id);

  /**
   * @brief 接收数据（通过IPC）- 返回数据内容而非写入GPU指针
   *
   * @param group_key 通信组标识
   * @param peer_rank 源rank
   * @param count 要接收的字节数
   * @param dtype 数据类型
   * @param executor_id 目标executor ID
   * @param data 输出：接收到的数据内容
   * @return Status
   */
  virtual Status Recv(const std::string& group_key, int peer_rank, size_t count, DataType dtype, int executor_id,
                      std::vector<uint8_t>& data);

  /**
   * @brief 批量发送数据（通过IPC，使用block_id和offset）
   *
   * @param group_key 通信组标识
   * @param peer_rank 目标rank
   * @param block_ids block ID列表
   * @param offsets block内偏移量列表（字节）
   * @param counts 元素数量列表
   * @param dtype 数据类型
   * @param executor_id 目标executor ID
   * @return Status
   */
  virtual Status SendGroup(const std::string& group_key, int peer_rank, const std::vector<int>& block_ids,
                           const std::vector<int64_t>& offsets, const std::vector<size_t>& counts, DataType dtype,
                           int executor_id);

  /**
   * @brief 批量接收数据（通过IPC，使用block_id和offset）
   *
   * @param group_key 通信组标识
   * @param peer_rank 源rank
   * @param block_ids block ID列表
   * @param offsets block内偏移量列表（字节）
   * @param counts 元素数量列表
   * @param dtype 数据类型
   * @param executor_id 目标executor ID
   * @return Status
   */
  virtual Status RecvGroup(const std::string& group_key, int peer_rank, const std::vector<int>& block_ids,
                           const std::vector<int64_t>& offsets, const std::vector<size_t>& counts, DataType dtype,
                           int executor_id);

  /**
   * @brief 获取通信组信息（通过IPC查询）
   *
   * @param group_key 通信组标识
   * @param executor_id 目标executor ID
   * @param rank 输出：当前rank
   * @param world_size 输出：总rank数
   * @return Status
   */
  virtual Status GetCommGroupInfo(const std::string& group_key, int executor_id, int& rank, int& world_size);

 private:
  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_;
};

}  // namespace ksana_llm