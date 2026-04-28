/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>
#include "device/common_device.h"
#include "distributed/control_message.h"
#include "expert_parallel_strategy.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

/**
 * @brief 专家并行管理器
 *
 * 管理专家并行的策略实例，提供统一的接口给上层使用
 */
class ExpertParallelManager {
 public:
  ExpertParallelManager() = default;
  ~ExpertParallelManager();

  /**
   * @brief 创建专家并行管理器（不包含 init_data）
   * @param rank 当前进程在本节点内的 rank
   * @param num_ranks 本节点的进程总数
   * @param world_size 所有节点的进程总数
   * @param node_rank 当前节点的编号
   * @param hidden_size 隐藏层大小
   * @param num_experts 专家总数
   * @param max_token_num 最大 token 数量
   * @param low_latency 是否启用低延迟模式
   * @return Status 创建状态
   */
  Status Create(int rank, int num_ranks, int world_size, int node_rank, int hidden_size, int num_experts,
                int max_token_num, bool low_latency);

  /**
   * @brief 获取当前 executor 的通信配置
   * @return EPCommunicationConfig 通信配置
   */
  EPCommunicationConfig GetCommunicationConfig();

  /**
   * @brief 使用聚合的通信配置初始化专家并行
   * @param comm_configs 各 executor 的通信配置（nvshmem_id, ipc_handles 等）
   * @return Status 初始化状态
   */
  Status Initialize(const std::vector<EPCommunicationConfig>& comm_configs);

  /**
   * @brief 执行 Dispatch 操作（分发 tokens 到各专家）
   * @param input_tensors 输入张量（hidden_states, topk_ids, topk_weights 等）
   * @param output_tensors 输出张量（分发后的数据）
   * @param stream Stream用于执行操作
   * @return Status 执行状态
   */
  Status Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors, Stream stream);

  /**
   * @brief 执行 Combine 操作（聚合各专家的输出）
   * @param input_tensors 输入张量（各专家的输出）
   * @param output_tensors 输出张量（聚合后的结果）
   * @param stream Stream用于执行操作
   * @return Status 执行状态
   */
  Status Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors, Stream stream);

  /**
   * @brief 获取接收到的 token 数量（Dispatch 后）
   * @return int 接收到的 token 数量
   */
  int GetRecvNumTokens() const;

 private:
  // 专家并行策略实例
  std::shared_ptr<ExpertParallelStrategy> strategy_ = nullptr;
};

}  // namespace ksana_llm