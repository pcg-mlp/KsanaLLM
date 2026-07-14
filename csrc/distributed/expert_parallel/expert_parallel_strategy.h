/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>
#include "distributed/control_message.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

/**
 * @brief 专家并行策略基类
 *
 * 定义了专家并行通信的标准接口，具体实现可以是 DeepEP、NCCL 等不同方案
 */
class ExpertParallelStrategy {
 public:
  virtual ~ExpertParallelStrategy() = default;

  /**
   * @brief 获取当前 executor 的通信配置
   * @return EPCommunicationConfig 通信配置
   */
  virtual EPCommunicationConfig GetCommunicationConfig() const = 0;

  /**
   * @brief 使用聚合的通信配置初始化
   * @param comm_configs 各 executor 的通信配置
   */
  virtual void Initialize(const std::vector<EPCommunicationConfig>& comm_configs) = 0;

  /**
   * @brief 执行 Dispatch 操作（分发 tokens 到各专家）
   * @param input_tensors 输入张量（hidden_states, topk_ids, topk_weights 等）
   * @param output_tensors 输出张量（分发后的数据）
   * @param stream Stream用于执行操作
   * @return Status 执行状态
   */
  virtual Status Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                          Stream stream) = 0;

  /**
   * @brief 执行 Combine 操作（聚合各专家的输出）
   * @param input_tensors 输入张量（各专家的输出）
   * @param output_tensors 输出张量（聚合后的结果）
   * @param stream Stream用于执行操作
   * @return Status 执行状态
   */
  virtual Status Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         Stream stream) = 0;

  /**
   * @brief 获取接收到的 token 数量（Dispatch 后）
   * @return int 接收到的 token 数量
   */
  virtual int GetRecvNumTokens() const = 0;
};

/**
 * @brief 专家并行策略工厂
 *
 * 根据配置创建具体的专家并行策略实例
 */
class ExpertParallelStrategyFactory {
 public:
  enum class StrategyType {
    DEEPEP,  // DeepEP 实现
    // 暂无其他实现
  };

  /**
   * @brief 创建专家并行策略实例（不包含 init_data，需要后续调用 Initialize）
   * @param type 策略类型
   * @param rank 当前进程在本节点内的 rank
   * @param num_ranks 本节点的进程总数
   * @param world_size 所有节点的进程总数
   * @param node_rank 当前节点的编号
   * @param hidden_size 隐藏层大小
   * @param num_experts 专家总数
   * @return std::shared_ptr<ExpertParallelStrategy> 策略实例
   */
  static std::shared_ptr<ExpertParallelStrategy> Create(StrategyType type, int rank, int num_ranks, int world_size,
                                                        int node_rank, int hidden_size, int num_experts);
};

}  // namespace ksana_llm