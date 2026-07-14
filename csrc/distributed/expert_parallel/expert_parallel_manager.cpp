/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "expert_parallel_manager.h"
#include "utils/logger.h"

namespace ksana_llm {

ExpertParallelManager::~ExpertParallelManager() = default;

Status ExpertParallelManager::Create(int rank, int num_ranks, int world_size, int node_rank, int hidden_size,
                                     int num_experts) {
  KLLM_LOG_DEBUG << fmt::format(
      "ExpertParallelManager::Create(rank={}, num_ranks={}, world_size={}, node_rank={}, hidden_size={}, "
      "num_experts={})",
      rank, num_ranks, world_size, node_rank, hidden_size, num_experts);

  // Actually only DeepEP strategy is supported
  strategy_ = ExpertParallelStrategyFactory::Create(ExpertParallelStrategyFactory::StrategyType::DEEPEP, rank,
                                                    num_ranks, world_size, node_rank, hidden_size, num_experts);
  KLLM_LOG_DEBUG << "ExpertParallelManager strategy created successfully";
  return Status();
}

EPCommunicationConfig ExpertParallelManager::GetCommunicationConfig() {
  if (!strategy_) {
    KLLM_THROW("ExpertParallelManager strategy is not created");
  }
  return strategy_->GetCommunicationConfig();
}

Status ExpertParallelManager::Initialize(const std::vector<EPCommunicationConfig>& comm_configs) {
  if (!strategy_) {
    KLLM_THROW("ExpertParallelManager strategy is not created");
  }
  strategy_->Initialize(comm_configs);
  KLLM_LOG_INFO << "ExpertParallelManager initialized successfully";
  return Status();
}

Status ExpertParallelManager::Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                                       Stream stream) {
  return strategy_->Dispatch(input_tensors, output_tensors, stream);
}

Status ExpertParallelManager::Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                                      Stream stream) {
  return strategy_->Combine(input_tensors, output_tensors, stream);
}

int ExpertParallelManager::GetRecvNumTokens() const { return strategy_->GetRecvNumTokens(); }

}  // namespace ksana_llm