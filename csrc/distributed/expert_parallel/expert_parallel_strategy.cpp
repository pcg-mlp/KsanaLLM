/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "expert_parallel_strategy.h"
#include "deepep_strategy.h"
#include "utils/logger.h"

namespace ksana_llm {

std::shared_ptr<ExpertParallelStrategy> ExpertParallelStrategyFactory::Create(StrategyType type, int rank,
                                                                              int num_ranks, int world_size,
                                                                              int node_rank, int hidden_size,
                                                                              int num_experts, int max_token_num,
                                                                              bool low_latency) {
  if (type == StrategyType::DEEPEP) {
    return std::make_shared<DeepEPStrategy>(rank, num_ranks, world_size, node_rank, hidden_size, num_experts,
                                            max_token_num, low_latency);
  } else {
    KLLM_THROW(fmt::format("Unsupported expert parallel strategy type: {}", static_cast<int>(type)));
  }
}

}  // namespace ksana_llm