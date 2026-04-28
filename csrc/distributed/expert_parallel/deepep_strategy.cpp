/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "distributed/expert_parallel/deepep_strategy.h"
#include "distributed/expert_parallel/deepep_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {

DeepEPStrategy::DeepEPStrategy(int rank, int num_ranks, int world_size, int node_rank, int hidden_size, int num_experts,
                               int max_token_num, bool low_latency_mode) {
  // Only create DeepEPWrapper, don't call Initialize
  deepep_wrapper_ = std::make_shared<DeepEPWrapper>(rank, num_ranks, world_size, node_rank, hidden_size, num_experts,
                                                    low_latency_mode, max_token_num);
}

EPCommunicationConfig DeepEPStrategy::GetCommunicationConfig() const {
  return deepep_wrapper_->GetCommunicationConfig();
}

void DeepEPStrategy::Initialize(const std::vector<EPCommunicationConfig>& comm_configs) {
  if (!deepep_wrapper_) {
    KLLM_THROW("deepep_wrapper_ is null, cannot initialize");
  }
  deepep_wrapper_->Initialize(comm_configs);
}

Status DeepEPStrategy::Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                                Stream stream) {
  deepep_wrapper_->Dispatch(input_tensors, output_tensors, stream);
  return Status();
}

Status DeepEPStrategy::Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                               Stream stream) {
  deepep_wrapper_->Combine(input_tensors, output_tensors, stream);
  return Status();
}

int DeepEPStrategy::GetRecvNumTokens() const { return deepep_wrapper_->GetRecvNumTokens(); }

}  // namespace ksana_llm