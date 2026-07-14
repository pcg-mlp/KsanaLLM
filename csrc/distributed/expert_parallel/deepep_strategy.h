/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>
#include "distributed/control_message.h"
#include "distributed/expert_parallel/expert_parallel_strategy.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

class DeepEPWrapper;

class DeepEPStrategy : public ExpertParallelStrategy {
 public:
  DeepEPStrategy(int rank, int num_ranks, int world_size, int node_rank, int hidden_size, int num_experts);
  ~DeepEPStrategy() override = default;

  // Get communicator config from this executor
  EPCommunicationConfig GetCommunicationConfig() const override;

  // Initialize with aggregated communicator config from all executors
  void Initialize(const std::vector<EPCommunicationConfig>& comm_configs) override;

  Status Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                  Stream stream) override;

  Status Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors, Stream stream) override;

  int GetRecvNumTokens() const override;

 private:
  std::shared_ptr<DeepEPWrapper> deepep_wrapper_;
};

}  // namespace ksana_llm