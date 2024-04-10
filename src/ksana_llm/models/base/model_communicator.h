/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/layers/custom_all_reduce_sum_layer.h"
#include "ksana_llm/layers/nccl_all_gather_layer.h"
#include "ksana_llm/layers/nccl_all_reduce_sum_layer.h"

namespace ksana_llm {

// The collective communicator library.
class ModelCommunicator {
 public:
  ModelCommunicator(Tensor* buffer, Tensor* input, int rank, std::shared_ptr<Context> context);
  ~ModelCommunicator();

  // The all-gather reduce.
  Status AllGather(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // The reduce-sum reduce.
  Status ReduceSum(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                   bool is_context_stage, bool use_custom);

 private:
  // Whether use the custom reduce layer.
  bool enable_custom_all_reduce_ = true;

  // The default all reduce layer.
  std::shared_ptr<NcclAllReduceSumLayer> nccl_all_reduce_sum_layer_;

  // The all gather layer
  std::shared_ptr<NcclAllGatherLayer> nccl_all_gather_layer_;

  // The custom all reduce layer.
  std::shared_ptr<CustomAllReduceSumLayer> custom_all_reduce_sum_layer_0_;

 private:
  int rank_;
  std::shared_ptr<Context> context_;

  // For custom all reduce layer.
  Tensor reduce_tensor_;
  Tensor rank_tensor_0_;

  // Use for custom all reduce layer.
  Tensor* buffer_;
  Tensor* input_;

  // Whether the communication is finished.
  Event nccl_finish_event_;
};

}  // namespace ksana_llm
