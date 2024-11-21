/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#ifdef ENABLE_CUDA
#  include "ksana_llm/layers/custom_all_reduce_sum_layer.h"
#  include "ksana_llm/layers/nccl_all_gather_layer.h"
#  include "ksana_llm/layers/nccl_all_reduce_sum_layer.h"
#elif defined(ENABLE_ACL)
#  include "ksana_llm/layers/hccl_all_gather_layer.h"
#  include "ksana_llm/layers/hccl_all_reduce_sum_layer.h"
#endif

namespace ksana_llm {

// The collective communicator library.
template <typename T>
class ModelCommunicator {
 public:
  ModelCommunicator(Tensor* buffer, Tensor* input, int rank, std::shared_ptr<Context> context);
  ~ModelCommunicator();

  // The all-gather reduce.
  Status AllGather(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // The reduce-sum reduce.
  Status ReduceSum(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors, bool is_context_stage,
                   bool use_custom);

 private:
  // Whether use the custom reduce layer.
  bool enable_custom_all_reduce_ = true;

#ifdef ENABLE_CUDA
  // The default all reduce layer.
  std::shared_ptr<NcclAllReduceSumLayer<T>> nccl_all_reduce_sum_layer_;

  // The all gather layer
  std::shared_ptr<NcclAllGatherLayer<T>> nccl_all_gather_layer_;

  // The custom all reduce layer.
  std::shared_ptr<CustomAllReduceSumLayer<T>> custom_all_reduce_sum_layer_0_;
#elif defined(ENABLE_ACL)
  // The default all reduce layer.
  std::shared_ptr<HcclAllReduceSumLayer<T>> hccl_all_reduce_sum_layer_;

  // The all gather layer
  std::shared_ptr<HcclAllGatherLayer<T>> hccl_all_gather_layer_;
#endif

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
  Event comm_finish_event_;

 private:
  bool CheckIfUseCustomReduceSum(size_t batch_size, bool use_custom) {
    return enable_custom_all_reduce_
          && use_custom
          && context_->GetSupportedCudaGraphCaptureSizes().find(batch_size)
              == context_->GetSupportedCudaGraphCaptureSizes().end();
  }
};

}  // namespace ksana_llm
