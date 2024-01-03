/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/nccl_all_reduce_sum_layer.h"

namespace numerous_llm {

Status NcclAllReduceSumLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  NCCL_CHECK(ncclGroupStart());
  NCCL_CHECK(ncclAllReduce(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                           output_tensors[0].GetPtr<void>(), input_tensors[0].GetElementNumber() * sizeof(half),
                           ncclHalf, ncclSum, context_->GetNCCLParam()[rank_].nccl_comm,
                           context_->GetNCCLStreams()[rank_]));
  NCCL_CHECK(ncclGroupEnd());
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}
}  // namespace numerous_llm
