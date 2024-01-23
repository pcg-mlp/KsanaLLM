/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/nccl_all_reduce_sum_layer.h"

namespace numerous_llm {

Status NcclAllReduceSumLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (context_->GetTensorParallelSize() > 1) {
    NCCL_CHECK(ncclGroupStart());
    ncclResult_t ncclError =
        ncclAllReduce(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()), output_tensors[0].GetPtr<void>(),
                      input_tensors[0].GetElementNumber(), ncclHalf, ncclSum, context_->GetNCCLParam()[rank_].nccl_comm,
                      context_->GetComputeStreams()[rank_]);

    if (ncclError != ncclSuccess) {
      NLLM_LOG_INFO << fmt::format("NCCL error: {}\n", ncclGetErrorString(ncclError));
    }
    NCCL_CHECK(ncclGroupEnd());
  } else {
    void* src = input_tensors[0].GetPtr<void>();
    void* dst = output_tensors[0].GetPtr<void>();
    CUDA_CHECK(cudaMemcpyAsync(dst, src, input_tensors[0].GetTotalBytes(), cudaMemcpyDeviceToDevice,
                               context_->GetComputeStreams()[rank_]));
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}
}  // namespace numerous_llm
