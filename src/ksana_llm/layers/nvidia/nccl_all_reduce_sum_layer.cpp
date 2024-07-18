/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/nccl_all_reduce_sum_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status NcclAllReduceSumLayer<T>::Forward(const std::vector<Tensor>& input_tensors,
                                         std::vector<Tensor>& output_tensors) {
  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl stream just enable when IsRunContextDecodeAndDecodeSerially == false
  cudaStream_t* stream;
  if (context_->IsRunContextDecodeAndDecodeSerially()) {
    stream = &(context_->GetComputeStreams()[rank_].Get());
  } else {
    stream = &(context_->GetNCCLStreams()[rank_].Get());
  }

  if (context_->GetTensorParallelSize() > 1) {
    NCCL_CHECK(ncclGroupStart());
    ncclResult_t ncclError =
        ncclAllReduce(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()), output_tensors[0].GetPtr<void>(),
                      input_tensors[0].GetElementNumber(), GetNcclDataType<T>(), ncclSum,
                      context_->ext->GetNCCLParam()[rank_].nccl_comm, *stream);
    if (ncclError != ncclSuccess) {
      KLLM_LOG_DEBUG << fmt::format("NCCL error: {}\n", ncclGetErrorString(ncclError));
    }
    NCCL_CHECK(ncclGroupEnd());
  } else {
    void* src = input_tensors[0].GetPtr<void>();
    void* dst = output_tensors[0].GetPtr<void>();
    CUDA_CHECK(cudaMemcpyAsync(dst, src, input_tensors[0].GetTotalBytes(), cudaMemcpyDeviceToDevice, *stream));
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template class NcclAllReduceSumLayer<float>;
template class NcclAllReduceSumLayer<half>;
#ifdef ENABLE_BFLOAT16
template class NcclAllReduceSumLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
