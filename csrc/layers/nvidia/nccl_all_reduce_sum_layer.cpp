/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/nccl_all_reduce_sum_layer.h"

#include "device/nvidia/nccl_utils.h"
#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status NcclAllReduceSumLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                   std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);

  // When using cudaMalloc and reduce operations with P2P enabled, the system may hang. This issue may be a bug in NCCL
  // or CUDA. Resolving it requires switching PyTorch's memory allocator to asynchronous mode. Alternatively, adding a
  // synchronization operation can prevent concurrent execution of malloc and reduce to avoid the hang.
  const char* const torch_alloc_config = std::getenv("PYTORCH_CUDA_ALLOC_CONF");
  const std::string torch_alloc_config_str = torch_alloc_config == nullptr ? "" : std::string(torch_alloc_config);
  need_sync_ = torch_alloc_config_str.find("backend:cudaMallocAsync") == std::string::npos;
  return Status();
}

Status NcclAllReduceSumLayer::Forward(const Tensor* input, Tensor* output, const LayerExecContext& exec_ctx) {
  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl stream just enable when IsRunContextDecodeAndDecodeSerially == false
  cudaStream_t* stream;
  if (context_->global->IsRunContextDecodeAndDecodeSerially()) {
    stream = &(context_->device->GetPrimaryComputeStream().Get());
  } else {
    stream = &(context_->device->GetCommStream().Get());
  }

  if (context_->global->GetTensorParallelSize() > 1) {
    // To avoid getting stuck during ncclAllReduce.
    if (need_sync_) {
      CUDA_CHECK(cudaStreamSynchronize(*stream));
    }
    ncclResult_t ncclError = ncclAllReduce(reinterpret_cast<const void*>(input->GetPtr<void>()), output->GetPtr<void>(),
                                           input->GetElementNumber(), GetNcclDataType(inter_data_type_), ncclSum,
                                           context_->device->ext->GetNCCLParam().nccl_comm, *stream);
    if (ncclError != ncclSuccess) {
      KLLM_LOG_ERROR << fmt::format("NCCL error: {}\n", ncclGetErrorString(ncclError));
      return Status(RetCode::RET_INFER_FAILED, "NCCL error");
    }
  } else {
    void* src = input->GetPtr<void>();
    void* dst = output->GetPtr<void>();
    CUDA_CHECK(cudaMemcpyAsync(dst, src, input->GetTotalBytes(), cudaMemcpyDeviceToDevice, *stream));
  }
  output->shape = input->shape;
  output->dtype = input->dtype;
  return Status();
}

}  // namespace ksana_llm
