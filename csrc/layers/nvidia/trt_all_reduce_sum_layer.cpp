/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/trt_all_reduce_sum_layer.h"

#include "kernels/nvidia/basic_kernel_wrapper.h"

namespace ksana_llm {

Status TrtAllReduceSumLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                  std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;

  // No need to initialize custom reduce sum when `tp == 1`.
  if (context_->global->GetTensorParallelSize() == 1) {
    return Status();
  }

  int parameter_index = 0;
  const size_t max_buffer_size = std::any_cast<const size_t>(parameters[parameter_index++]);

  // For DeepSeek-V3, roughly 28MB of extra memory is allocated
  AllocTrtAllReduceWorkspace(rank_, max_buffer_size, context_->device->ext->GetTrtAllReduceBuffers(),
                             context_->device->ext->GetTrtAllReduceFlags(),
                             context_->device->ext->GetTrtAllReduceWorkspaces(),
                             context_->device->GetPrimaryComputeStream().Get());

  // When using cudaMalloc and reduce operations with P2P enabled, the system may hang. This issue may be a bug in NCCL
  // or CUDA. Resolving it requires switching PyTorch's memory allocator to asynchronous mode. Alternatively, adding a
  // synchronization operation can prevent concurrent execution of malloc and reduce to avoid the hang.
  const char* const torch_alloc_config = std::getenv("PYTORCH_CUDA_ALLOC_CONF");
  const std::string torch_alloc_config_str = torch_alloc_config == nullptr ? "" : std::string(torch_alloc_config);
  need_sync_ = torch_alloc_config_str.find("backend:cudaMallocAsync") == std::string::npos;
  return Status();
}

Status TrtAllReduceSumLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

void TrtAllReduceSumLayer::Clear() {
  if (is_init_) {
    FreeTrtAllReduceWorkspace(rank_, context_->device->ext->GetTrtAllReduceBuffers(),
                              context_->device->ext->GetTrtAllReduceFlags(),
                              context_->device->ext->GetTrtAllReduceWorkspaces());
  }
}

template <typename T>
Status TrtAllReduceSumLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  cudaStream_t* stream;
  if (context_->global->IsRunContextDecodeAndDecodeSerially()) {
    stream = &(context_->device->GetPrimaryComputeStream().Get());
  } else {
    stream = &(context_->device->GetCommStream().Get());
  }
  if (context_->global->GetTensorParallelSize() > 1) {
    void* input = input_tensors[0].GetPtr<void>();
    void* result = output_tensors[0].GetPtr<void>();
    // Initialize the workspace on the first forward
    if (!is_init_) {
      InitTrtAllReduceWorkspace(rank_, context_->device->ext->GetTrtAllReduceBuffers(),
                                context_->device->ext->GetTrtAllReduceFlags(),
                                context_->device->ext->GetTrtAllReduceWorkspaces(), *stream);
      is_init_ = true;
    }
    const size_t token_num = input_tensors[0].shape[0];
    const size_t hidden_dim = input_tensors[0].shape[1];
    RunTrtAllReduce<T>(input, rank_, token_num, hidden_dim, context_->device->ext->GetTrtAllReduceWorkspaces(), result,
                       *stream);
    // To avoid getting stuck during TrtAllReduce.
    if (need_sync_) {
      CUDA_CHECK(cudaStreamSynchronize(*stream));
    }
  } else {
    void* src = input_tensors[0].GetPtr<void>();
    void* dst = output_tensors[0].GetPtr<void>();
    CUDA_CHECK(cudaMemcpyAsync(dst, src, input_tensors[0].GetTotalBytes(), cudaMemcpyDeviceToDevice, *stream));
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm
