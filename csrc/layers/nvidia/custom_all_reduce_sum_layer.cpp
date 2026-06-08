/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/custom_all_reduce_sum_layer.h"

#include <unistd.h>

#include "csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status CustomAllReduceSumLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                     std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;

  // No need to initialize custom reduce sum when `tp == 1`.
  if (context_->global->GetTensorParallelSize() == 1) {
    return Status();
  }

  int parameter_index = 0;
  void* input = std::any_cast<void*>(parameters[parameter_index++]);
  void* signal = std::any_cast<void*>(parameters[parameter_index++]);
  size_t signal_sz = std::any_cast<size_t>(parameters[parameter_index++]);
  rank_data_ = std::any_cast<void*>(parameters[parameter_index++]);
  rank_data_sz_ = std::any_cast<size_t>(parameters[parameter_index++]);
  is_group_custom_all_reduce_ = std::any_cast<bool>(parameters[parameter_index++]);

  // NOTE(karlluo): For attention data parallelism, we do all reduce as group allreduce: just do allreduce with between
  // some gpus. The root rank is the first rank of the attention data parallel group. For example, if the rank is 0, 1,
  // 2, 3, and the attention data parallel size is 2, the root rank is 0. If the rank is 4, 5, 6, 7, and the attention
  // data parallel size is 2, the root rank is 4. The root rank is used to determine the group of ranks that will
  // perform the all-reduce operation. The root rank is the first rank of the attention data parallel group.
  uint32_t attn_dp_para_size = runtime_config.parallel_basic_config.attn_data_parallel_size;
  if (attn_dp_para_size > 1 && is_group_custom_all_reduce_) {
    uint32_t tp_para_size = context_->global->GetTensorParallelSize();
    uint32_t world_size = tp_para_size / attn_dp_para_size;
    uint32_t attn_dp_group_id = rank_ / world_size;
    world_size_ = world_size;
    if (world_size_ == 1) {
      // NOTE(karlluo): We do not need to do all reduce for attention data parallelism when the world size is 1.
      return Status();
    }
    root_rank_ = attn_dp_group_id * world_size;
  } else {
    world_size_ = context_->global->GetTensorParallelSize();
  }

  if (signal != nullptr) {
    CUDA_CHECK(cudaMemset(signal, 0x0, signal_sz));
  }
  CUDA_CHECK(cudaMemset(rank_data_, 0x0, rank_data_sz_));

  signals_ = context_->device->ext->GetCustomAllReduceSignals();
  input_handles_ = context_->device->ext->GetCustomAllReduceInputs();

  signals_[rank_] = signal;
  input_handles_[rank_] = input;

  // is full nvlink on each device
  is_full_nvlink_ = context_->device->ext->IsFullNvLink();

  // When using cudaMalloc and reduce operations with P2P enabled, the system may hang. This issue may be a bug in NCCL
  // or CUDA. Resolving it requires switching PyTorch's memory allocator to asynchronous mode. Alternatively, adding a
  // synchronization operation can prevent concurrent execution of malloc and reduce to avoid the hang.
  const char* const torch_alloc_config = std::getenv("PYTORCH_CUDA_ALLOC_CONF");
  const std::string torch_alloc_config_str = torch_alloc_config == nullptr ? "" : std::string(torch_alloc_config);
  need_sync_ = torch_alloc_config_str.find("backend:cudaMallocAsync") == std::string::npos;
  return Status();
}

void CustomAllReduceSumLayer::Clear() {
  if (reduce_op_ != nullptr) {
    delete static_cast<llm_kernels::nvidia::CustomAllreduce*>(reduce_op_);
    for (uint32_t rank = 0; rank < world_size_; rank++) {
      if (static_cast<int>(rank) == rank_) {
        continue;
      }
      if (signals_[rank] != nullptr) {
        CUDA_CHECK(cudaIpcCloseMemHandle(signals_[rank]));
      }
      if (input_handles_[rank] != nullptr) {
        CUDA_CHECK(cudaIpcCloseMemHandle(input_handles_[rank]));
      }
    }
  }
}

template <typename T>
Status CustomAllReduceSumLayer::ForwardT(const Tensor& input, Tensor& output) {
  cudaStream_t* stream;
  if (context_->global->IsRunContextDecodeAndDecodeSerially()) {
    stream = &(context_->device->GetPrimaryComputeStream().Get());
  } else {
    stream = &(context_->device->GetCommStream().Get());
  }
  if (context_->global->GetTensorParallelSize() > 1) {
    void* inp = input.GetPtr<void>();
    void* result = output.GetPtr<void>();
    int data_size = input.GetElementNumber();
    // Initialize the workspace on the first forward
    if (!is_init_) {
      CustomAllReduceInit(&reduce_op_, rank_data_, rank_data_sz_, rank_, world_size_, is_full_nvlink_, root_rank_);

      // Should register new signal buffer every time later.
      CustomAllReduceRegisterSignalBuffer(reduce_op_, signals_);
      CustomAllReduceRegisterBuffer(reduce_op_, input_handles_, *stream);

      // Must sync, otherwise the CustomAllReduceRun maybe hanged.
      cudaStreamSynchronize(*stream);
      is_init_ = true;
    }
    CustomAllReduceRun<T>(reduce_op_, inp, result, data_size, *stream);
    // To avoid getting stuck during CustomAllReduce.
    if (need_sync_) {
      CUDA_CHECK(cudaStreamSynchronize(*stream));
    }
  } else {
    void* src = input.GetPtr<void>();
    void* dst = output.GetPtr<void>();
    CUDA_CHECK(cudaMemcpyAsync(dst, src, input.GetTotalBytes(), cudaMemcpyDeviceToDevice, *stream));
  }
  output.shape = input.shape;
  output.dtype = input.dtype;
  return Status();
}

Status CustomAllReduceSumLayer::Forward(const Tensor* input, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *input, *output);
}

}  // namespace ksana_llm
