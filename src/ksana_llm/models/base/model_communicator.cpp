/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/model_communicator.h"

namespace ksana_llm {

template <typename T>
ModelCommunicator<T>::ModelCommunicator(Tensor* buffer, Tensor* input, int rank, std::shared_ptr<Context> context)
    : rank_(rank), context_(context), buffer_(buffer), input_(input) {
  EventCreateWithFlags(&nccl_finish_event_, EVENT_DISABLE_TIMING);

#ifdef ENABLE_CUDA
  nccl_all_reduce_sum_layer_ = std::make_shared<NcclAllReduceSumLayer<T>>();
  nccl_all_reduce_sum_layer_->Init({}, context_, rank_);

  nccl_all_gather_layer_ = std::make_shared<NcclAllGatherLayer<T>>();
  nccl_all_gather_layer_->Init({}, context_, rank_);

  // TODO(catheywang): CustomAllReduceSum not supported on more than two PCIe-only GPUs.
  enable_custom_all_reduce_ &= context->GetTensorParallelSize() == 2;
  if (enable_custom_all_reduce_) {
    custom_all_reduce_sum_layer_0_ = std::make_shared<CustomAllReduceSumLayer<T>>();

    Event create_reduce_tensor_event;
    EventCreateWithFlags(&create_reduce_tensor_event, EVENT_DISABLE_TIMING);

    constexpr size_t reduce_buffer_size = 256;
    STATUS_CHECK_FAILURE(CreateTensor(reduce_tensor_, {reduce_buffer_size}, TYPE_UINT8, rank_, MEMORY_DEVICE));

    size_t rank_data_sz = context_->GetTensorParallelSize() * 128;
    STATUS_CHECK_FAILURE(CreateTensor(rank_tensor_0_, {rank_data_sz}, TYPE_UINT8, rank_, MEMORY_DEVICE));
    EventRecord(create_reduce_tensor_event, context_->GetMemoryManageStreams()[rank_]);

    StreamWaitEvent(context_->GetMemoryManageStreams()[rank_], create_reduce_tensor_event);

    MemsetAsync(reduce_tensor_.GetPtr<void>(), 0, reduce_buffer_size, context_->GetMemoryManageStreams()[rank_]);

    custom_all_reduce_sum_layer_0_->Init({reduce_tensor_.GetPtr<void>(), buffer_->GetPtr<void>(),
                                          rank_tensor_0_.GetPtr<void>(), rank_data_sz, input_->GetPtr<void>(), 0},
                                         context_, rank_);
    EventDestroy(create_reduce_tensor_event);
  }
#elif defined(ENABLE_ACL)
  hccl_all_reduce_sum_layer_ = std::make_shared<HcclAllReduceSumLayer<T>>();
  hccl_all_reduce_sum_layer_->Init({}, context, rank);

  hccl_all_gather_layer_ = std::make_shared<HcclAllGatherLayer<T>>();
  hccl_all_gather_layer_->Init({}, context, rank);
#endif
}
template <typename T>
ModelCommunicator<T>::~ModelCommunicator() {
#ifdef ENABLE_CUDA
  STATUS_CHECK_FAILURE(DestroyTensor(reduce_tensor_, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(rank_tensor_0_, rank_));
#endif

  EventDestroy(nccl_finish_event_);
}

template <typename T>
Status ModelCommunicator<T>::AllGather(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
#ifdef ENABLE_CUDA
  STATUS_CHECK_RETURN(nccl_all_gather_layer_->Forward(input_tensors, output_tensors));
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(nccl_finish_event_, context_->GetNCCLStreams()[rank_]);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], nccl_finish_event_);
  }
#endif

#ifdef ENABLE_ACL
  MemcpyAsync(output_tensors[0].GetPtr<void>(), input_tensors[0].GetPtr<void>(), input_tensors[0].GetTotalBytes(),
              MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);
#endif
  return Status();
}

template <typename T>
Status ModelCommunicator<T>::ReduceSum(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                                       bool is_context_stage, bool use_custom) {
#ifdef ENABLE_CUDA
  if (is_context_stage) {
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(input_tensors, output_tensors));
  } else {
    if (enable_custom_all_reduce_ && use_custom) {
      STATUS_CHECK_RETURN(custom_all_reduce_sum_layer_0_->Forward(input_tensors, output_tensors));
    } else {
      STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(input_tensors, output_tensors));
    }
  }
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(nccl_finish_event_, context_->GetNCCLStreams()[rank_]);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], nccl_finish_event_);
  }
#endif

#ifdef ENABLE_ACL
  MemcpyAsync(output_tensors[0].GetPtr<void>(), input_tensors[0].GetPtr<void>(), input_tensors[0].GetTotalBytes(),
              MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
#endif

  return Status();
}

template class ModelCommunicator<float>;
template class ModelCommunicator<float16>;
#ifdef ENABLE_BFLOAT16
template class ModelCommunicator<bfloat16>;
#endif
}  // namespace ksana_llm
