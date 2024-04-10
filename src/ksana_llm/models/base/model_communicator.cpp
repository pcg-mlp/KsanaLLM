/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/model_communicator.h"

namespace ksana_llm {

ModelCommunicator::ModelCommunicator(Tensor* buffer, Tensor* input, int rank, std::shared_ptr<Context> context)
    : rank_(rank), context_(context), buffer_(buffer), input_(input) {
  EventCreateWithFlags(&nccl_finish_event_, EVENT_DISABLE_TIMING);

  nccl_all_reduce_sum_layer_ = std::make_shared<NcclAllReduceSumLayer>();
  nccl_all_reduce_sum_layer_->Init({}, context_, rank_);

  nccl_all_gather_layer_ = std::make_shared<NcclAllGatherLayer>();
  nccl_all_gather_layer_->Init({}, context_, rank_);

  if (enable_custom_all_reduce_) {
    custom_all_reduce_sum_layer_0_ = std::make_shared<CustomAllReduceSumLayer>();

    Event create_reduce_tensor_event;
    EventCreateWithFlags(&create_reduce_tensor_event, EVENT_DISABLE_TIMING);

    constexpr size_t reduce_buffer_size = 256;
    STATUS_CHECK_FAILURE(CreateTensor(reduce_tensor_, {reduce_buffer_size}, TYPE_UINT8, rank_, MEMORY_DEVICE));

    size_t rank_data_sz = context_->GetTensorParallelSize() * 128;
    STATUS_CHECK_FAILURE(CreateTensor(rank_tensor_0_, {rank_data_sz}, TYPE_UINT8, rank_, MEMORY_DEVICE));
    EventRecord(create_reduce_tensor_event, context_->GetMemoryManageStreams()[rank_]);

    StreamWaitEvent(context_->GetMemoryManageStreams()[rank_], create_reduce_tensor_event);

    MemsetAsync(reduce_tensor_.GetPtr<void>(), 0, reduce_buffer_size, context_->GetMemoryManageStreams()[rank_]);

    custom_all_reduce_sum_layer_0_->Init({reduce_tensor_.GetPtr<void>(), buffer_->GetPtr<void>(), reduce_buffer_size,
                                          rank_tensor_0_.GetPtr<void>(), rank_data_sz, input_->GetPtr<void>(), 0},
                                         context_, rank_);
    EventDestroy(create_reduce_tensor_event);
  }
}
ModelCommunicator::~ModelCommunicator() {
  STATUS_CHECK_FAILURE(DestroyTensor(reduce_tensor_, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(rank_tensor_0_, rank_));

  EventDestroy(nccl_finish_event_);
}

Status ModelCommunicator::AllGather(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(nccl_all_gather_layer_->Forward(input_tensors, output_tensors));
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(nccl_finish_event_, context_->GetNCCLStreams()[rank_]);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], nccl_finish_event_);
  }
  return Status();
}

Status ModelCommunicator::ReduceSum(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                                    bool is_context_stage, bool use_custom) {
  if (is_context_stage) {
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(input_tensors, output_tensors));
  } else {
    if (enable_custom_all_reduce_ && use_custom) {
      STATUS_CHECK_RETURN(custom_all_reduce_sum_layer_0_->Forward({input_tensors[0]}, output_tensors));
    } else {
      STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(input_tensors, output_tensors));
    }
  }
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(nccl_finish_event_, context_->GetNCCLStreams()[rank_]);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], nccl_finish_event_);
  }
  return Status();
}

}  // namespace ksana_llm
