/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/model_communicator.h"

#include "configure/environment.h"
#include "utils/singleton.h"

#ifdef ENABLE_CUDA
#  include "3rdparty/LLM_kernels/csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#endif

namespace ksana_llm {

ModelCommunicator::ModelCommunicator(Tensor* input, int rank, const RuntimeConfig& runtime_config,
                                     std::shared_ptr<Context> context)
    : rank_(rank), context_(context), input_(input) {
  EventCreateWithFlags(&comm_finish_event_, EVENT_DISABLE_TIMING);

#ifdef ENABLE_CUDA
  // TODO(rockcao): set SELECT_ALL_REDUCE_BY_SIZE as true by default
  if (std::getenv("SELECT_ALL_REDUCE_BY_SIZE") != nullptr) {
    KLLM_LOG_INFO << "SELECT_ALL_REDUCE_BY_SIZE is enabled";
    select_all_reduce_by_size_ = true;
  }

  nccl_all_reduce_sum_layer_ = std::make_shared<NcclAllReduceSumLayer>();
  nccl_all_reduce_sum_layer_->Init({}, runtime_config, context_, rank_);

  nccl_all_gather_layer_ = std::make_shared<NcclAllGatherLayer>();
  nccl_all_gather_layer_->Init({}, runtime_config, context_, rank_);

  is_full_nvlink_ = context_->device->ext->IsFullNvLink();

  tp_size_ = context_->global->GetTensorParallelSize();

  if (context_->device->ext->IsP2PAccessSupported()) {
    if (tp_size_ == 2 || context_->device->ext->IsFullNvLink()) {
      // CustomReduceSumLayer for tensor parallelism
      InitTensorParaCustomAllReduceSumLayer(input, runtime_config);
    }

    // TODO(yfnjin): Temporarily disable trt reduce for mla-based models due to precision issues
    ModelConfig model_config;
    Singleton<Environment>::GetInstance()->GetModelConfig(model_config);
    if (tp_size_ == 2 && !model_config.use_mla) {
      // When nvlink is not available, lower the threshold for using trt allreduce
      if (!context_->device->ext->IsFullNvLink()) {
        kTrtReduceThreshold >>= 1;
      }
      // TrtReduceSumLayer for tensor parallelism
      tp_trt_all_reduce_sum_layer_ = std::make_shared<TrtAllReduceSumLayer>();
      tp_trt_all_reduce_sum_layer_->Init({kTrtReduceThreshold}, runtime_config, context_, rank_);
    }
  }
#elif defined(ENABLE_ACL)
  hccl_all_reduce_sum_layer_ = std::make_shared<HcclAllReduceSumLayer>();
  hccl_all_reduce_sum_layer_->Init({}, runtime_config, context, rank);

  hccl_all_gather_layer_ = std::make_shared<HcclAllGatherLayer>();
  hccl_all_gather_layer_->Init({}, runtime_config, context, rank);
#endif
}

#ifdef ENABLE_CUDA

void ModelCommunicator::InitTensorParaCustomAllReduceSumLayer(Tensor* input, const RuntimeConfig& runtime_config) {
  size_t max_size = input->GetTotalBytes();
  size_t largest_part = max_size / tp_size_ + max_size % tp_size_;
  size_t signal_sz = sizeof(llm_kernels::nvidia::Signal) + largest_part;
  Stream* stream = &(context_->device->GetMemoryManageStream());

  // Use fixed size now, change to dynamic size later.
  tp_signal_tensor_ = Tensor(MemoryLocation::LOCATION_DEVICE_IPC, TYPE_UINT8, {signal_sz}, nullptr, stream);

  // This is a buffer for storing the tuples of pointers pointing to
  // IPC buffers from all ranks. Each registered tuple has size of
  // 8*world_size bytes where world_size is at most 8. Allocating 8MB
  // is enough for 131072 such tuples. The largest model I've seen only
  // needs less than 10000 of registered tuples.
  constexpr size_t rank_data_sz = 8 * 1024 * 1024;
  tp_custom_all_reduce_sum_layer_ = std::make_shared<CustomAllReduceSumLayer>();
  tp_custom_all_reduce_rank_tensor_ =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {rank_data_sz}, nullptr, stream);
  StreamSynchronize(*stream);

  tp_custom_all_reduce_sum_layer_->Init(
      {input->GetPtr<void>(false), tp_signal_tensor_.GetPtr<void>(false), signal_sz,
       tp_custom_all_reduce_rank_tensor_.GetPtr<void>(), rank_data_sz, /*is_group_custom_all_reduce*/ false},
      runtime_config, context_, rank_);
}

#endif

ModelCommunicator::~ModelCommunicator() {
  EventDestroy(comm_finish_event_);
  if (tp_custom_all_reduce_sum_layer_) {
    tp_custom_all_reduce_sum_layer_->Clear();
  }
  if (tp_trt_all_reduce_sum_layer_) {
    tp_trt_all_reduce_sum_layer_->Clear();
  }
}

Status ModelCommunicator::AllGather(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
#ifdef ENABLE_CUDA
  STATUS_CHECK_RETURN(nccl_all_gather_layer_->Forward(input_tensors, output_tensors));
  if (!context_->global->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(comm_finish_event_, context_->device->GetCommStream());
    StreamWaitEvent(context_->device->GetPrimaryComputeStream(), comm_finish_event_);
  }
#endif

#ifdef ENABLE_ACL
  if (context_->global->GetTensorParallelSize() > 1) {
    STATUS_CHECK_RETURN(hccl_all_gather_layer_->Forward(input_tensors, output_tensors));
    if (!context_->global->IsRunContextDecodeAndDecodeSerially()) {
      EventRecord(comm_finish_event_, context_->device->GetCommStream());
      StreamWaitEvent(context_->device->GetPrimaryComputeStream(), comm_finish_event_);
    }
  } else {
    MemcpyAsync(output_tensors[0].GetPtr<void>(), input_tensors[0].GetPtr<void>(), input_tensors[0].GetTotalBytes(),
                MEMCPY_DEVICE_TO_DEVICE, context_->device->GetPrimaryComputeStream());
  }
#endif
  return Status();
}

Status ModelCommunicator::ReduceSum(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                                    bool is_multi_token_forward, bool use_custom) {
#ifdef ENABLE_CUDA
  const size_t data_size = input_tensors[0].GetTotalBytes();
  if (use_custom && tp_trt_all_reduce_sum_layer_ != nullptr && data_size <= kTrtReduceThreshold) {
    STATUS_CHECK_RETURN(tp_trt_all_reduce_sum_layer_->Forward(input_tensors, output_tensors));
  } else if (use_custom && tp_custom_all_reduce_sum_layer_ != nullptr &&
             !(select_all_reduce_by_size_ && data_size > kAllReduceThreshold)) {
    // custom all reduce may hang in dynamic buffer mode, disable it temporarily. Reopen it after this bug fixed.
    STATUS_CHECK_RETURN(tp_custom_all_reduce_sum_layer_->Forward(input_tensors, output_tensors));
  } else {
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(input_tensors, output_tensors));
  }

  if (!context_->global->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(comm_finish_event_, context_->device->GetCommStream());
    StreamWaitEvent(context_->device->GetPrimaryComputeStream(), comm_finish_event_);
  }
#endif

#ifdef ENABLE_ACL
  if (context_->global->GetTensorParallelSize() > 1) {
    STATUS_CHECK_RETURN(hccl_all_reduce_sum_layer_->Forward(input_tensors, output_tensors));
    if (!context_->global->IsRunContextDecodeAndDecodeSerially()) {
      EventRecord(comm_finish_event_, context_->device->GetCommStream());
      StreamWaitEvent(context_->device->GetPrimaryComputeStream(), comm_finish_event_);
    }
  } else {
    MemcpyAsync(output_tensors[0].GetPtr<void>(), input_tensors[0].GetPtr<void>(), input_tensors[0].GetTotalBytes(),
                MEMCPY_DEVICE_TO_DEVICE, context_->device->GetPrimaryComputeStream());
    output_tensors[0].shape = input_tensors[0].shape;
    output_tensors[0].dtype = input_tensors[0].dtype;
  }
#endif

  return Status();
}

}  // namespace ksana_llm
