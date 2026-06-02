/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/nccl_all_gather_layer.h"
#include "kernels/nvidia/kernel_wrapper.h"
#include "profiler/profile_event.h"

namespace ksana_llm {

Status NcclAllGatherLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status NcclAllGatherLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  size_t tp_size = context_->global->GetTensorParallelSize();
  if (tp_size == 1) {
    return Status();
  }
  // if gather along dim 1, need to permute, so need a buffer in input_tensors for permute
  bool gather_along_dim_0 = input_tensors.size() == 1;

  size_t h = input_tensors[0].shape[0];
  size_t w_per = input_tensors[0].shape[1];

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl stream just enable when IsRunContextDecodeAndDecodeSerially == false
  cudaStream_t* stream;
  if (context_->global->IsRunContextDecodeAndDecodeSerially()) {
    stream = &(context_->device->GetPrimaryComputeStream().Get());
  } else {
    stream = &(context_->device->GetCommStream().Get());
  }

  {
    PROFILE_EVENT_SCOPE(nccl_allgather_, "nccl_allgather_{}", rank_);
    void* output_ptr = gather_along_dim_0 ? output_tensors[0].GetPtr<void>() : input_tensors[1].GetPtr<void>();
    NCCL_CHECK(ncclGroupStart());
    ncclResult_t ncclError = ncclAllGather(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()), output_ptr,
                                           input_tensors[0].GetElementNumber(), GetNcclDataType(inter_data_type_),
                                           context_->device->ext->GetNCCLParam().nccl_comm, *stream);
    if (ncclError != ncclSuccess) {
      KLLM_LOG_ERROR << fmt::format("NCCL error: {}\n", ncclGetErrorString(ncclError));
      return Status(RetCode::RET_INFER_FAILED, "NCCL error");
    }
    NCCL_CHECK(ncclGroupEnd());
  }

  if (gather_along_dim_0) {
    output_tensors[0].shape = {tp_size * h, w_per};
  } else {
    InvokePermute<T>(input_tensors[1].GetPtr<void>(), output_tensors[0].GetPtr<void>(), {tp_size, h, w_per}, {1, 0, 2},
                     *stream);
    output_tensors[0].shape = {h, tp_size * w_per};
  }
  return Status();
}

}  // namespace ksana_llm
