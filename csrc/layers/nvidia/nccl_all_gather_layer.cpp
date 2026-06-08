/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/nccl_all_gather_layer.h"
#include "kernels/nvidia/kernel_wrapper.h"
#include "profiler/profile_event.h"

namespace ksana_llm {

template <typename T>
Status NcclAllGatherLayer::ForwardT(const Tensor& local, Tensor& output, const Tensor* permute_buf,
                                    const LayerExecContext& exec_ctx) {
  size_t tp_size = context_->global->GetTensorParallelSize();
  if (tp_size == 1) {
    return Status();
  }
  bool gather_along_dim_0 = (permute_buf == nullptr);

  size_t h = local.shape[0];
  size_t w_per = local.shape[1];

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
    void* output_ptr = gather_along_dim_0 ? output.GetPtr<void>() : permute_buf->GetPtr<void>();
    NCCL_CHECK(ncclGroupStart());
    ncclResult_t ncclError =
        ncclAllGather(reinterpret_cast<const void*>(local.GetPtr<void>()), output_ptr, local.GetElementNumber(),
                      GetNcclDataType(inter_data_type_), context_->device->ext->GetNCCLParam().nccl_comm, *stream);
    if (ncclError != ncclSuccess) {
      KLLM_LOG_ERROR << fmt::format("NCCL error: {}\n", ncclGetErrorString(ncclError));
      return Status(RetCode::RET_INFER_FAILED, "NCCL error");
    }
    NCCL_CHECK(ncclGroupEnd());
  }

  if (gather_along_dim_0) {
    output.shape = {tp_size * h, w_per};
  } else {
    InvokePermute<T>(permute_buf->GetPtr<void>(), output.GetPtr<void>(), {tp_size, h, w_per}, {1, 0, 2}, *stream);
    output.shape = {h, tp_size * w_per};
  }
  return Status();
}

Status NcclAllGatherLayer::Forward(const Tensor* local, Tensor* output, const Tensor* permute_buf,
                                   const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *local, *output, permute_buf, exec_ctx);
}

}  // namespace ksana_llm
