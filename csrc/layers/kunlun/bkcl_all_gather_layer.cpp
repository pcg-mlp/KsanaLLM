/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/bkcl_all_gather_layer.h"

#include "device/device_utils.h"
#include "device/kunlun/bkcl_utils.h"
#include "kernels/permute.h"

namespace ksana_llm {

Status BkclAllGatherLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  size_t tp_size = context_->global->GetTensorParallelSize();
  if (tp_size <= 1) {
    return Status();
  }

  XPUStream stream;
  if (context_->global->IsRunContextDecodeAndDecodeSerially()) {
    stream = context_->device->GetPrimaryComputeStream().Get();
  } else {
    stream = context_->device->GetCommStream().Get();
  }

  BKCLContext_t comm = context_->device->ext->GetBKCLParam().bkcl_comm;
  BKCLDataType dtype = GetBkclDataType(inter_data_type_);
  size_t send_count = input_tensors[0].GetElementNumber();

  // If input_tensors.size() == 1: gather along dim 0, output directly to output_tensors[0].
  // If input_tensors.size() >= 2: gather along dim 1 (hidden dimension).
  //   input_tensors[1] is used as a temp buffer for the all_gather result,
  //   then we permute [tp_size, h, w_per] → [h, tp_size, w_per] → reshape [h, tp_size * w_per].
  //   This matches the NCCL path (nccl_all_gather_layer.cpp).
  bool gather_along_dim_0 = (input_tensors.size() == 1);

  size_t h = input_tensors[0].shape[0];
  size_t w_per = input_tensors[0].shape[1];

  void* gather_dst = gather_along_dim_0 ? output_tensors[0].GetPtr<void>() : input_tensors[1].GetPtr<void>();

  BKCLResult_t ret = bkcl_all_gather(comm, input_tensors[0].GetPtr<void>(), send_count, gather_dst, dtype, stream);
  if (ret != BKCL_SUCCESS) {
    KLLM_LOG_ERROR << fmt::format("BKCL allgather error: {}", GetBKCLErrorString(ret));
    return Status(RetCode::RET_INFER_FAILED, "BKCL allgather error");
  }

  if (gather_along_dim_0) {
    output_tensors[0].shape = {tp_size * h, w_per};
  } else {
    // Permute [tp_size, h, w_per] → [h, tp_size, w_per] via xdnn::transpose kernel.
    // This is equivalent to the NVIDIA path's InvokePermute<T>(..., {1,0,2}, stream).
    auto& rearrange_stream = context_->global->IsRunContextDecodeAndDecodeSerially()
                                 ? context_->device->GetPrimaryComputeStream()
                                 : context_->device->GetCommStream();

    // Synchronize to ensure bkcl_all_gather has fully completed before permute reads the data.
    // BKCL may use internal streams for data transfer; the user-facing stream only guarantees
    // the operation has been enqueued, not that the data is ready in the output buffer.
    StreamSynchronize(rearrange_stream);

    // Set up temp tensor with 3D shape for permute
    Tensor temp_3d = input_tensors[1];
    temp_3d.shape = {tp_size, h, w_per};

    Tensor output_3d = output_tensors[0];
    output_3d.shape = {h, tp_size, w_per};

    STATUS_CHECK_RETURN(Permute(temp_3d, output_3d, {1, 0, 2}, rearrange_stream,
                                context_->device->ext->GetXdnnContext(rearrange_stream.Get())));

    output_tensors[0].shape = {h, tp_size * w_per};
  }
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm
