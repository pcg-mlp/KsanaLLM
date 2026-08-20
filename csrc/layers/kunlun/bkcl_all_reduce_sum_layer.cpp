/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/bkcl_all_reduce_sum_layer.h"

#include "device/kunlun/bkcl_utils.h"

namespace ksana_llm {

Status BkclAllReduceSumLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                   std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

Status BkclAllReduceSumLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  XPUStream stream;
  if (context_->global->IsRunContextDecodeAndDecodeSerially()) {
    stream = context_->device->GetPrimaryComputeStream().Get();
  } else {
    stream = context_->device->GetCommStream().Get();
  }

  if (context_->global->GetTensorParallelSize() > 1) {
    BKCLContext_t comm = context_->device->ext->GetBKCLParam().bkcl_comm;
    BKCLDataType dtype = GetBkclDataType(inter_data_type_);
    size_t count = input_tensors[0].GetElementNumber();

    BKCLResult_t ret = bkcl_all_reduce(comm, input_tensors[0].GetPtr<void>(), output_tensors[0].GetPtr<void>(), count,
                                       dtype, BKCL_ADD, stream);
    if (ret != BKCL_SUCCESS) {
      KLLM_LOG_ERROR << fmt::format("BKCL allreduce error: {}", GetBKCLErrorString(ret));
      return Status(RetCode::RET_INFER_FAILED, "BKCL allreduce error");
    }
  } else {
    MemcpyAsync(output_tensors[0].GetPtr<void>(), input_tensors[0].GetPtr<void>(), input_tensors[0].GetTotalBytes(),
                MEMCPY_DEVICE_TO_DEVICE, context_->device->GetPrimaryComputeStream());
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm
