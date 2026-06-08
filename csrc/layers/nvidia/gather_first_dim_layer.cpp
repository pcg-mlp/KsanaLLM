/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/gather_first_dim_layer.h"

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/gather/gather.h"
#include "device/nvidia/cuda_utils.h"

namespace ksana_llm {

template <typename SRC_DTYPE>
Status GatherFirstDimLayer::ForwardT(const Tensor& data, const Tensor& indices, Tensor& output) {
  const size_t token_num = data.shape[0];
  const size_t gathered_token_num = indices.shape[0];

  if (gathered_token_num > 0) {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::GatherFirstDim(
        data.GetPtr<void>(), indices.GetPtr<int>(), gathered_token_num, data.GetTotalBytes() / token_num,
        output.GetPtr<void>(), context_->device->GetPrimaryComputeStream().Get()));
  }

  output.shape = {gathered_token_num, data.shape[1]};
  output.dtype = data.dtype;

  return Status();
}

Status GatherFirstDimLayer::Forward(const Tensor* data, const Tensor* indices, Tensor* output,
                                    const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *data, *indices, *output);
}

}  // namespace ksana_llm
