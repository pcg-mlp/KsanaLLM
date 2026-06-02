/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/gather_first_dim_layer.h"

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/gather/gather.h"
#include "device/nvidia/cuda_utils.h"

namespace ksana_llm {

Status GatherFirstDimLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename SRC_DTYPE>
Status GatherFirstDimLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //   0: data (void*)
  //   1: copy index of data (int*)
  // Copy data from input at specified indices to output
  const size_t token_num = input_tensors[0].shape[0];
  const size_t gathered_token_num = input_tensors[1].shape[0];

  if (gathered_token_num > 0) {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::GatherFirstDim(
        input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<int>(), gathered_token_num,
        input_tensors[0].GetTotalBytes() / token_num, output_tensors[0].GetPtr<void>(),
        context_->device->GetPrimaryComputeStream().Get()));
  }

  output_tensors[0].shape = {gathered_token_num, input_tensors[0].shape[1]};
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}

}  // namespace ksana_llm
