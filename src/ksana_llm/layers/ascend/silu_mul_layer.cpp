/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/silu_mul_layer.h"

#include "csrc/kernels/ascend/silu_mul/silu_mul.h"
#include "csrc/utils/ascend/common.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

template <typename T>
Status SiluMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int64_t total_seq_len = static_cast<int64_t>(input_tensors[0].shape[0]);
  int64_t ffn_size = static_cast<int64_t>(input_tensors[0].shape[1]);

  void* silu_input_buf_ptr = input_tensors[0].GetPtr<void>();
  void* silu_output_buf_ptr = output_tensors[0].GetPtr<void>();
  void* gated_weight_buf_ptr = input_tensors[1].GetPtr<void>();

  llm_kernels::ascend::InvokeSiluMul<T>((T*)silu_input_buf_ptr, (T*)gated_weight_buf_ptr, total_seq_len, ffn_size,
                                        (T*)silu_output_buf_ptr, context_->GetComputeStreams()[rank_].Get(),
                                        GetWorkSpaceFunc());
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}
template class SiluMulLayer<float>;
template class SiluMulLayer<float16>;
}  // namespace ksana_llm
