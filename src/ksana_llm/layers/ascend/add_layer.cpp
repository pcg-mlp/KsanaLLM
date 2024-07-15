/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/add_layer.h"

#include "csrc/kernels/ascend/add/add.h"
#include "csrc/kernels/ascend/elementwise/elementwise.h"
#include "csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

template <typename T>
Status AddLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (input_tensors[0].shape[0] == input_tensors[1].shape[0]) {
    int64_t total_seq_len = input_tensors[0].shape[0];
    int64_t hidden_size = input_tensors[0].shape[1];
    void* add_out_buf = output_tensors[0].GetPtr<void>();
    void* a_ptr = reinterpret_cast<void*>(input_tensors[0].GetPtr<void>());
    void* b_ptr = reinterpret_cast<void*>(input_tensors[1].GetPtr<void>());
    llm_kernels::ascend::InvokeAdd<T>(reinterpret_cast<T*>(a_ptr), reinterpret_cast<T*>(b_ptr), nullptr,
                                      reinterpret_cast<T*>(add_out_buf), static_cast<uint32_t>(hidden_size),
                                      static_cast<uint32_t>(total_seq_len), context_->GetComputeStreams()[rank_].Get(),
                                      GetWorkSpaceFunc());
  } else {
    return Status(RET_SEGMENT_FAULT, "add bias not implemented");
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}
template class AddLayer<float>;
template class AddLayer<float16>;
}  // namespace ksana_llm
