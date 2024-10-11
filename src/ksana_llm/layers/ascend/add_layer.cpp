/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/add_layer.h"

#include "csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

template <typename T>
Status AddLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);

  atb::infer::ElewiseParam elewise_param;
  elewise_param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
  atb_op_executor_.Init(rank, elewise_param);
  return Status();
}

template <typename T>
Status AddLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;

  if (input_tensors[0].shape[0] == input_tensors[1].shape[0]) {
    void* out_ptr = output_tensors[0].GetPtr<void>();
    void* a_ptr = reinterpret_cast<void*>(input_tensors[0].GetPtr<void>());
    void* b_ptr = reinterpret_cast<void*>(input_tensors[1].GetPtr<void>());
    reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_))
        ->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
    atb_op_executor_.ResetVariantPack();
    atb_op_executor_.SetInputTensor(a_ptr, input_tensors[0].shape, static_cast<aclDataType>(input_tensors[0].dtype));
    atb_op_executor_.SetInputTensor(b_ptr, input_tensors[1].shape, static_cast<aclDataType>(input_tensors[1].dtype));
    atb_op_executor_.SetOutputTensor(out_ptr, output_tensors[0].shape,
                                     static_cast<aclDataType>(output_tensors[0].dtype));
    atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_)), GetWorkSpaceFunc());
  } else {
    return Status(RET_SEGMENT_FAULT, "add bias not implemented");
  }

  return Status();
}
template class AddLayer<float>;
template class AddLayer<float16>;
#ifdef ENABLE_BFLOAT16
template class AddLayer<bfloat16>;
#endif
}  // namespace ksana_llm
