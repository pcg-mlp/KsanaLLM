/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/cast_layer.h"

#include "csrc/kernels/ascend/pointwise/pointwise.h"
#include "csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

template <typename T>
Status CastLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  GetBlockManager()->SetDeviceId(rank_);
  Tensor* input_normal_tensor_ptr = (Tensor*)(&(input_tensors[0]));
  aclTensor* input_tensor_ptr = input_normal_tensor_ptr->GetDeviceTensor();
  std::vector<int64_t> input_shape = GetAclTensorShape(input_tensor_ptr);
  aclTensor* reshaped_output_tensor = nullptr;
  void* output_buffer_space_ptr = output_tensors[0].GetPtr<void>();
  llm_kernels::utils::CreateAclTensorWithData(input_shape, &(output_buffer_space_ptr), aclDataType::ACL_FLOAT,
                                              aclFormat::ACL_FORMAT_ND, &reshaped_output_tensor);
  llm_kernels::ascend::Cast(input_tensor_ptr, aclDataType::ACL_FLOAT, &reshaped_output_tensor,
                            context_->GetComputeStreams()[rank_].Get(), GetWorkSpaceFunc());
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = DataType::TYPE_FP32;
  return Status();
}
template class CastLayer<float>;
template class CastLayer<float16>;
}  // namespace ksana_llm
