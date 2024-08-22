/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/matmul_layer.h"

#include "csrc/kernels/ascend/matmul/matmul.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

template <typename T>
Status MatMulLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  if (std::is_same<T, float16>::value) {
    aclnn_mm_type_ = llm_kernels::utils::ACLNNMatmulComputeType::USE_FP16;
    aclnn_dtype_ = aclDataType::ACL_FLOAT16;
  } else if (std::is_same<T, float>::value) {
    aclnn_mm_type_ = llm_kernels::utils::ACLNNMatmulComputeType::KEEP_DTYPE;
    aclnn_dtype_ = aclDataType::ACL_FLOAT;
  } else {
    KLLM_THROW("Invalid matmul type, only support float16 or float32.");
  }

#ifdef ENABLE_ACL_ATB
  atb::infer::LinearParam linear_param;
  linear_param.transposeA = false;
  linear_param.transposeB = false;
  linear_param.hasBias = false;
  linear_param.outDataType = ACL_DT_UNDEFINED;
  atb_op_executor_.Init(rank, linear_param);
#endif

  return Status();
}

template <typename T>
Status MatMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // TODO(karlluo): support bias
  int64_t m = input_tensors[0].shape[0];
#ifndef ENABLE_ACL_ATB
  int64_t k = input_tensors[0].shape[1];
#endif
  int64_t n = input_tensors[1].shape[1];
  output_tensors[0].shape = {static_cast<size_t>(m), static_cast<size_t>(n)};
#ifdef ENABLE_ACL_ATB
  reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_))
      ->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_op_executor_.ResetVariantPack();
  atb_op_executor_.SetInputTensor(input_tensors[0].GetPtr<void>(), input_tensors[0].shape,
                                  static_cast<aclDataType>(input_tensors[0].dtype));
  atb_op_executor_.SetInputTensor(input_tensors[1].GetPtr<void>(), input_tensors[1].shape,
                                  static_cast<aclDataType>(input_tensors[1].dtype));
  atb_op_executor_.SetOutputTensor(output_tensors[0].GetPtr<void>(), output_tensors[0].shape,
                                   static_cast<aclDataType>(output_tensors[0].dtype));
  atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_)), GetWorkSpaceFunc());
  ACL_CHECK_RET(aclrtSynchronizeStream(context_->GetComputeStreams()[rank_].Get()));
#else
  std::vector<int64_t> matmul_input_shape = {m, k};
  std::vector<int64_t> matmul_weight_shape = {k, n};
  std::vector<int64_t> matmul_output_shape = {m, n};
  aclTensor* matmul_input = nullptr;
  aclTensor* matmul_weight = nullptr;
  aclTensor* matmul_output = nullptr;
  void* matmul_input_buf_ptr = input_tensors[0].GetPtr<void>();
  void* matmul_weight_buf_ptr = input_tensors[1].GetPtr<void>();
  void* matmul_output_buf_ptr = output_tensors[0].GetPtr<void>();
  llm_kernels::utils::CreateAclTensorWithData(matmul_input_shape, &matmul_input_buf_ptr, aclnn_dtype_,
                                              aclFormat::ACL_FORMAT_ND, &matmul_input);
  llm_kernels::utils::CreateAclTensorWithData(matmul_weight_shape, &matmul_weight_buf_ptr, aclnn_dtype_,
                                              aclFormat::ACL_FORMAT_ND, &matmul_weight);
  llm_kernels::utils::CreateAclTensorWithData(matmul_output_shape, &matmul_output_buf_ptr, aclnn_dtype_,
                                              aclFormat::ACL_FORMAT_ND, &matmul_output);
  llm_kernels::ascend::InvokeAclNNMatMul(matmul_input, matmul_weight, aclnn_mm_type_, &matmul_output,
                                         context_->GetComputeStreams()[rank_].Get(), GetWorkSpaceFunc());
  output_tensors[0].shape = {m, n};
  output_tensors[0].dtype = input_tensors[0].dtype;
  ACL_CHECK(aclDestroyTensor(matmul_input));
  ACL_CHECK(aclDestroyTensor(matmul_weight));
  ACL_CHECK_RET(aclrtSynchronizeStream(context_->GetComputeStreams()[rank_].Get()));
#endif  // ENABLE_ACL_ATB
  return Status();
}

template class MatMulLayer<float>;
template class MatMulLayer<float16>;
}  // namespace ksana_llm
