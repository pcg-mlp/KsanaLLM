/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/layernorm_layer.h"

#include <cstdint>

#include "kernels/ascend/kernel_wrapper.h"

#include "csrc/utils/ascend/common.h"
#include "device/ascend/acl_utils.h"

namespace ksana_llm {

Status LayernormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  int parameter_index = 0;
  context_ = context;
  rank_ = rank;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);

  atb::infer::RmsNormParam rms_norm_param;
  rms_norm_param.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  rms_norm_param.normParam.epsilon = rms_norm_eps_;
  rms_norm_param.normParam.layerNormEps = rms_norm_eps_;
  atb_op_executor_.Init(rank, rms_norm_param);

  return Status();
}

template <typename T>
Status LayernormLayer::ForwardNoWeightT(const Tensor& input, Tensor& output) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer: no-weight per-head RMSNorm not supported on Ascend.");
}

template <typename T>
Status LayernormLayer::ForwardWeightT(const Tensor& input, const Tensor& weight, Tensor& output, const Tensor* bias) {
  if (bias != nullptr) {
    return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer: bias not supported on Ascend.");
  }
  void* lm_input_buf_ptr = input.GetPtr<void>();
  void* lm_weight_buf_ptr = weight.GetPtr<void>();
  void* lm_output_buf_ptr = output.GetPtr<void>();
  output.shape = input.shape;
  output.dtype = input.dtype;
  reinterpret_cast<atb::Context*>(GetRuntimeContext())->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_op_executor_.ResetVariantPack();
  atb_op_executor_.SetInputTensor(lm_input_buf_ptr, input.shape, static_cast<aclDataType>(DataType(input.dtype)));
  atb_op_executor_.SetInputTensor(lm_weight_buf_ptr, weight.shape, static_cast<aclDataType>(DataType(weight.dtype)));
  atb_op_executor_.SetOutputTensor(lm_output_buf_ptr, output.shape, static_cast<aclDataType>(DataType(output.dtype)));
  atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext()), GetWorkSpaceFunc());
  return Status();
}

template <typename T>
Status LayernormLayer::ForwardParallelT(const Tensor& in1, const Tensor& w1, const Tensor& in2, const Tensor& w2,
                                        Tensor& out1, Tensor& out2, int stride1, int stride2) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer: parallel norm not supported on Ascend.");
}

Status LayernormLayer::Forward(const Tensor* input, Tensor* output, const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer: no-weight RMSNorm not supported on Ascend.");
}

Status LayernormLayer::Forward(const Tensor* input, const Tensor* weight, Tensor* output, const Tensor* bias,
                               const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardWeightT, *input, *weight, *output, bias);
}

Status LayernormLayer::Forward(const Tensor* input1, const Tensor* weight1, const Tensor* input2, const Tensor* weight2,
                               Tensor* output1, Tensor* output2, int stride1, int stride2,
                               const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer: parallel norm not supported on Ascend.");
}

}  // namespace ksana_llm
