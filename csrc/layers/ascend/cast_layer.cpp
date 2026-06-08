/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/cast_layer.h"

#include "csrc/utils/ascend/atb_executor.h"
#include "csrc/utils/ascend/common.h"
#include "device/ascend/acl_utils.h"

namespace ksana_llm {

Status CastLayer::Forward(const Tensor* input, size_t byte_offset, size_t vocab_size, size_t vocab_size_pad,
                          Tensor* output, const LayerExecContext& exec_ctx) {
  output->shape = input->shape;
  output->dtype = DataType::TYPE_FP32;
  atb::infer::ElewiseParam param;
  param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
  param.outTensorType = static_cast<aclDataType>(DataType(output->dtype));
  llm_kernels::utils::ATBOperationExecutor atb_op_executor;
  atb_op_executor.Init(rank_, param);
  reinterpret_cast<atb::Context*>(GetRuntimeContext())->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_op_executor.ResetVariantPack();
  atb_op_executor.SetInputTensor(input->GetPtr<void>(), input->shape, static_cast<aclDataType>(DataType(input->dtype)));
  atb_op_executor.SetOutputTensor(reinterpret_cast<uint8_t*>(output->GetPtr<void>()) + byte_offset, output->shape,
                                  static_cast<aclDataType>(DataType(output->dtype)));
  atb_op_executor.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext()), GetWorkSpaceFunc());
  return Status();
}

}  // namespace ksana_llm
