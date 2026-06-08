/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/hccl_all_gather_layer.h"

namespace ksana_llm {

Status HcclAllGatherLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  context_ = context;
  rank_ = rank;

  atb::infer::AllGatherParam all_gather_param;
  all_gather_param.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
  all_gather_param.rank = rank;
  all_gather_param.hcclComm = context_->ext->GetHCCLComm()[rank_];
  all_gather_param.rankSize = context_->GetComputeStreams().size();
  atb_all_gather_executor_.Init(rank, all_gather_param);

  atb::infer::TransposeParam permute_param;
  permute_param.perm.push_back(1);
  permute_param.perm.push_back(0);
  permute_param.perm.push_back(2);
  atb_permute_executor_.Init(rank, permute_param);
  return Status();
}

Status HcclAllGatherLayer::Forward(const Tensor* local, const Tensor* internal_buf, Tensor* output,
                                   const LayerExecContext& exec_ctx) {
  output->dtype = local->dtype;
  size_t tp_size = context_->GetTensorParallelSize();
  if (tp_size == 1) {
    return Status();
  }
  size_t h = local->shape[0];
  size_t w_per = local->shape[1];
  output->shape = {h, tp_size, w_per};
  std::vector<size_t> internal_tensor_shape = {tp_size, h, w_per};
  // all gather op
  reinterpret_cast<atb::Context*>(GetRuntimeContext())->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_all_gather_executor_.ResetVariantPack();
  atb_all_gather_executor_.SetInputTensor(local->GetPtr<void>(), local->shape,
                                          static_cast<aclDataType>(DataType(local->dtype)));
  atb_all_gather_executor_.SetOutputTensor(internal_buf->GetPtr<void>(), internal_tensor_shape,
                                           static_cast<aclDataType>(DataType(internal_buf->dtype)));
  atb_all_gather_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext()), GetWorkSpaceFunc());
  // permute op
  atb_permute_executor_.ResetVariantPack();
  atb_permute_executor_.SetInputTensor(internal_buf->GetPtr<void>(), internal_tensor_shape,
                                       static_cast<aclDataType>(DataType(internal_buf->dtype)));
  atb_permute_executor_.SetOutputTensor(output->GetPtr<void>(), output->shape,
                                        static_cast<aclDataType>(DataType(output->dtype)));
  atb_permute_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext()), GetWorkSpaceFunc());
  output->shape = {h, tp_size * w_per};
  return Status();
}

}  // namespace ksana_llm
