/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/hccl_all_reduce_sum_layer.h"

namespace ksana_llm {

Status HcclAllReduceSumLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                   std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  context_ = context;
  rank_ = rank;

  atb::infer::AllReduceParam all_reduce_param;
  all_reduce_param.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
  all_reduce_param.rank = rank;
  all_reduce_param.hcclComm = context_->ext->GetHCCLComm()[rank_];
  all_reduce_param.rankSize = context_->GetComputeStreams().size();
  atb_op_executor_.Init(rank, all_reduce_param);
  return Status();
}

Status HcclAllReduceSumLayer::Forward(const Tensor* input, Tensor* output, const LayerExecContext& exec_ctx) {
  output->shape = input->shape;
  output->dtype = input->dtype;

  if (context_->GetTensorParallelSize() > 1) {
    reinterpret_cast<atb::Context*>(GetRuntimeContext())->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
    atb_op_executor_.ResetVariantPack();
    atb_op_executor_.SetInputTensor(input->GetPtr<void>(), input->shape,
                                    static_cast<aclDataType>(DataType(input->dtype)));
    atb_op_executor_.SetOutputTensor(output->GetPtr<void>(), output->shape,
                                     static_cast<aclDataType>(DataType(output->dtype)));
    atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext()), GetWorkSpaceFunc());
  } else {
    void* src = input->GetPtr<void>();
    void* dst = output->GetPtr<void>();
    MemcpyAsync(dst, src, input->GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
  }
  return Status();
}

}  // namespace ksana_llm
