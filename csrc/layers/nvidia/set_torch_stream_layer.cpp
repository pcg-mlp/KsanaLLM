/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/set_torch_stream_layer.h"
#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status SetTorchStreamLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                 std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}
void SetTorchStreamLayer::Clear() { torch_stream_ = InvokeSetTorchStream(torch_stream_, this->rank_); }

Status SetTorchStreamLayer::Forward(const LayerExecContext& exec_ctx) {
  torch_stream_ = InvokeSetTorchStream(this->context_->device->GetPrimaryComputeStream().Get(), this->rank_);
  return Status();
}

}  // namespace ksana_llm
