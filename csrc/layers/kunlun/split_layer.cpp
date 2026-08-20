/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/split_layer.h"
#include "device/kunlun/kunlun_device_context_impl.h"

#include "kernels/kunlun/basic_kernel_wrapper.h"

namespace ksana_llm {

namespace xpukernel = xpytorch::xpu::api;

Status SplitLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                        std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}

Status SplitLayer::Forward(const Tensor* input, const std::vector<size_t>& split_sizes, std::vector<Tensor*> outputs,
                           const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input, split_sizes, outputs);
}

template <typename T>
Status SplitLayer::ForwardT(const Tensor* input, const std::vector<size_t>& split_sizes,
                            std::vector<Tensor*> outputs) {
  // Split input of shape [rows, total_cols] into multiple output tensors along the column dimension.
  // split_sizes[i] = column count for outputs[i], and sum(split_sizes) == total_cols.
  const int64_t rows = static_cast<int64_t>(input->shape[0]);
  const int64_t total_cols = static_cast<int64_t>(input->shape[1]);

  auto* xpuctx = static_cast<xpukernel::Context*>(
      context_->device->ext->GetXpuKernelContext(context_->device->GetPrimaryComputeStream().Get()));

  std::vector<int64_t> xshape = {rows, total_cols};
  std::vector<int64_t> split_list;
  split_list.reserve(split_sizes.size());
  for (size_t i = 0; i < split_sizes.size(); ++i) {
    split_list.push_back(static_cast<int64_t>(split_sizes[i]));
  }

  std::vector<T*> y_list;
  y_list.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    y_list.push_back(outputs[i]->GetPtr<T>());
  }

  int ret = XpuSplit<T>(xpuctx, input->GetPtr<T>(), y_list, xshape, split_list, /*axis=*/1);
  if (ret != 0) {
    KLLM_LOG_ERROR << "SplitLayer: XpuSplit failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "SplitLayer: XpuSplit failed");
  }
  return Status();
}

}  // namespace ksana_llm
