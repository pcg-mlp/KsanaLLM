/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/cast_layer.h"
#include "device/kunlun/kunlun_device_context_impl.h"

#include "kernels/kunlun/basic_kernel_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {

template <typename SRC_DTYPE>
Status CastLayer::ForwardT(const Tensor* input, size_t byte_offset, size_t vocab_size, size_t vocab_size_pad,
                           Tensor* output) {
  namespace xpukernel = xpytorch::xpu::api;
  auto* xpuctx = static_cast<xpukernel::Context*>(
      context_->device->ext->GetXpuKernelContext(context_->device->GetPrimaryComputeStream().Get()));

  const SRC_DTYPE* src = input->GetPtr<SRC_DTYPE>();
  int64_t len = static_cast<int64_t>(input->GetElementNumber());

  size_t output_offset = byte_offset;
  float* dst = reinterpret_cast<float*>(reinterpret_cast<char*>(output->GetPtr<void>()) + output_offset);

  KLLM_LOG_DEBUG << "CastLayer: len=" << len << " offset=" << output_offset
                 << " src=" << reinterpret_cast<const void*>(src) << " dst=" << reinterpret_cast<void*>(dst)
                 << " shape=[" << input->shape[0]
                 << (input->shape.size() > 1 ? "," + std::to_string(input->shape[1]) : "") << "]";

  int ret = XpuCast<SRC_DTYPE, float>(xpuctx, src, dst, len);
  if (ret != 0) {
    KLLM_LOG_ERROR << "CastLayer: XpuCast FAILED ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "XpuCast failed in CastLayer.");
  }

  output->dtype = DataType::TYPE_FP32;
  output->shape = input->shape;
  return Status();
}

Status CastLayer::Forward(const Tensor* input, size_t byte_offset, size_t vocab_size, size_t vocab_size_pad,
                          Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input, byte_offset, vocab_size, vocab_size_pad, output);
}

}  // namespace ksana_llm
