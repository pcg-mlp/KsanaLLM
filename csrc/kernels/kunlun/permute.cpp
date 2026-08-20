/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "kernels/permute.h"

#include <cstring>
#include <vector>

#include "device/device_utils.h"
#include "kernels/kunlun/basic_kernel_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {

// xdnn::transpose requires SDK's own float16/bfloat16 struct types, but Ksana
// typedefs float16 as int16_t. Dispatch outside ksana_llm namespace to use ::float16.
namespace {
int XdnnTranspose(xdnn::Context* ctx, const void* src, void* dst, const std::vector<int64_t>& shape,
                  const std::vector<int64_t>& perm, DataType dtype) {
  switch (dtype) {
    case DataType::TYPE_FP16:
      return xdnn::transpose<::float16>(ctx, reinterpret_cast<const ::float16*>(src), reinterpret_cast<::float16*>(dst),
                                        shape, perm);
    case DataType::TYPE_BF16:
      return xdnn::transpose<::bfloat16>(ctx, reinterpret_cast<const ::bfloat16*>(src),
                                         reinterpret_cast<::bfloat16*>(dst), shape, perm);
    case DataType::TYPE_FP32:
      return xdnn::transpose<float>(ctx, reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst), shape,
                                    perm);
    case DataType::TYPE_INT32:
      return xdnn::transpose<int>(ctx, reinterpret_cast<const int*>(src), reinterpret_cast<int*>(dst), shape, perm);
    case DataType::TYPE_INT8:
      return xdnn::transpose<signed char>(ctx, reinterpret_cast<const signed char*>(src),
                                          reinterpret_cast<signed char*>(dst), shape, perm);
    default:
      return -1;
  }
}
}  // namespace

Status Permute(Tensor& input_tensor, Tensor& output_tensor, const std::vector<size_t>& permutation, Stream& stream,
               void* device_ctx, void* workspace_ptr) {
  KLLM_CHECK_WITH_INFO(device_ctx != nullptr, "Permute on Kunlun requires device_ctx (xdnn::Context*)");

  std::vector<int64_t> xshape(input_tensor.shape.begin(), input_tensor.shape.end());
  std::vector<int64_t> xperm(permutation.begin(), permutation.end());

  xdnn::Context* ctx = static_cast<xdnn::Context*>(device_ctx);
  int ret =
      XdnnTranspose(ctx, input_tensor.GetPtr<void>(), output_tensor.GetPtr<void>(), xshape, xperm, input_tensor.dtype);
  if (ret != 0) {
    KLLM_LOG_ERROR << fmt::format("xdnn::transpose failed with error {}", ret);
    return Status(RET_INFER_FAILED, "xdnn::transpose error");
  }
  return Status();
}

}  // namespace ksana_llm
