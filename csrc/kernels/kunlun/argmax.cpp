/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernels/argmax.h"

#include "xpu/refactor/xpukernel_aten.h"

#include "kernels/kunlun/basic_kernel_wrapper.h"
#include "utils/logger.h"
#include "utils/status.h"
#include "utils/string_utils.h"

namespace ksana_llm {

namespace xpukernel = xpytorch::xpu::api;

// XSGLANG runs argmax via PyTorch's torch.argmax → torch_xmlir backend →
// xpytorch::xpu::api::argmax<TX, int64_t>, producing the SDNN device kernels
//   arg_max_min_mt<float, float, long, int, ...>
//   arg_max_min_cluster_reduce<float, long, int, ...>
// (864 calls in trace).  KSANA's caller-facing ArgMax interface returns
// uint32_t* (chosen for write-bandwidth savings and shared with NVIDIA / Ascend
// backends), so we run the device argmax with int64_t to match XSGLANG's exact
// kernel signature, then cast<int64_t,int32_t> the result down to the caller's
// 32-bit slot.  The cast lowers to `cluster_cast_kl3` — the same kernel
// XSGLANG uses for any int64→int32 narrowing, so total dispatched kernels stay
// aligned: argmax_mt(long) + cluster_cast_kl3.

template <typename T>
Status ArgMax(const T* input, const int32_t batch_size, const int32_t vocab_size, uint32_t* result, Stream& stream,
              void* device_ctx, void* buffer_ptr) {
  KLLM_CHECK_WITH_INFO(device_ctx != nullptr, "ArgMax on Kunlun requires device_ctx (xpukernel::Context*)");
  KLLM_CHECK_WITH_INFO(buffer_ptr != nullptr,
                       "ArgMax on Kunlun requires pre-allocated buffer_ptr (sizeof(int64_t)*batch_size)");

  auto* ctx = static_cast<xpukernel::Context*>(device_ctx);

  // Step 1: device-side int64 buffer for the int64-indexed argmax.
  // Pre-allocated by the caller (TopkSampling ctor), reused across decode steps.
  int64_t* idx64 = static_cast<int64_t*>(buffer_ptr);

  int ret = xpukernel::argmax(ctx, xcast(input), idx64,
                              {static_cast<int64_t>(batch_size), static_cast<int64_t>(vocab_size)}, /*axis=*/1);
  if (ret != 0) {
    return Status(RET_RUNTIME_FAILED, FormatStr("xpukernel argmax failed, ret=%d", ret));
  }

  // Step 2: narrow int64 → int32 to fit the caller-facing uint32_t* slot.
  // Lowered device kernel: cluster_cast_kl3 — same as XSGLANG for any int64→int32 narrow.
  ret = xpukernel::cast(ctx, idx64, reinterpret_cast<int32_t*>(result), static_cast<int64_t>(batch_size));
  if (ret != 0) {
    return Status(RET_RUNTIME_FAILED, FormatStr("xpukernel cast int64->int32 failed, ret=%d", ret));
  }

  return Status();
}

template Status ArgMax<float>(const float*, const int32_t, const int32_t, uint32_t*, Stream&, void*, void*);
template Status ArgMax<int16_t>(const int16_t*, const int32_t, const int32_t, uint32_t*, Stream&, void*, void*);
template Status ArgMax<uint16_t>(const uint16_t*, const int32_t, const int32_t, uint32_t*, Stream&, void*, void*);

}  // namespace ksana_llm
