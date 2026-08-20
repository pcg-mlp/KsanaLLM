/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "kernels/index_select_1d.h"

#include "kernels/kunlun/basic_kernel_wrapper.h"
#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ksana_llm {

template <typename T, typename TID>
Status IndexSelect1D(const T* table, int64_t table_len, const TID* index, T* output, int64_t num_indices,
                     Stream& stream, void* device_ctx) {
  KLLM_CHECK_WITH_INFO(device_ctx != nullptr, "IndexSelect1D on Kunlun requires device_ctx (xpukernel::Context*)");

  auto* ctx = static_cast<xpytorch::xpu::api::Context*>(device_ctx);

  int ret = XpuIndexSelect<T, TID>(ctx, table, index, output, {table_len, 1}, num_indices, /*axis=*/0);
  if (ret != 0) {
    return Status(RET_RUNTIME_FAILED, FormatStr("XpuIndexSelect (1D) failed, ret=%d", ret));
  }

  return Status();
}

template Status IndexSelect1D<int32_t, int32_t>(const int32_t*, int64_t, const int32_t*, int32_t*, int64_t, Stream&,
                                                void*);
template Status IndexSelect1D<int32_t, int64_t>(const int32_t*, int64_t, const int64_t*, int32_t*, int64_t, Stream&,
                                                void*);
template Status IndexSelect1D<int64_t, int32_t>(const int64_t*, int64_t, const int32_t*, int64_t*, int64_t, Stream&,
                                                void*);
template Status IndexSelect1D<int64_t, int64_t>(const int64_t*, int64_t, const int64_t*, int64_t*, int64_t, Stream&,
                                                void*);

}  // namespace ksana_llm
