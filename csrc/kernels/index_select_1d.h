/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>

#include "device/device_utils.h"
#include "utils/status.h"

namespace ksana_llm {

// 1D index-select: output[i] = table[index[i]], i in [0, num_indices).
// T = data type (table/output), TID = index type.
// NOTE: output must NOT alias index (xpukernel::index_select does not support in-place).
template <typename T, typename TID = T>
Status IndexSelect1D(const T* table, int64_t table_len, const TID* index, T* output, int64_t num_indices,
                     Stream& stream, void* device_ctx = nullptr);

}  // namespace ksana_llm
