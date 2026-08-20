/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <any>

#include "configure/context.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

template <typename T>
Status ArgMax(const T* input, const int32_t batch_size, const int32_t vocab_size, uint32_t* result, Stream& stream,
              void* device_ctx = nullptr, void* buffer_ptr = nullptr);

}  // namespace ksana_llm
