/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <any>

#include "configure/context.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

Status Permute(Tensor& input_tensor, Tensor& output_tensor, const std::vector<size_t>& permutation, Stream& stream,
               void* device_ctx = nullptr, void* workspace_ptr = nullptr);

}  // namespace ksana_llm
