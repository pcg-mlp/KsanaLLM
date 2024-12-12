/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "tops/tops_runtime.h"

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/ret_code.h"

namespace ksana_llm {

template <typename T>
void CheckTopsError(T result, const char* func, const char* file, const int line) {
  if (result != topsSuccess) {
    KLLM_LOG_ERROR << fmt::format("Tops runtime error {}: {} {}:{}@{}", result, topsGetErrorString(result), file, line,
                                  func);
    abort();
    exit(RetCode::RET_INVALID_ARGUMENT);
  }
}

#define TOPS_CHECK(val) CheckTopsError((val), #val, __FILE__, __LINE__)

}  // namespace ksana_llm