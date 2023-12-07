/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <stdexcept>
#include <string>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

static const char* GetErrorString(cudaError_t error) { return cudaGetErrorString(error); }

template <typename T>
void CheckCUDAError(T result, const char* func, const char* file, const int line) {
  if (result) {
    NLLM_LOG_ERROR << fmt::format("CUDA runtime error: {} {}:{}@{}", GetErrorString(result), file, line, func);
    exit(RetCode::RET_INVALID_ARGUMENT);
  }
}

#define CUDA_CHECK(val) CheckCUDAError((val), #val, __FILE__, __LINE__)

}  // namespace numerous_llm