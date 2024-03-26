/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/ret_code.h"

namespace ksana_llm {

static const char* GetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "UNKNOWN";
}

static const char* GetErrorString(cudaError_t error) { return cudaGetErrorString(error); }

template <typename T>
void CheckCUDAError(T result, const char* func, const char* file, const int line) {
  if (result) {
    NLLM_LOG_ERROR << fmt::format("CUDA runtime error: {} {}:{}@{}", GetErrorString(result), file, line, func);
    abort();
    exit(RetCode::RET_INVALID_ARGUMENT);
  }
}

#define CUDA_CHECK(val) CheckCUDAError((val), #val, __FILE__, __LINE__)

}  // namespace ksana_llm
