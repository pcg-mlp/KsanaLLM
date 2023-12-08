/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "src/numerous_llm/utils/nvidia/cuda_utils.h"

namespace numerous_llm {

int GetDeviceNumber() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

}  // namespace numerous_llm