/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "src/ksana_llm/utils/nvidia/cuda_utils.h"

#include <iostream>

namespace ksana_llm {

int GetDeviceNumber() {
  int device = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device));
  return device;
}

}  // namespace ksana_llm