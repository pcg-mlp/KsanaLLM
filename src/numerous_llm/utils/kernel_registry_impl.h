/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "numerous_llm/utils/kernel_registry.h"

#include "3rdparty/LLM_kernels/csrc/nvidia/embedding/embedding.h"

// Register all the kernels here:
//
// Step 1: Include the header:
// #include "xxx.h"
//
// Step 2: register kernel func:
// REGISTER_NVIDIA_KERNEL(a, kernel_a);
// REGISTER_NVIDIA_KERNEL(a_t, kernel_a_template<int>);
//

namespace numerous_llm {
  REGISTER_NVIDIA_KERNEL(LookupFusedEmbedding, LookupFusedEmbedding<half>);
}  // namespace numerous_llm
