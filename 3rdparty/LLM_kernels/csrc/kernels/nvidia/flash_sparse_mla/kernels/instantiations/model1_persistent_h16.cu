/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#include "csrc/kernels/nvidia/flash_sparse_mla/kernels/fp8_splitkv_mla.cuh"

namespace llm_kernels {
namespace nvidia {

template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1, 16>(const SparseAttnDecodeParams &params);

}  // namespace nvidia
}  // namespace llm_kernels