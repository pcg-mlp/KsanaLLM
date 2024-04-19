/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

namespace llm_kernels {
namespace nvidia {

void GetAlibiSlopesCuda(float* slopes,  // [total_num_heads]
                        int total_num_heads, cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
