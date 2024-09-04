// Copyright 2024 Tencent Inc.  All rights reserved.

#include "paged_attention_dtypes.h"
#include "paged_attention.h"
#include "paged_attention.cu"
// TODO(zhongzhicao): The include of ".cu" should be removed in the future.
// To explicitly instantiate a template class without including a .cu file, the complete definitions of the class
// member functions must be in the .h file. However, CUDA kernel functions must be defined in a .cu file to be compiled
// correctly, thus creating a conflict.

namespace llm_kernels {
namespace nvidia {

template class PagedAttentionCuda<uint16_t, uint16_t, llm_kernels::utils::KVCacheType::kAuto>;

}  // namespace nvidia
}  // namespace llm_kernels
