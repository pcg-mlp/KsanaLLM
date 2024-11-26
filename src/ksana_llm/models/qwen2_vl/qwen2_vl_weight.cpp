/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/qwen2_vl/qwen2_vl_weight.h"

namespace ksana_llm {

template <typename T>
Qwen2VLWeight<T>::Qwen2VLWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, rank, context) {}

template class Qwen2VLWeight<float>;
template class Qwen2VLWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class Qwen2VLWeight<bfloat16>;
#endif

}  // namespace ksana_llm
