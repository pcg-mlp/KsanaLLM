/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <any>

#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

Status LookupEmbedding(const void* ids, const void* offset, const void* emb, const void* pos, void* output,
                       int vocab_size, int hidden_size, int bs, int step, int vocab_id, Stream& stream,
                       void* workspace_ptr = nullptr);

}  // namespace ksana_llm