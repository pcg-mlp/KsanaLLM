/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/attention_meta/attention_meta_builder.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta.h"

namespace ksana_llm {

// Prepares per-batch MQA metadata for the flash (prefill) path.
// Uses flash_meta.dp_reqs (already categorised by the standard builder).
// Also handles the one-time SWA / tail base-pointer initialisation (once-flag in mqa_shared).
// No-op when flash_meta.dp_reqs is empty.
void PrepareMqaFlash(MqaAttentionMeta& mqa_meta, MqaSharedAttentionMeta& mqa_shared,
                     const FlashAttentionMeta& flash_meta, const AttentionBuilderConfig& config);

// Prepares per-batch MQA metadata for the paged (decode) path.
// Uses paged_meta.dp_reqs and paged_meta.q_seq_len.
// Also handles the one-time SWA / tail base-pointer initialisation if not yet done
// (e.g. pure-decode steps where PrepareMqaFlash was never called).
// No-op when paged_meta.dp_reqs is empty.
void PrepareMqaPaged(MqaAttentionMeta& mqa_meta, MqaSharedAttentionMeta& mqa_shared,
                     const PagedAttentionMeta& paged_meta, const AttentionBuilderConfig& config);

}  // namespace ksana_llm
