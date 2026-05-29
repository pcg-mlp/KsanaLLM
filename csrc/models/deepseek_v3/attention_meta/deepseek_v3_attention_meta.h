/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/attention_meta/attention_meta.h"
#include "models/base/attention_meta/attention_meta_builder.h"

namespace ksana_llm {

// DeepSeekV3-specific shared attention tensors (standard fields + MLA/DSA extensions).
struct DeepSeekV3SharedAttentionMeta : public StandardSharedAttentionMeta {
  MLASharedAttentionMeta mla;
  DSASharedAttentionMeta dsa;

  // TODO(临时方案): 实现 SharedAttentionMeta 的临时 getter 接口，见基类注释。
  MLASharedAttentionMeta* GetMLA() override { return &mla; }
  const MLASharedAttentionMeta* GetMLA() const override { return &mla; }
  DSASharedAttentionMeta* GetDSA() override { return &dsa; }
  const DSASharedAttentionMeta* GetDSA() const override { return &dsa; }
};

struct DeepSeekV3FlashAttentionMeta : public FlashAttentionMeta {
  MLAAttentionMeta mla;
  DSAAttentionMeta dsa;

  // TODO(临时方案): 实现 AttentionMeta 的临时 getter 接口，见基类注释。
  MLAAttentionMeta* GetMLA() override { return &mla; }
  const MLAAttentionMeta* GetMLA() const override { return &mla; }
  DSAAttentionMeta* GetDSA() override { return &dsa; }
  const DSAAttentionMeta* GetDSA() const override { return &dsa; }
};

struct DeepSeekV3PagedAttentionMeta : public PagedAttentionMeta {
  MLAAttentionMeta mla;
  DSAAttentionMeta dsa;

  // TODO(临时方案): 实现 AttentionMeta 的临时 getter 接口，见基类注释。
  MLAAttentionMeta* GetMLA() override { return &mla; }
  const MLAAttentionMeta* GetMLA() const override { return &mla; }
  DSAAttentionMeta* GetDSA() override { return &dsa; }
  const DSAAttentionMeta* GetDSA() const override { return &dsa; }
};

}  // namespace ksana_llm
