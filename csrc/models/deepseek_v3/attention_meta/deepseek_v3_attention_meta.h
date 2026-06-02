/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/attention_meta/attention_meta.h"
#include "models/base/attention_meta/attention_meta_builder.h"
#include "models/base/attention_meta/indexer_attention_meta.h"
#include "models/base/attention_meta/indexer_shared_attention_meta.h"

namespace ksana_llm {

// DeepSeekV3-specific shared attention tensors (standard fields + MLA/DSA/Indexer extensions).
struct DeepSeekV3SharedAttentionMeta : public StandardSharedAttentionMeta {
  MLASharedAttentionMeta mla;
  DSASharedAttentionMeta dsa;
  IndexerSharedAttentionMeta indexer;

  // TODO(临时方案): 实现 SharedAttentionMeta 的临时 getter 接口，见基类注释。
  MLASharedAttentionMeta* GetMLA() override { return &mla; }
  const MLASharedAttentionMeta* GetMLA() const override { return &mla; }
  DSASharedAttentionMeta* GetDSA() override { return &dsa; }
  const DSASharedAttentionMeta* GetDSA() const override { return &dsa; }
  IndexerSharedAttentionMeta* GetIndexer() override { return &indexer; }
  const IndexerSharedAttentionMeta* GetIndexer() const override { return &indexer; }
};

struct DeepSeekV3FlashAttentionMeta : public FlashAttentionMeta {
  MLAAttentionMeta mla;
  DSAAttentionMeta dsa;
  IndexerAttentionMeta indexer;

  // TODO(临时方案): 实现 AttentionMeta 的临时 getter 接口，见基类注释。
  MLAAttentionMeta* GetMLA() override { return &mla; }
  const MLAAttentionMeta* GetMLA() const override { return &mla; }
  DSAAttentionMeta* GetDSA() override { return &dsa; }
  const DSAAttentionMeta* GetDSA() const override { return &dsa; }
  IndexerAttentionMeta* GetIndexer() override { return &indexer; }
  const IndexerAttentionMeta* GetIndexer() const override { return &indexer; }
};

struct DeepSeekV3PagedAttentionMeta : public PagedAttentionMeta {
  MLAAttentionMeta mla;
  DSAAttentionMeta dsa;
  IndexerAttentionMeta indexer;

  // TODO(临时方案): 实现 AttentionMeta 的临时 getter 接口，见基类注释。
  MLAAttentionMeta* GetMLA() override { return &mla; }
  const MLAAttentionMeta* GetMLA() const override { return &mla; }
  DSAAttentionMeta* GetDSA() override { return &dsa; }
  const DSAAttentionMeta* GetDSA() const override { return &dsa; }
  IndexerAttentionMeta* GetIndexer() override { return &indexer; }
  const IndexerAttentionMeta* GetIndexer() const override { return &indexer; }
};

}  // namespace ksana_llm
