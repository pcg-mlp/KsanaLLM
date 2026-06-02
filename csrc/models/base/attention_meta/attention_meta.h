/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "runtime/forward_request.h"
#include "utils/tensor.h"

namespace ksana_llm {

struct MLAAttentionMeta;
struct DSAAttentionMeta;
struct IndexerAttentionMeta;  // forward declaration for AttentionMeta::GetIndexer()

// Minimal base class for all attention input types.
// Holds only the per-batch control fields that every attention path shares.
// Model-specific tensor fields (MLA / DSA) live in MLAAttentionMeta / DSAAttentionMeta
// on DeepSeek model derived types; use GetMLA / GetDSA to reach them.
//
// TODO(临时方案): GetMLA/GetDSA 是替代旧内存布局转换的临时接口，
// 避免后者对派生类型字段顺序的强假设。正式方案应在类型体系完成重构后统一清理。
// 调用方可直接解引用返回指针，无需判空（调用方明确自身使用的派生类型）。
struct AttentionMeta {
  AttentionMeta() = default;
  AttentionMeta(AttentionMeta&&) = default;
  AttentionMeta& operator=(AttentionMeta&&) = default;
  // Disable the time-consuming copy
  AttentionMeta(const AttentionMeta&) = delete;
  AttentionMeta& operator=(const AttentionMeta&) = delete;

  virtual ~AttentionMeta() = default;

  virtual MLAAttentionMeta* GetMLA() { return nullptr; }
  virtual const MLAAttentionMeta* GetMLA() const { return nullptr; }
  virtual DSAAttentionMeta* GetDSA() { return nullptr; }
  virtual const DSAAttentionMeta* GetDSA() const { return nullptr; }
  virtual IndexerAttentionMeta* GetIndexer() { return nullptr; }
  virtual const IndexerAttentionMeta* GetIndexer() const { return nullptr; }

  std::vector<ForwardRequest*> dp_reqs;

  size_t q_seq_len = 0;
  size_t expanded_batch_size = 1;
  size_t total_dp_input_ids_len = 0;

  void Reset() {
    dp_reqs.clear();
    q_seq_len = 0;
    expanded_batch_size = 1;
    total_dp_input_ids_len = 0;
  }
};

// MLA-specific per-batch attention input tensors.
struct MLAAttentionMeta {
  Tensor tile_scheduler_metadata;
  Tensor num_splits;
};

// DSA-specific per-batch attention input tensors.
struct DSAAttentionMeta {
  Tensor paged_schedule_meta;
  Tensor store_out_loc;
  Tensor expanded_block_table;
};

// Standard tensor fields shared by both flash (prefill) and paged (decode) paths.
struct StandardAttentionMeta : public AttentionMeta {
  // Tensors in input_meta are views of the shared pool tensors in ModelInput.
  // Their shape is used to locate the offset of each input_meta in the shared tensors.
  Tensor input_length;
  Tensor kv_list;
  Tensor kv_cache_offset;
  Tensor rotary_embedding_pos;
  Tensor rotary_embedding_mask;
  Tensor block_table;
};

// Attention input for the flash (prefill) path.
struct FlashAttentionMeta : public StandardAttentionMeta {};

// Attention input for the paged (decode) path.
struct PagedAttentionMeta : public StandardAttentionMeta {};

}  // namespace ksana_llm
