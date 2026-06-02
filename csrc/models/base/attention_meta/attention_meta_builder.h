/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/context.h"
#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "device/device_utils.h"
#include "models/base/attention_meta/attention_meta.h"
#include "models/base/attention_meta/dsa_shared_attention_meta.h"
#include "models/base/attention_meta/mla_shared_attention_meta.h"

namespace ksana_llm {

struct StandardSharedAttentionMeta;  // forward declaration for SharedAttentionMeta::GetStandard()
struct MqaSharedAttentionMeta;       // forward declaration for SharedAttentionMeta::GetMqa()
struct IndexerSharedAttentionMeta;   // forward declaration for SharedAttentionMeta::GetIndexer()

struct AttentionBuilderConfig {
  const ModelConfig& model_config;
  const RuntimeConfig& runtime_config;
  bool enable_blocked_multi_token_forwarding_kv = false;
  bool use_flashinfer_for_decode = false;
  int rank = 0;
  int layer_num_on_node = 0;
  int block_size = 0;
  const std::vector<std::string>& kv_cache_types;
  const std::unordered_map<std::string, std::vector<KvCacheLayerOffsets>>& kv_cache_layer_offsets;
  size_t attn_dp_group_id = 0;
  int attn_dp_rank_id = 0;
  std::shared_ptr<Context> context;
  Event& kvcache_offset_event;
  Event& rotary_embedding_event;
};

// Empty base class for all shared attention meta objects.
//
// TODO(临时方案): GetStandard/GetMLA/GetDSA 是替代 AsStandardShared/AccessMLAShared/AccessDSAShared
// 的临时接口，避免后者对内存布局的强假设（UB 风险）。正式方案应在类型体系完成重构后统一清理。
// 调用方可直接解引用返回指针，无需判空（调用方明确自身使用的派生类型）。
struct SharedAttentionMeta {
  virtual ~SharedAttentionMeta() = default;

  virtual StandardSharedAttentionMeta* GetStandard() { return nullptr; }
  virtual const StandardSharedAttentionMeta* GetStandard() const { return nullptr; }
  virtual MLASharedAttentionMeta* GetMLA() { return nullptr; }
  virtual const MLASharedAttentionMeta* GetMLA() const { return nullptr; }
  virtual DSASharedAttentionMeta* GetDSA() { return nullptr; }
  virtual const DSASharedAttentionMeta* GetDSA() const { return nullptr; }
  virtual MqaSharedAttentionMeta* GetMqa() { return nullptr; }
  virtual const MqaSharedAttentionMeta* GetMqa() const { return nullptr; }
  virtual IndexerSharedAttentionMeta* GetIndexer() { return nullptr; }
  virtual const IndexerSharedAttentionMeta* GetIndexer() const { return nullptr; }
};

// Shared attention tensors and reusable host scratch buffers owned by ModelInput.
struct StandardSharedAttentionMeta : public SharedAttentionMeta {
  Tensor input_length;
  Tensor kv_list;
  Tensor kv_cache_offset;
  Tensor rotary_embedding_pos;
  Tensor rotary_embedding_mask;
  Tensor block_table;

  Tensor layer_kv_cache_ptr;
  bool layer_kv_cache_ptr_initialized = false;

  Tensor dp_flexible_rotary_embedding_mask;
  Tensor dp_src_flexible_rotary_embedding_pos;
  Tensor dp_dst_flexible_rotary_embedding_pos;
  Tensor dp_dst_flexible_kv_cache_tensor;
  Tensor dp_src_flexible_kv_cache_tensor;
  Tensor dp_dst_flexible_token_idx_tensor;
  Tensor dp_src_flexible_token_idx_tensor;
  Tensor dp_flexible_offset_uint64_tensor;

  std::vector<int> input_length_host;

  StandardSharedAttentionMeta* GetStandard() override { return this; }
  const StandardSharedAttentionMeta* GetStandard() const override { return this; }
};

namespace attention_meta_builder_detail {

void PrepareKVCacheBlocks(StandardAttentionMeta& meta, StandardSharedAttentionMeta& shared_meta,
                          const AttentionBuilderConfig& config);

// Prepares the block table and layer_kv_cache_ptr for flashinfer / blocked-multi-token paths.
// layer_kv_cache_ptr_layer_factor: number of layers encoded per block slot (use 1 for DSA, layer_num_on_node otherwise)
// layer_kv_cache_ptr_kv_factor: number of pointers per layer slot (1 for MLA, 2 for standard K+V)
void PrepareKVCacheBlockTable(StandardAttentionMeta& meta, StandardSharedAttentionMeta& shared_meta,
                              const AttentionBuilderConfig& config, int layer_kv_cache_ptr_layer_factor,
                              int layer_kv_cache_ptr_kv_factor);

}  // namespace attention_meta_builder_detail

}  // namespace ksana_llm
