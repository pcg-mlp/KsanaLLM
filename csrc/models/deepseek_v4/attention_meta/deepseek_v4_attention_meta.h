/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <queue>
#include <unordered_map>
#include <vector>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/model_config.h"
#include "models/base/attention_meta/attention_meta.h"
#include "models/base/attention_meta/attention_meta_builder.h"
#include "models/base/attention_meta/c4_indexer_attention_meta.h"
#include "models/base/attention_meta/c4_indexer_shared_attention_meta.h"
#include "runtime/forward_request.h"

namespace ksana_llm {

// Compressor-specific shared tensors and persistent per-request state.
// Pre-allocated once at model startup; shared tensors serve as the backing store
// from which per-batch CompressorAttentionMeta tensor views are sliced.
struct CompressorSharedAttentionMeta {
  // Pre-allocated shared backing tensors (per-batch views are sliced from these)
  Tensor c4_indices_tensor;
  Tensor c128_indices_tensor;
  Tensor c4_indices_bundle_tensor;
  Tensor c128_load_indices_tensor;
  Tensor c4_indices_prev_tensor;
  Tensor seq_lens_tensor;
  Tensor extend_lens_tensor;
  Tensor c4_out_loc_tensor;
  Tensor c128_out_loc_tensor;

  // Prefill plan tensors (shape[2] used as init flag by compressor kernel)
  Tensor c4_compress_plan_tensor;
  Tensor c4_write_plan_tensor;
  Tensor c128_compress_plan_tensor;
  Tensor c128_write_plan_tensor;

  // Per-layer state base pointers (HOST, INT64, [layer_num_on_node])
  Tensor layer_c4_state_ptr_tensor;
  Tensor layer_c4_indexer_state_ptr_tensor;
  Tensor layer_c128_state_ptr_tensor;
  bool layer_state_ptr_initialized = false;

  // Prefix cache tail indices (allocated only when enable_prefix_caching is true)
  Tensor tail_read_block_tensor;
  Tensor tail_target_indices_tensor;
  Tensor tail_write_indices_tensor;

  // State buffers (DEVICE, BYTES, [max_batch_size, slot_num, block_cache_size])
  Tensor c4_state_buffer;
  Tensor c4_indexer_state_buffer;
  Tensor c128_state_buffer;

  // Persistent cross-batch slot management
  std::unordered_map<int64_t, int> req_to_slot;
  std::queue<int> free_slot;

  // Configuration fields (set by Init, used by PrepareCompressor*)
  int c4_slot_num = 2;
  int c128_slot_num = 1;
  int c4_layer_num_on_node = 0;
  int c128_layer_num_on_node = 0;
  int layer_num_on_node = 0;
  int layer_start_on_node = 0;
  int block_token_num = 0;
  uint32_t attn_dp_group_id = 0;
  uint32_t attn_dp_rank_id = 0;

  // Allocates all tensors and initialises slot management.
  // No-op when use_dsv4_attn is false.
  void Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int layer_num_on_node,
            int layer_start_on_node, int rank, std::shared_ptr<BaseModelCacheLayout> block_cache_layout);
};

// MQA-specific shared tensors and persistent cross-batch state.
// Pre-allocated once at model startup; shared tensors serve as the backing store
// from which per-batch MqaAttentionMeta tensor views are sliced.
struct MqaSharedAttentionMeta {
  // Kernel-defined layout constants (tightly coupled to flash_sparse_mla kernel)
  static constexpr size_t kSwaSize = 128;
  static constexpr size_t kC4Ratio = 4;
  static constexpr size_t kC128Ratio = 128;
  static constexpr size_t kTopkBlockSize = 64;  // FlashMLA kernel performs block-style index reads
  static_assert(kSwaSize % kTopkBlockSize == 0, "kSwaSize must be a multiple of kTopkBlockSize");

  // Pre-allocated backing tensors (per-batch views are sliced from these)
  Tensor raw_out_loc_tensor;         // DEVICE, INT32, [max_token_num]
  Tensor swa_block_indices_tensor;   // DEVICE, INT32, [max_token_num, kSwaSize]
  Tensor c128_block_indices_tensor;  // DEVICE, INT32, [max_token_num, aligned_max_c128_seq_len]
  Tensor c128_topk_lengths_tensor;   // DEVICE, INT32, [max_batch_size]

  // tile_scheduler_metadatas (one entry per q_seq_len [1..threshold], plus one extra for flash prefill)
  std::vector<Tensor> c4_tile_scheduler_metadatas;
  std::vector<Tensor> c128_tile_scheduler_metadatas;
  Tensor c4_num_splits_tensor;    // DEVICE, INT32, [max_batch_size + 1 + decode_token_num_threshold]
  Tensor c128_num_splits_tensor;  // DEVICE, INT32, [max_batch_size + 1 + decode_token_num_threshold]

  // Shared tensors accessed directly by attention modules
  // layer_swa_kv_cache_ptr_tensor layout:
  //   slot 0       : device_block_num (int64)
  //   slot 1       : swa_block_size   (int64)
  //   slot 2..2+N-1: per-layer SWA base ptr (void*)
  Tensor layer_swa_kv_cache_ptr_tensor;  // HOST, INT64, [2 + layer_num_on_node]
  Tensor c4_block_indices_tensor;        // DEVICE, INT32, [max_token_num, index_topk] — written by C4 Indexer
  // tail tensor layout:
  //   slot 0     : device_block_num (int64)
  //   slot 1..N-1: per-layer tail base ptr (void*)
  Tensor layer_tail_attn_cache_ptr_tensor;     // HOST, INT64, [1 + layer_num_on_node]
  Tensor layer_tail_indexer_cache_ptr_tensor;  // HOST, INT64, [1 + layer_num_on_node]

  // SWA cache buffer for non-contiguous SWA block support.
  // Provides a contiguous block space that mirrors the pool layout, used when
  // any flash request has non-contiguous swa_kv_cache_block_idx_offsets.
  Tensor swa_cache_buffer;                   // DEVICE, UINT8, [swa_buffer_num_blocks * per_layer_swa_block_size]
  Tensor layer_swa_cache_buffer_ptr_tensor;  // HOST, INT64, [2 + layer_num_on_node] (same layout as
                                             //   layer_swa_kv_cache_ptr_tensor but pointing into buffer)
  int swa_buffer_num_blocks = 0;
  int per_layer_swa_block_size = 0;  // = swa_block_size / layer_num_on_node; stride per buffer slot

  // Backing device tensors for BatchMemoryD2D addr lists (per-batch views sliced from these).
  Tensor swa_pre_copy_src_tensor;  // DEVICE, UINT64, [swa_buffer_num_blocks]
  Tensor swa_pre_copy_dst_tensor;
  Tensor swa_post_copy_src_tensor;
  Tensor swa_post_copy_dst_tensor;

  // Scratch tensors for the per-meta InvokePrepareDsv4MqaMeta GPU kernel.
  Tensor cu_extend_lens_tensor;                 // DEVICE, INT32, [max_batch_size + 1]
  Tensor cu_prefix_lens_tensor;                 // DEVICE, INT32, [max_batch_size + 1]
  Tensor swa_prefix_tail_block_indices_tensor;  // DEVICE, INT32, [max_batch_size, kSwaSize]
  std::vector<int> cu_extend_lens_host;
  std::vector<int> cu_prefix_lens_host;
  std::vector<int> swa_prefix_tail_block_indices_host;

  // Configuration (set by Init, read-only during Prepare*)
  int layer_num_on_node = 0;
  int block_size = 0;
  int swa_block_size = 0;
  int block_token_num = 0;
  int index_topk = 0;
  uint32_t attn_dp_group_id = 0;
  uint32_t attn_dp_rank_id = 0;

  // Persistent cross-batch state: SWA / tail base addresses are resolved on first Prepare
  bool layer_kv_cache_ptr_initialized = false;

  // Allocates all tensors and initialises configuration. No-op when use_dsv4_attn is false.
  void Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int layer_num_on_node, int rank,
            std::shared_ptr<BaseModelCacheLayout> block_cache_layout);
};

// Compressor-specific per-batch attention input tensors.
// Each tensor is a view sliced from CompressorSharedAttentionMeta's backing tensors.
struct CompressorAttentionMeta {
  // Requests handled by this path (flash or paged) for this dp group.
  // Populated by PrepareCompressor(); uses its own IsPaged routing, which differs from
  // FlashAttentionMeta::dp_reqs (standard kFlash routing). MTP decode requests
  // (q_seq_len > 1, kv_cached > 0) belong to flash_compressor but are kPage in standard routing.
  std::vector<const ForwardRequest*> dp_reqs;

  // Total input token count for this compressor path, i.e. sum of GetInputIdsLength()
  // over dp_reqs. Populated by PrepareCompressor(); used by compressor.cpp as the kv_score
  // slice boundary instead of model_input->dp_context_tokens / dp_decode_tokens.
  size_t dp_tokens = 0;

  // Tensor views of the shared backing tensors
  Tensor c4_indices_tensor;
  Tensor c128_indices_tensor;
  Tensor c4_indices_bundle_tensor;
  Tensor c128_load_indices_tensor;
  Tensor c4_indices_prev_tensor;
  Tensor seq_lens_tensor;
  Tensor extend_lens_tensor;
  Tensor c4_out_loc_tensor;
  Tensor c128_out_loc_tensor;

  // Persistent host staging vectors (must outlive async H2D memcpy)
  std::vector<int> c4_indices_host;
  std::vector<int> c128_indices_host;
  std::vector<int> c4_indices_bundle_host;
  std::vector<int> c128_load_indices_host;
  std::vector<int> c4_indices_prev_host;
  std::vector<int> seq_lens_host;
  std::vector<int> extend_lens_host;
  std::vector<int> c4_out_loc_host;
  std::vector<int> c128_out_loc_host;

  void Reset() {
    dp_reqs.clear();
    dp_tokens = 0;
    c4_indices_host.clear();
    c128_indices_host.clear();
    c4_indices_bundle_host.clear();
    c128_load_indices_host.clear();
    c4_indices_prev_host.clear();
    seq_lens_host.clear();
    extend_lens_host.clear();
    c4_out_loc_host.clear();
    c128_out_loc_host.clear();
  }
};

// MQA-specific per-batch attention input tensors.
// Each tensor is a view sliced from MqaSharedAttentionMeta's backing tensors.
// dp_reqs / q_seq_len / total_dp_input_ids_len are inherited from AttentionMeta (the parent
// FlashAttentionMeta / PagedAttentionMeta) and are NOT duplicated here.
struct MqaAttentionMeta {
  // Tensor views of the shared backing tensors
  Tensor raw_out_loc_tensor;
  Tensor swa_block_indices_tensor;
  Tensor c128_block_indices_tensor;
  Tensor c128_topk_lengths_tensor;

  // tile_scheduler_metadata & num_splits sliced from shared pool per batch
  Tensor c4_tile_scheduler_metadata;
  Tensor c128_tile_scheduler_metadata;
  Tensor c4_num_splits;
  Tensor c128_num_splits;

  // Persistent host staging vectors (must outlive async H2D memcpy)
  std::vector<int> raw_out_loc_host;

  // Non-contiguous SWA cache buffer support.
  // True when any request in this flash batch has non-contiguous swa_kv_cache_block_idx_offsets.
  bool has_non_contiguous_swa = false;
  // Device tensor views (sliced from MqaSharedAttentionMeta backing tensors) for BatchMemoryD2D.
  Tensor swa_pre_copy_src_tensor;  // pool blocks → buffer (historical SWA window)
  Tensor swa_pre_copy_dst_tensor;
  Tensor swa_post_copy_src_tensor;  // buffer → pool blocks (new tokens written this step)
  Tensor swa_post_copy_dst_tensor;
  // Host staging for the addr lists above (must outlive H2D memcpy in PrepareMqaFlash).
  std::vector<uint64_t> swa_pre_copy_src_host;
  std::vector<uint64_t> swa_pre_copy_dst_host;
  std::vector<uint64_t> swa_post_copy_src_host;
  std::vector<uint64_t> swa_post_copy_dst_host;

  void Reset() {
    raw_out_loc_host.clear();
    has_non_contiguous_swa = false;
    swa_pre_copy_src_host.clear();
    swa_pre_copy_dst_host.clear();
    swa_post_copy_src_host.clear();
    swa_post_copy_dst_host.clear();
  }
};

// DeepSeekV4-specific shared attention tensors (standard + MLA/DSA/Compressor/MQA/C4Indexer).
struct DeepSeekV4SharedAttentionMeta : public StandardSharedAttentionMeta {
  MLASharedAttentionMeta mla;
  DSASharedAttentionMeta dsa;
  CompressorSharedAttentionMeta compressor;
  MqaSharedAttentionMeta mqa;
  C4IndexerSharedAttentionMeta c4_indexer;

  // TODO(临时方案): 实现 SharedAttentionMeta 的临时 getter 接口，见基类注释。
  MLASharedAttentionMeta* GetMLA() override { return &mla; }
  const MLASharedAttentionMeta* GetMLA() const override { return &mla; }
  DSASharedAttentionMeta* GetDSA() override { return &dsa; }
  const DSASharedAttentionMeta* GetDSA() const override { return &dsa; }
  MqaSharedAttentionMeta* GetMqa() override { return &mqa; }
  const MqaSharedAttentionMeta* GetMqa() const override { return &mqa; }
  IndexerSharedAttentionMeta* GetIndexer() override { return &c4_indexer; }
  const IndexerSharedAttentionMeta* GetIndexer() const override { return &c4_indexer; }
};

// DeepSeekV4-specific flash (prefill) attention input.
struct DeepSeekV4FlashAttentionMeta : public FlashAttentionMeta {
  MLAAttentionMeta mla;
  DSAAttentionMeta dsa;
  CompressorAttentionMeta compressor;
  MqaAttentionMeta mqa;
  C4IndexerAttentionMeta c4_indexer;

  // TODO(临时方案): 实现 AttentionMeta 的临时 getter 接口，见基类注释。
  MLAAttentionMeta* GetMLA() override { return &mla; }
  const MLAAttentionMeta* GetMLA() const override { return &mla; }
  DSAAttentionMeta* GetDSA() override { return &dsa; }
  const DSAAttentionMeta* GetDSA() const override { return &dsa; }
  IndexerAttentionMeta* GetIndexer() override { return &c4_indexer; }
  const IndexerAttentionMeta* GetIndexer() const override { return &c4_indexer; }
};

// DeepSeekV4-specific paged (decode) attention input.
struct DeepSeekV4PagedAttentionMeta : public PagedAttentionMeta {
  MLAAttentionMeta mla;
  DSAAttentionMeta dsa;
  CompressorAttentionMeta compressor;
  MqaAttentionMeta mqa;
  C4IndexerAttentionMeta c4_indexer;

  // TODO(临时方案): 实现 AttentionMeta 的临时 getter 接口，见基类注释。
  MLAAttentionMeta* GetMLA() override { return &mla; }
  const MLAAttentionMeta* GetMLA() const override { return &mla; }
  DSAAttentionMeta* GetDSA() override { return &dsa; }
  const DSAAttentionMeta* GetDSA() const override { return &dsa; }
  IndexerAttentionMeta* GetIndexer() override { return &c4_indexer; }
  const IndexerAttentionMeta* GetIndexer() const override { return &c4_indexer; }
};

}  // namespace ksana_llm
