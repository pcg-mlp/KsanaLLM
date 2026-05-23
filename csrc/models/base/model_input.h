/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/environment.h"
#include "models/base/base_model.h"
#include "models/base/forward_meta/c4_indexer_meta.h"
#include "models/base/forward_meta/compressor_meta.h"
#include "models/base/forward_meta/dsv4_mqa_meta.h"
#include "runtime/forward_request.h"
#include "runtime/infer_stage.h"
#include "utils/tensor.h"

namespace ksana_llm {

// Convert input ids to expected format.
class ModelInput {
 public:
  ModelInput(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
             std::shared_ptr<Context> context, std::shared_ptr<BaseModelCacheLayout> block_cache_layout);
  ~ModelInput();

  // Parse forward request.
  void ParseFromRequests(const std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode = RunMode::kMain);

 private:
  void PrepareInputRefit(const std::vector<ForwardRequest>& forward_reqs);
  void PrepareVLInputRefit(const std::vector<ForwardRequest>& forward_reqs);
  void CreateVLTensors();
  void PrepareVLRequest(const std::vector<ForwardRequest>& forward_reqs);
  void PrepareCutoffLayer(const std::vector<ForwardRequest>& forward_reqs);
  void PrepareNextNGatherIdx(const std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode);

  // Prepare MRope position for qwen2_vl
  void PrepareMRopePos(const std::vector<ForwardRequest>& forward_reqs);
  // Prepare XDRope position for arc_hunyuan_video
  void PrepareXDRopePos(const std::vector<ForwardRequest>& forward_reqs);
  // Prepare DeepStack injection data for qwen3_vl
  void PrepareDeepstack(const std::vector<ForwardRequest>& forward_reqs);

#ifdef ENABLE_CUDA
  template <typename T>
  void PrepareImgMask(size_t pos_num);

  void PrepareCudagraphParams(const std::vector<ForwardRequest>& forward_reqs);
#endif

#ifdef ENABLE_ACL
  void PrepareATBKVCache(const std::vector<ForwardRequest>& forward_reqs, bool is_multi_token_forward);
#endif

 public:
  // The input batch size.
  size_t batch_size = 0;
  size_t dp_batch_size = 0;

  // Number of dp tokens in context/decode phase
  size_t dp_context_tokens = 0;
  size_t dp_decode_tokens = 0;

  // Number of kv cache blocks in context/decode phase
  size_t context_kv_cache_block_num = 0;
  size_t decode_kv_cache_block_num = 0;

  // Number of requests who are forwarding multi-tokens in this step.
  size_t multi_token_request_num = 0;
  size_t dp_multi_token_request_num = 0;

  // Number of requests who are forwarding single-token in this step.
  size_t single_token_request_num = 0;
  size_t dp_single_token_request_num = 0;

  // The max tokens.
  size_t multi_token_request_max_tokens = 0;
  size_t single_token_request_max_tokens = 0;
  size_t dp_multi_token_request_max_tokens = 0;
  size_t dp_single_token_request_max_tokens = 0;

  // The total dp prefix length.
  size_t dp_total_prefix_len = 0;

  // current request batchsize matches cudagraph catpure range
  bool is_cudagraph_batchsize_matched = false;

  // For cutoff layer
  int cutoff_layer = 0;

  // Whether to use kv cache.
  bool use_cache = true;

  std::vector<size_t> dp_input_offset_list_uint64;
  std::vector<size_t> dp_input_prefix_list_uint64;
  std::vector<size_t> input_offset_list_uint64;
  std::vector<size_t> input_prefix_list_uint64;

  std::vector<int> input_ids_cpu;

  // The infer stage, context decode or decode.
  InferStage infer_stage;

  // The input ids, int32
  Tensor input_ids;

  // The ids offset tensor, uint64
  Tensor input_offset_uint64_tensor;
  Tensor dp_input_offset_uint64_tensor;
  Tensor dp_input_offset_int32_tensor;

  // The input's prefix length
  Tensor input_prefix_uint64_tensor;
  Tensor dp_input_prefix_uint64_tensor;

  Tensor dp_prefill_q_offset_uint64_tensor;
  Tensor dp_prefill_q_offset_int32_tensor;

  // Due to the optimization of PrefixCaching for computation reuse, incorporating the effects of flexible caching, a
  // mask is used during the flexible rotary_embedding computation to avoid multiple executions of flexible
  // rotary_embedding on the prefix block.
  Tensor dp_flexible_rotary_embedding_mask;

  // Indicate the corresponding index position of the input during the flexible rotary_embedding kernel computation,
  // considering the impact of flexible cache optimization.
  // Stores pos information of src, used for all models to reverse rope
  Tensor dp_src_flexible_rotary_embedding_pos;
  // Stores pos information of dst, used for deepseek models to embed correct rope
  Tensor dp_dst_flexible_rotary_embedding_pos;

  // The 3-dimentional index position for multimodal rotarty embedding.
  Tensor dp_mrotary_embedding_pos;
  // The 4-dimentional index position for xd rotarty embedding.
  Tensor dp_xdrotary_embedding_pos;

  // Record which logits in the output of all tokens need to be extracted for subsequent sampling calculations
  // Due to the presence of logits_custom_length and speculative_decoding, a single request may require extracting more
  // than one logit. In the standard case, only the last logit of each request needs to be retrieved
  Tensor logits_idx_uint64_tensor;

  Tensor nextn_hidden_idx_uint64_tensor;

  Tensor dp_dst_flexible_kv_cache_tensor;
  Tensor dp_src_flexible_kv_cache_tensor;
  Tensor dp_dst_flexible_token_idx_tensor;
  Tensor dp_src_flexible_token_idx_tensor;
  Tensor dp_flexible_offset_uint64_tensor;

  // Tensors to hold pairs(pos, data_length) and embeddings ptr of positions for input_refit on the CPU.
  struct {
    Tensor pos_tensor;
    std::vector<Tensor> emb_tensors;
  } cpu_input_refit_tensor;

  // DeepStack injection data for Qwen3-VL prefill.
  // GPU buffers are pre-allocated at construction to avoid runtime OOM.
  struct DeepstackData {
    bool active = false;
    int num_layers = 0;
    // Per-segment descriptor for scatter-add.
    struct Segment {
      int batch_pos;     // absolute token position in the batched residual buffer
      int embed_offset;  // row offset in the per-layer embed tensor
      int length;        // number of visual tokens in this segment
    };
    std::vector<Segment> segments;
    // Pre-allocated GPU embed tensors, one per deepstack layer.
    // Shape: [max_step_token_num, hidden_size]. Actual valid rows per batch <= max_step_token_num.
    std::vector<Tensor> layer_embeds;
    // Maps text decoder layer index to deepstack embed index for O(1) lookup during Forward.
    std::unordered_map<int, int> layer_to_deepstack_idx;

    // Reset per-batch metadata. Preserves pre-allocated layer_embeds and num_layers.
    void Reset() {
      active = false;
      segments.clear();
      layer_to_deepstack_idx.clear();
    }
  } deepstack_data;

  // IXC model use PLoRA
  bool is_mask = false;
  Tensor im_mask;

  Event kvcache_offset_event;
  Event rotary_embedding_event;
  Event input_ids_event;

#ifdef ENABLE_ACL
  // record all reqs token number on host, shape: [batch_size]
  Tensor seq_len_host;
  // Tensor to save kv cache base. detail doc please refer:
  // docs/Technology/kvcache-relationship-between-ascend-atb-and-ksana.md shape: [total_k/v_blocks, block_token_num,
  // kv_head_num, head_dim]
  Tensor k_cache_blocks_base;
  Tensor v_cache_blocks_base;

  // for multi-token forwarding: layers_slot_mapping shape is [num_layers, all_reqs_tokens_num]
  // for single-token forwarding: layers_block_table shape is [num_layers, batch_size]
  std::vector<int32_t> layers_slot_mapping_host;
  Tensor layers_slot_mapping;

  // only used for single-token forwarding: layers_block_table shape is [num_layers, batch_size *
  // max_num_blocks_per_query]
  std::vector<int32_t> layers_block_table_host;
  Tensor layers_block_table;

  // since layer's forward only support Tensor as input (nothing to do with karlluo), such crappy design ignore runtime
  // attribute, so we need a tensor to be attribute.
  // shape: [2]; 0: layers_slot_mapping_dim_1; 1: max_num_blocks_per_query
  Tensor atb_attention_attr;

  // assemble last token index for gather, dtype is int64_t
  Tensor last_token_index_tensor;

  std::vector<void*> kv_cache_ptrs;
  Tensor kv_cache_ptrs_tensor;
#endif
  // Record the number of kv cache blocks and the addresses of kv cache lists for each layer on the host
  Tensor layer_kv_cache_ptr;          // for core attention
  Tensor layer_indexer_kv_cache_ptr;  // for optional indexer

  size_t dp_max_forwarding_tokens = 0;

  // current rank related attention data para group id
  // NOTE(karlluo): for example: machine has 4 GPUs, Attention Data Parallelism is 2, Tensor Parallelism is 2.
  // |----Attn DP Group id 0----|----Attn DP Group id 1----|
  // |     TP 0   |     TP1     |     TP0    |     TP1     |
  // |     GPU0   |     GPU1    |     GPU2   |     GPU3    |
  size_t attn_dp_group_id_ = 0;
  int attn_dp_rank_id_ = 0;

  size_t attn_dp_group_size_;

  // The starting offset of tokens handled by each DP group,
  // in format [group0_offset, group1_offset, ...]
  std::vector<int> attn_dp_group_offsets_;

 private:
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  ConnectorConfig connector_config_;

  bool enable_blocked_multi_token_forwarding_kv_;
  bool use_flashinfer_for_decode_;

  const int rank_;
  std::shared_ptr<Context> context_;

  // The cache layout of every kv cache block.
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;

  // Active kv cache types for the current model ("attention" / optionally "indexer").
  // The model layout is fixed at startup, so this list is computed once after construction.
  std::vector<std::string> kv_cache_types_;

  // Per-(kv_cache_type, layer_index) offsets, computed once after construction.
  // Outer key is the kv_cache_type; inner index is the layer index.
  std::unordered_map<std::string, std::vector<KvCacheLayerOffsets>> kv_cache_layer_offsets_;

  // Total bytes occupied by one kv cache block
  int block_size_;
  int layer_num_on_node_;
  size_t total_sampling_token_num_;

  // for nextn layer(MTP), record each req's first token index in hidden output
  std::unordered_map<size_t, size_t> mtp_req_id_to_pos_;

  // `layer_kv_cache_ptr`/`layer_indexer_kv_cache_ptr` only needs to be initialized once
  bool layer_kv_cache_ptr_initialized_ = false;

  // Tensors shared by all input_infos
  Tensor input_length;           // only for page, forwarding_tokens.size()
  Tensor expanded_input_length;  // input_length after expansion of q_seq_len
  Tensor kv_list;                // for core attention
  Tensor indexer_kv_list;        // for optional indexer
  Tensor kv_cache_offset;
  Tensor rotary_embedding_pos;
  Tensor rotary_embedding_mask;
  Tensor block_table;
  Tensor expanded_block_table;                   // block_table after expansion of q_seq_len
  std::vector<Tensor> tile_scheduler_metadatas;  // only for flash mla of page
  Tensor num_splits;                             // only for flash mla of page
  Tensor cur_seq_len_start;                      // 每个token可以看到的开始位置的token索引
  Tensor cur_seq_len_end;                        // 每个token可以看到的结束位置的token索引
  std::vector<Tensor> paged_schedule_metas;      // for paged sparse mla indexer
  Tensor store_out_loc;                          // used to store indexer kv cache

  // Reused vectors during model_input preparation,
  // avoid destruction during MemcpyAsync
  std::vector<int> input_length_host_;
  std::vector<int> expanded_input_length_host_;

 public:
  struct input_info {
    input_info() = default;
    input_info(input_info&&) = default;
    input_info& operator=(input_info&&) = default;
    // Disable the time-consuming copy
    input_info(const input_info&) = delete;
    input_info& operator=(const input_info&) = delete;

    std::vector<ForwardRequest*> dp_reqs;

    // Tensors in input_info are views of the above shared tensors
    // Their shape is used to locate the offset of each input_info in the shared tensors
    Tensor input_length;
    Tensor expanded_input_length;
    Tensor kv_list;
    Tensor indexer_kv_list;
    Tensor kv_cache_offset;
    Tensor rotary_embedding_pos;
    Tensor rotary_embedding_mask;
    Tensor block_table;
    Tensor expanded_block_table;
    Tensor tile_scheduler_metadata;
    Tensor num_splits;
    Tensor cur_seq_len_start;    // 每个token可以看到的开始位置的token索引
    Tensor cur_seq_len_end;      // 每个token可以看到的结束位置的token索引
    Tensor paged_schedule_meta;  // for paged sparse mla indexer
    Tensor store_out_loc;

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

  input_info flash_input;               // input_ids length is non-specialized, use flash attention
  std::vector<input_info> page_inputs;  // input_ids length is fixed (`[1, decode_token_num_threshold]`), use page
                                        // attention, maintain only `page_input` with non-empty `dp_reqs`

  std::unique_ptr<CompressorMeta> compressor_meta_;  // meta for compress module
  std::unique_ptr<C4IndexerMeta> c4_indexer_meta_;   // meta for c4 indexer module
  std::unique_ptr<Dsv4MqaMeta> dsv4_mqa_meta_;       // meta for dsv4 mqa attention module

  // Divide the forward requests into two categories: flash, page (with different lengths)
  void PrepareInputInfo(const std::vector<ForwardRequest>& forward_reqs);
  // Prepare information related to tokens of the current batch of requests
  void PrepareInputIds(const std::vector<ForwardRequest>& forward_reqs);

  void PreparePrefill();
  void PrepareDecode();

  // Determine whether to use cache for the current batch of multi token requests
  void PrepareUseCache(input_info& input);
  void PrepareInputLength(input_info& input);
  void PrepareKVCacheBlocks(input_info& info);
  void PrepareKVCacheBlockTable(input_info& info);
  void PrepareDecodeRotary(input_info& input);
  void PrepareFlashMla(input_info& input);
  void PrepareFlashRotary(input_info& input);
  void PrepareFlexibleCache(input_info& input);
  void PrepareCuSeqLen(input_info& input, bool is_paged);
  void PrepareStoreOutLoc(input_info& input);
};

}  // namespace ksana_llm
