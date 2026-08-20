/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/environment.h"
#include "models/base/attention_meta/attention_meta.h"
#include "models/base/attention_meta/attention_meta_creator.h"
#include "models/base/attention_meta/flash_attention_meta_builder.h"
#include "models/base/attention_meta/paged_attention_meta_builder.h"
#include "models/base/base_model.h"
#include "runtime/forward_request.h"
#include "runtime/infer_stage.h"
#include "utils/tensor.h"

namespace ksana_llm {

// Forward declaration for gather index builder injection
class GatherIndexBuilderInterface;

// Convert input ids to expected format.
class ModelInput {
 public:
  ModelInput(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
             std::shared_ptr<Context> context, std::shared_ptr<BaseModelCacheLayout> block_cache_layout);
  ~ModelInput();

  // Parse forward request into model input tensors.
  void ParseFromRequests(const std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode = RunMode::kMain);

  // build-ahead: 指定本 step 使用的 ModelInput slot，H2D 走对应 slot stream。
  void SetParseSlot(size_t slot) { parse_slot_ = slot; }

  // Propagate inflight resource manager to CompressorSharedAttentionMeta (DSv4-specific;
  // no-op for other models). Called from ForwardingContext::SetInflightResourceManager.
  void SetInflightResourceManager(InflightResourceManager* mgr);

  // Inject an external gather index builder (owned by MtpDecoder).
  // When non-null, PrepareNextNGatherIdx delegates all state management to this builder.
  void SetExternalGatherIdxBuilder(GatherIndexBuilderInterface* builder) { external_gather_idx_builder_ = builder; }

  // Return the currently injected external gather index builder (nullptr if none).
  // Used by speculative decoders to temporarily detach the builder around calls that would otherwise
  // trigger an unwanted UpdateMainForward (see ExecuteForward).
  GatherIndexBuilderInterface* GetExternalGatherIdxBuilder() const { return external_gather_idx_builder_; }

  // Return the external builder's gather index tensor.
  // The builder must be injected via SetExternalGatherIdxBuilder before calling this method.
  // Called by MTP layers (DeepSeekV4MtpLayer, GLM4MoE) via forwarding_context.GetModelInput().
  Tensor* GetEffectiveNextNGatherIdxTensor();

  // Build MTP-specific gather indices from external builder state.
  // Called by MtpDecoder::ExecuteForward per MTP step.
  void PrepareMtpDraftGatherIndices(const std::vector<ForwardRequest>& forward_reqs);

 private:
  void PrepareInputRefit(const std::vector<ForwardRequest>& forward_reqs);
  void PrepareVLInputRefit(const std::vector<ForwardRequest>& forward_reqs);
  void CreateVLTensors();
  void PrepareVLRequest(const std::vector<ForwardRequest>& forward_reqs);
  void PrepareCutoffLayer(const std::vector<ForwardRequest>& forward_reqs);
  void PrepareMainForwardGather(const std::vector<ForwardRequest>& forward_reqs);

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

#if defined(ENABLE_ACL) || defined(ENABLE_KUNLUN)
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

  // The max prefix length among all dp requests (for prefix cross-attention max_kv_tokens).
  size_t dp_max_prefix_per_req = 0;

  // current request batchsize matches cudagraph catpure range
  bool is_cudagraph_batchsize_matched = false;

  // Whether the current batch concerns the request_target
  bool has_request_target = false;

  // For cutoff layer
  int cutoff_layer = 0;

  // Whether to use kv cache.
  bool use_cache = true;

  std::vector<size_t> dp_input_prefix_list_uint64;
  std::vector<size_t> input_offset_list_uint64;
  std::vector<size_t> input_prefix_list_uint64;

  std::vector<int> input_ids_cpu;

  // The infer stage, context decode or decode.
  InferStage infer_stage;

  // The input ids, int32
  Tensor input_ids;

  // 本步 batch 的 (req_id, last_token_offset) 配对 tensor（int64，容量 2 * max_batch_size）。
  // ReplaceLastTokenKernel 作查询键，WriteRingKernel 取 req_id（offset 部分忽略）。
  // 慢路径不 launch 读它的 kernel，但 PrepareInputIds 仍 fill 一次 H2D 以简化分支。
  Tensor cur_batch_pairs_tensor;

  // The ids offset tensor, uint64
  Tensor input_offset_uint64_tensor;
  Tensor dp_input_offset_uint64_host;
  Tensor dp_input_offset_uint64_tensor;

  // The input's prefix length
  Tensor input_prefix_uint64_tensor;
  Tensor dp_input_prefix_uint64_tensor;

  Tensor dp_prefill_q_offset_uint64_host;
  Tensor dp_prefill_q_offset_uint64_tensor;

#ifdef ENABLE_KUNLUN
  // Kunlun-specific int32 tensors for xfa VectorParam (qlod/kvlod/prefix_kvlod).
  // device tensors are packed into input_pack_device_ (see BindInputPackDeviceViews).
  // host tensors are standalone LOCATION_HOST (directly consumed by layer ForwardT).
  Tensor dp_prefill_q_offset_int32_tensor;   // device: qlod
  Tensor dp_prefill_q_offset_int32_host_tensor;  // host: qlod

  Tensor dp_input_offset_int32_tensor;        // device: kvlod
  Tensor dp_input_offset_int32_host_tensor;       // host: kvlod

  Tensor dp_input_prefix_int32_tensor;           // device: prefix_kvlod (packed into input_pack_device_)
  Tensor dp_input_prefix_int32_host_tensor;      // host: prefix_kvlod [0, prefix_len_0, ...]
#endif

  // The 3-dimentional index position for multimodal rotarty embedding.
  Tensor dp_mrotary_embedding_pos;
  // The 4-dimentional index position for xd rotarty embedding.
  Tensor dp_xdrotary_embedding_pos;

  // Record which logits in the output of all tokens need to be extracted for subsequent sampling calculations
  // Due to the presence of logits_custom_length and speculative_decoding, a single request may require extracting more
  // than one logit. In the standard case, only the last logit of each request needs to be retrieved
  Tensor logits_idx_uint64_tensor;

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
#if defined(ENABLE_KUNLUN)
  // Signaled when all H2D copies in PrepareATBKVCache complete.
  // compute stream waits on this before first attention layer instead of
  // blocking the host with StreamSynchronize(H2DStream).
  Event h2d_atb_event;
#endif

#if defined(ENABLE_ACL) || defined(ENABLE_KUNLUN)
  // record all reqs token number on host, shape: [batch_size]
  Tensor seq_len_host;
  // device mirror of seq_len_host, prepared once per batch in PrepareATBKVCache
  Tensor context_len_device;
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

  // Cache signature for the P2 block-table skip-if-unchanged optimization. Captures batch_size and each
  // selected request's block-id vector; when unchanged across steps we skip the host triple-loop construction
  // and the H2D copy. Not updated under xpu_graph_mode_ (graph capture/replay must always see the fresh table).
  std::vector<int32_t> block_table_cache_sig_;

  // Reusable all-ones host buffer for PrepareDecodeRotary's rotary mask (P3). The decode mask is always all
  // ones and never mutated; we reuse this buffer across steps to avoid the per-step vector allocation + fill(1).
  // The H2D is still performed each step because PrepareFlashRotary (prefill) writes non-ones into the same
  // device buffer, so a skip would read stale prefill data.
  std::vector<int64_t> decode_rotary_mask_ones_;

  // Reusable host staging buffer for PrepareDecodeRotary's rotary positions. On KUNLUN that path no longer
  // does a host-side StreamSynchronize after the H2D (it publishes rotary_embedding_event and lets the
  // consumer StreamWaitEvent on it instead), so the copy's *source* must outlive the enqueue: a function
  // local vector would be freed the instant PrepareDecodeRotary returns, while the DMA may still be in
  // flight. Keeping it as a member also removes a per-step allocation. Safe to overwrite on the next step
  // because the intervening forward has already waited on rotary_embedding_event.
  std::vector<int64_t> decode_rotary_pos_host_;

  // since layer's forward only support Tensor as input (nothing to do with karlluo), such crappy design ignore runtime
  // attribute, so we need a tensor to be attribute.
  // shape: [2]; 0: layers_slot_mapping_dim_1; 1: max_num_blocks_per_query
  Tensor atb_attention_attr;

  // assemble last token index for gather, dtype is int64_t
  Tensor last_token_index_tensor;

  std::vector<void*> kv_cache_ptrs;
  Tensor kv_cache_ptrs_tensor;

  // XPU Graph mode: fixed-address device buffers for decode graph capture/replay.
  // Activated by KSANA_DECODE_GRAPH=1 env var.
  bool xpu_graph_mode_{false};
  size_t graph_fixed_max_blocks_per_query_{0};
  Tensor graph_rotary_pos_buf;
  Tensor graph_seq_len_buf;
#endif
  // Shared attention tensors, flexible-cache tensors, and lazy-initialized kv-cache pointer state.
  std::unique_ptr<SharedAttentionMeta> shared_attention_meta;

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

  // External gather index builder injected by speculative decoders.
  // When non-null, PrepareNextNGatherIdx delegates to this builder.
  GatherIndexBuilderInterface* external_gather_idx_builder_ = nullptr;

  // host 侧缓冲，与 cur_batch_pairs_tensor 一一对应，layout [req_id_0, offset_0, ...]。
  std::vector<int64_t> cur_batch_pairs_cpu_;

  // PrepareInputIds 统一 H2D：固定 meta 区在前，末尾接 input_ids（按本步实际长度传输）。
  // 布局（字节）：pairs | logits | input_offset | input_prefix | dp_input_offset |
  //               dp_input_prefix | dp_prefill_q_offset | input_ids
  struct InputPackLayout {
    size_t off_cur_batch_pairs = 0;
    size_t off_logits_idx = 0;
    size_t off_input_offset = 0;
    size_t off_input_prefix = 0;
    size_t off_dp_input_offset = 0;
    size_t off_dp_input_prefix = 0;
    size_t off_dp_prefill_q_offset = 0;
    size_t off_input_ids = 0;  // == meta_pack_bytes
    size_t meta_pack_bytes = 0;
    size_t total_pack_bytes = 0;
    size_t cur_batch_pairs_bytes = 0;
    size_t logits_idx_bytes = 0;
    size_t batch_offset_bytes = 0;  // (max_batch + 1) * sizeof(uint64_t)
#ifdef ENABLE_KUNLUN
    size_t off_dp_prefill_q_offset_int32 = 0;
    size_t off_dp_input_offset_int32 = 0;
    size_t off_dp_input_prefix_int32 = 0;
    size_t int32_offset_bytes = 0;  // (max_batch + 1) * sizeof(int32_t)
#endif
  };
  InputPackLayout input_pack_layout_;
  Tensor input_pack_host_;
  Tensor input_pack_device_;

  // ParseFromRequests 期间使用的 slot (决定 GetParseH2DStream).
  size_t parse_slot_ = 0;
  Stream& GetParseH2DStream();

  // 将各公开 device Tensor 绑到 input_pack_device_ 的固定偏移（shared view）。
  void BindInputPackDeviceViews();

  // Shared config for all attention meta builders.
  std::unique_ptr<AttentionBuilderConfig> builder_config_;
  // Builders that construct flash_meta / page_metas for each forward step.
  // Initialized in the constructor with the standard implementations;
  // can be replaced (e.g., via a setter) to swap in alternative backends.
  std::unique_ptr<AttentionMetaCreator> attention_meta_creator_;
  std::unique_ptr<FlashAttentionMetaBuilder> flash_builder_;
  std::unique_ptr<PagedAttentionMetaBuilder> paged_builder_;

 public:
  // input_ids length is non-specialized, use flash attention
  std::unique_ptr<FlashAttentionMeta> flash_meta;
  // input_ids length is fixed (`[1, decode_token_num_threshold]`), use paged attention.
  // Only entries with non-empty `dp_reqs` are maintained.
  std::vector<std::unique_ptr<PagedAttentionMeta>> page_metas;

  // Typed accessors for use by attention modules.
  FlashAttentionMeta& GetFlashMeta() { return *flash_meta; }
  const FlashAttentionMeta& GetFlashMeta() const { return *flash_meta; }
  std::vector<std::unique_ptr<PagedAttentionMeta>>& GetPagedMetas() { return page_metas; }
  const std::vector<std::unique_ptr<PagedAttentionMeta>>& GetPagedMetas() const { return page_metas; }

  // Return the config used to construct the current AttentionMetaCreator.
  // Useful when replacing the creator with a model-specific subclass.
  const AttentionMetaCreator::Config& GetCreatorConfig() const;

  // Replace the attention info creator and rebuild shared_attention_meta / flash_meta.
  // Called during model initialization to swap in a model-specific creator.
  void ReplaceAttentionMetaCreator(std::unique_ptr<AttentionMetaCreator> creator);

  // Divide the forward requests into two categories: flash, page (with different lengths)
  void PrepareInputInfo(const std::vector<ForwardRequest>& forward_reqs);
  // Prepare information related to tokens of the current batch of requests
  void PrepareInputIds(const std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode = RunMode::kMain);

  void PreparePrefill();
  void PrepareDecode();
};

}  // namespace ksana_llm
