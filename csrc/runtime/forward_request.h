/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "runtime/infer_stage.h"
#include "utils/logger.h"
#include "utils/request.h"

namespace ksana_llm {

// The classified types of forward requests
enum class ForwardRequestType : uint8_t {
  kFlash = 0,  // variable input length, use flash attention
  kPage = 1,   // fixed input length (`[1, decode_token_num_threshold]`), use paged attention
};

// The information used for forward.
struct ForwardRequest {
  // The request id.
  int64_t req_id;

  // The infer stage, context decode or decode.
  InferStage infer_stage = InferStage::kContext;

  // The decode step, 0 for context decode, and then 1, 2, 3...
  int step = 0;

  // The number of tokens whose kv caches have been generated.
  int kv_cached_token_num = 0;

  // pd_v2: peer Prefill RDMA-wrote the K/V for the cached prefix; trust
  // the InferRequest's kv_cached_token_num + infer_stage as set by the
  // hook and route straight to kPage. Mirrors InferRequest::kv_from_remote;
  // propagated by model_builder. See InferRequest's longer
  // comment for the full lifecycle. Future short-request locally-completed
  // path keeps this false → falls through to the standard Type computation.
  bool kv_from_remote = false;

  // pd_v2 Prefill node: this req must RDMA-write its K/V to a Decode peer
  // (set iff InferRequest::pd_v2_decode_targets is non-empty). The per-layer
  // KV-dispatch hook (RecordLayerProgress) lives ONLY in the flash/context
  // attention path; without forcing kFlash, a full-prefix-hit req
  // (input_ids_len <= GetDecodeTokenNumThreshold(), kv_cached_token_num > 0)
  // would fall through to kPage (decode kernel), the hook would never fire,
  // and the Decode peer would wait forever for KV + PREFILL_RETURN.
  bool kv_to_remote = false;

  // The custom length for the logits output, allowing for a specific size of logits to be generated.
  size_t logits_custom_length = 0;

  size_t sampling_token_num = 1;

  size_t last_step_token_num = 1;

  // Multimodal rotary position embedding offset, this points to the corresponding member in infer_request.
  int64_t* mrotary_embedding_pos_offset = nullptr;
  int64_t* xdrotary_embedding_pos_offset = nullptr;

  // forwarding_tokens contains tokens used in forwarding step. There are two parts:
  // 1. tokens have kv-caches, kv_cached_token_num is the number
  // 2. tokens need to be processed, their kv-caches are generated during computation
  std::shared_ptr<std::vector<int>> forwarding_tokens;

  // Embedding slice used to refit input embedding
  EmbeddingSlice* input_refit_embedding;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  std::shared_ptr<const std::unordered_map<std::string, TargetDescribe>> request_target;

  // The result of request_target.
  std::unordered_map<std::string, PythonTensor>* response = nullptr;

  size_t logits_offset;

  // The kv cache addresses
  std::vector<void*> kv_cache_ptrs;

  // The SWA kv cache block IDs for sliding window attention.
  // Empty when SWA is disabled.
  std::vector<int> swa_kv_cache_blocks;

  // The base address of the first SWA kv cache block (block_id=0).
  // Since SWA blocks are allocated in contiguous memory (each block is block_size apart),
  // only the base address is needed. nullptr when SWA is disabled.
  void* swa_block_base_addr = nullptr;

  // Per-block FA global index: swa_kv_cache_block_idx_offsets[i] is the FA block
  // global index corresponding to swa_kv_cache_blocks[i].
  // Must have the same length as swa_kv_cache_blocks.
  std::vector<size_t> swa_kv_cache_block_idx_offsets;

  // Total number of blocks in the SWA pool. Filled by model_builder in SWA-enabled paths
  // (taken from swa_block_allocator_manager_); used by the attention kernel for bounds checks.
  // 0 indicates SWA pool is not enabled.
  int64_t swa_pool_block_num = 0;

  // The length of the flexible cache, indicating the number of elements stored in the flexible cache for potential
  // reuse in subsequent computations.
  int flexible_cache_len = 0;

  // A vector containing pointers to FlexibleCachedCopyTask objects, which represent tasks that involve copying data
  // flexibly between different memory regions.
  std::vector<FlexibleCachedCopyTask>* flexible_cached_copy_tasks = nullptr;

  // The prefix cache tokens number
  int prefix_cache_len = 0;

  // The sampling config.
  SamplingConfig* sampling_config = nullptr;

  std::shared_ptr<BlockAllocatorManager> block_allocator_manager;

  // current froward request related attention data para group id
  // NOTE(karlluo): for example: machine has 4 GPUs, Attention Data Parallelism is 2, Tensor Parallelism is 2.
  // |----Attn DP Group id 0----|----Attn DP Group id 1----|
  // |     TP 0   |     TP1     |     TP0    |     TP1     |
  // |     GPU0   |     GPU1    |     GPU2   |     GPU3    |
  uint32_t attn_dp_group_id = 0;

#if defined(ENABLE_ACL) || defined(ENABLE_CUDA)
  // NOTE(xingjinglu)When ENABLE_CUDA, if enable_blocked_multi_token_fowarding_kv is true, it
  // will use the variable the same as ACL. Now, regardless of whether enable_blocked_multi_token_fowarding_kv is
  // turned on, the variable definition here will be enabled.
  // NOTE(karlluo): for ATB, all device blocks locate on a
  // flatten plane memory space. The Ksana kv cache consists of blocks, each of which is an independent storage space.
  // The blocks are not guaranteed to be contiguous in memory. Each block has a shape of [2, layer_num, block_token_num,
  // head_num, head_dim], where 2 represents key and value. The Ascend ATB kv cache consists of kcache and vcache, which
  // are independent contiguous storage spaces. The shapes of kcache and vcache are [num_blocks * layer_num,
  // block_token_num, head_num, head_dim]. Each block has a size of [block_token_num, head_num, head_dim]. To interface
  // with the NPU, Ascend ATB (hereinafter referred to as ATB) needs to be used. In order for the NPU's self/paged
  // attention to utilize Ksana's kv cache and share the underlying memory/GPU memory management capabilities, the Ksana
  // kv cache needs to be converted to the Ascend ATB kv cache format.
  // 1. Change the block allocation method so that the blocks are contiguous in physical memory, while the upper-level
  // pointers point to different storage spaces. Originally, each block in the Ksana kv cache called malloc once. This
  // should be changed to pre-allocate a contiguous storage space of size [num_blocks, 2, layer_num, block_token_num,
  // head_num, head_dim]. The pointers of each block should then point to cache_base_ptr + (block index * 2 *
  // layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
  // 2. During each inference process, each prompt will carry an array of block IDs, which can be used to obtain the
  // pointers to the storage space. For ATB, conversion is required to use these pointers. The conversion process is
  // as follows:
  //    - Given a block ID array [b0, b1, b2, b3, b4] and the base address pointer of the Ksana kv cache after the
  //    modification in step 1, cache_base_ptr.
  //    - For ATB: The Ksana kv cache has a total of num_blocks * 2 * layer_num blocks.
  //    - Therefore, the block ID array for ATB is [b0 * layer_num * 2, b1 * layer_num * 2, b2 * layer_num * 2, b3 *
  //    layer_num * 2, b4 * layer_num * 2].
  //    - Ksana's kv cache swaps memory/GPU memory at the block level, so to reuse Ksana's kv cache's underlying
  //    memory/GPU memory management capabilities, ATB's kcache and vcache share the same Ksana kv cache.
  //    - Since each block in Ksana is divided into K and V parts, each part having a size of [layer_num,
  //    block_token_num, head_num, head_dim].
  //    - To allow ATB's kcache and vcache to share the same block ID array, the kcache pointer is cache_base_ptr, and
  //    the vcache pointer is cache_base_ptr + (layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
  //    - Therefore, the block ID array for kcache/vcache is [b0 * layer_num * 2 + layer_idx, b1 * layer_num * 2 +
  //    layer_idx, b2 * layer_num * 2 + layer_idx, b3 * layer_num * 2 + layer_idx, b4 * layer_num * 2 + layer_idx].
  std::vector<std::vector<int32_t>> atb_kv_cache_base_blk_ids;
#endif

  // Actual input length during model forward pass, excluding cached length
  size_t GetInputIdsLength() const {
    // Due to flexible cache, during context phase, `prefix_cache_len` of this forward request is increased by
    // `flexible_cache_len`, potentially exceed `kv_cached_token_num`; so the maximum of the two is taken here
    return forwarding_tokens->size() - std::max(kv_cached_token_num, prefix_cache_len);
  }

  // Get the request type, flash for prefill, page for decode
  ForwardRequestType GetType() const {
    // pd_v2 fast-path: peer Prefill staged the K/V; this is the first
    // decode-step after PrefillComplete, NOT a fresh prefill. Force
    // kPage so the model picks paged_mla (decode kernel) instead of
    // flash_mla (prefill kernel). If we let kFlash fire, Decode's own
    // prefill would recompute K/V over the bytes Prefill RDMA'd into
    // the KV pool, producing degenerate output (bytes arrive byte-perfect,
    // but model overwrites them).
    //
    // The kv_from_remote flag is the contract; future "short-request
    // locally completed on Decode without forwarding to Prefill"
    // feature can reuse this kPage path or fall through to kFlash
    // depending on whether KV is local-vs-remote — that distinction
    // lives entirely on this flag.
    if (kv_from_remote) {
      return ForwardRequestType::kPage;
    }
    // pd_v2 Prefill node: a req destined for a Decode peer must run the
    // flash/context kernel so the per-layer KV-dispatch hook
    // (RecordLayerProgress -> PdV2KvDispatcher) fires and the K/V is RDMA'd to
    // Decode. Full-prefix-hit reqs (1 sampler token over a fully-cached
    // prompt) have GetInputIdsLength() <= threshold and would otherwise hit
    // the kPage branch below — which has no dispatch hook — orphaning Decode.
    if (kv_to_remote) {
      return ForwardRequestType::kFlash;
    }
    if (GetInputIdsLength() <= GetDecodeTokenNumThreshold() && kv_cached_token_num > 0) {
      return ForwardRequestType::kPage;
    } else {
      return ForwardRequestType::kFlash;
    }
  }
};

}  // namespace ksana_llm
