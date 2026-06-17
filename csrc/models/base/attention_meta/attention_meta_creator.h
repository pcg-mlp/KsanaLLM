/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "models/base/attention_meta/attention_meta_builder.h"
#include "models/base/attention_meta/flash_attention_meta_builder.h"
#include "models/base/attention_meta/paged_attention_meta_builder.h"
#include "runtime/forward_request.h"

namespace ksana_llm {

// Forward declaration: keeps models/base/ free of any executor/ dependency.
// Full definition is only needed in model-specific creator .cpp files that
// override WireInflightResources.
class InflightResourceManager;

// Factory interface for attention input meta objects owned by ModelInput.
// All construction parameters are injected once at construction time via Config;
// the factory methods themselves take no arguments.
// Concrete attention backends can provide derived creators that return model-specific
// meta objects and cast from these base pointers in their own implementation.
class AttentionMetaCreator {
 public:
  // All parameters needed to allocate shared / flash / paged attention meta tensors.
  struct Config {
    const ModelConfig& model_config;
    const RuntimeConfig& runtime_config;
    int layer_num_on_node;
    size_t max_table_block_num;
    size_t max_token_num;
    size_t decode_token_num_threshold;
    bool enable_blocked_multi_token_forwarding_kv;
    int rank;
    // Index of the first layer on this node (pipeline parallelism offset).
    // Used by model-specific creators (e.g. DeepSeekV4) to map per-layer cache offsets.
    int layer_start_on_node = 0;
    // KV cache layout; used by model-specific creators that need block-level offset information.
    std::shared_ptr<BaseModelCacheLayout> block_cache_layout = nullptr;
    // Execution context; used by model-specific creators that need it in BuildAttentionMetas
    // (e.g. DeepSeekV4 for PrepareCompressor).
    std::shared_ptr<Context> context = nullptr;
  };

  // Per-batch routing result returned by BuildAttentionMetas.
  struct MetaArrays {
    std::unique_ptr<FlashAttentionMeta> flash_meta;
    std::vector<std::unique_ptr<PagedAttentionMeta>> page_metas;
    size_t multi_token_request_num = 0;
    size_t dp_multi_token_request_num = 0;
    size_t single_token_request_num = 0;
    size_t dp_single_token_request_num = 0;
  };

  explicit AttentionMetaCreator(const Config& config) : config_(config) {}
  virtual ~AttentionMetaCreator() = default;

  virtual std::unique_ptr<SharedAttentionMeta> CreateSharedAttentionMeta();
  virtual std::unique_ptr<FlashAttentionMeta> CreateFlashAttentionMeta();
  virtual std::unique_ptr<PagedAttentionMeta> CreatePagedAttentionMeta();

  // Routes forward_reqs into flash / paged meta objects and returns them with request-count stats.
  // Default implementation uses ForwardRequestType (kFlash → flash, kPage → paged by q_seq_len).
  // Override in model-specific creators to apply a different routing policy
  // (e.g. Compressor: q_seq_len == 1 → paged, others → flash).
  // shared_meta is the current batch's shared attention meta; model-specific overrides may use
  // it to prepare embedded sub-metas (e.g. CompressorAttentionMeta) alongside routing.
  virtual MetaArrays BuildAttentionMetas(const std::vector<ForwardRequest>& forward_reqs, uint32_t attn_dp_group_id,
                                         SharedAttentionMeta& shared_meta);

  // Builder factories: return the model-specific builder for the flash / paged attention paths.
  // The default implementations return Standard*Builder; model-specific creators override these
  // to return their own derived builders (e.g. DeepSeekV3FlashAttentionMetaBuilder).
  virtual std::unique_ptr<FlashAttentionMetaBuilder> CreateFlashBuilder(const AttentionBuilderConfig& config);
  virtual std::unique_ptr<PagedAttentionMetaBuilder> CreatePagedBuilder(const AttentionBuilderConfig& config);

  // Expose the stored configuration so derived creators or ModelInput can access it.
  const Config& GetConfig() const { return config_; }

  // Wire model-specific inflight resources into shared_meta.  Default is a no-op;
  // model-specific creators override this to populate resource pointers cached
  // directly on the shared meta (e.g. CompressorSlotResource for DeepSeekV4).
  // Called once from ModelInput::SetInflightResourceManager after Load() completes.
  virtual void WireInflightResources(InflightResourceManager* /*mgr*/, SharedAttentionMeta* /*shared_meta*/) {}

 protected:
  void InitStandardSharedAttentionMeta(StandardSharedAttentionMeta& shared_meta);

  // Decide whether the flash requests should populate kv-cache tensors.
  virtual bool ShouldFlashUseCache(const FlashAttentionMeta& flash_meta) const;

  Config config_;
};

}  // namespace ksana_llm
