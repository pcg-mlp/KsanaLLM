/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <filesystem>

#ifdef ENABLE_CUDA
#  include "device/nvidia/nccl_utils.h"
#endif
#include "configure/global_context.h"
#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"
#include "models/base/attention_meta/c4_indexer_attention_meta_builder.h"
#include "models/base/attention_meta/indexer_attention_meta_builder.h"
#include "tests/test.h"
#include "utils/singleton.h"

namespace ksana_llm {

class IndexerAttentionMetaTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto& env = Singleton<Environment>::GetInstance();
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();
    env->ParseConfig(config_path);

    auto context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(1, 1, 1, env));
    context->global->InitDistributedConfig();

#ifdef ENABLE_CUDA
    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context->device = std::make_shared<DeviceContext>(1);
    context->device->ext->InitNcclParam(nccl_uid);
#else
    context->device = std::make_shared<DeviceContext>(1);
#endif

    env->GetModelConfig(model_config_);
    InitPipelineConfigForTest(env, model_config_);

    auto block_cache_layout = GetTestModelCacheLayout(env);
    env->InitializeBlockManagerConfig(block_cache_layout);
    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.block_host_memory_factor = 0.0;
    block_manager_config.device_allocator_config.blocks_num = 10;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;
    env->SetBlockManagerConfig(block_manager_config);

    env->GetRuntimeConfig(runtime_config_);
    runtime_config_.max_chunked_kv_tokens = 100;
    runtime_config_.max_batch_size = 16;
    runtime_config_.max_step_token_num = 256;
    runtime_config_.max_seq_len = 8192;
    runtime_config_.parallel_basic_config.attn_tensor_parallel_size = 1;

    context_ = context;

    EventCreateWithFlags(&kvcache_offset_event_, EVENT_DISABLE_TIMING);
    EventCreateWithFlags(&rotary_embedding_event_, EVENT_DISABLE_TIMING);

    InitIndexerShared(/*compress_ratio=*/1, runtime_config_);
  }

  // (Re-)initialize IndexerSharedAttentionMeta fields directly without going through Init().
  // This allows tests to reconfigure chunk limits mid-test.
  void InitIndexerShared(int compress_ratio, const RuntimeConfig& rc) {
    const size_t max_batch_size = rc.max_batch_size;
    const size_t max_token_num = rc.max_step_token_num;
    indexer_shared_ = IndexerSharedAttentionMeta{};
    indexer_shared_.compress_ratio = compress_ratio;
    indexer_shared_.max_chunked_q_tokens =
        static_cast<int>(ComputeIndexerMaxQChunkSize(rc.max_seq_len / compress_ratio, rc.max_step_token_num));
    indexer_shared_.max_chunked_kv_tokens = static_cast<int>(rc.max_chunked_kv_tokens);
    indexer_shared_.attn_dp_group_id = 0;
    indexer_shared_.chunk_token_starts_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {max_batch_size + 1});
    indexer_shared_.chunk_kv_token_ranges_host =
        Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {max_batch_size * 2});
    indexer_shared_.chunk_max_compressed_kv_lens_host =
        Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {max_batch_size});
    indexer_shared_.max_compressed_seq_len_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {1});
    // Pre-allocate device backing tensors at full capacity.
    indexer_shared_.cu_seq_len_start_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
    indexer_shared_.cu_seq_len_end_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
    indexer_shared_.batch_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
    indexer_shared_.expanded_seq_len_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size, GetDecodeTokenNumThreshold()});
    // Reset all write-offset cursors to zero (shape[0] tracks how many elements are in use).
    ResetSharedOffsets();
  }

  void ResetSharedOffsets() {
    indexer_shared_.cu_seq_len_start_tensor.shape = {0};
    indexer_shared_.cu_seq_len_end_tensor.shape = {0};
    indexer_shared_.batch_indices_tensor.shape = {0};
    indexer_shared_.expanded_seq_len_tensor.shape = {0};
    indexer_shared_.dp_context_tokens = 0;
    indexer_shared_.dp_decode_tokens = 0;
  }

  AttentionBuilderConfig MakeConfig() {
    return AttentionBuilderConfig{model_config_,
                                  runtime_config_,
                                  /*enable_blocked_multi_token_forwarding_kv=*/false,
                                  /*use_flashinfer_for_decode=*/false,
                                  /*rank=*/0,
                                  /*layer_num_on_node=*/0,
                                  /*block_size=*/0,
                                  kv_cache_types_,
                                  kv_cache_layer_offsets_,
                                  /*attn_dp_group_id=*/0,
                                  /*attn_dp_rank_id=*/0,
                                  context_,
                                  kvcache_offset_event_,
                                  rotary_embedding_event_};
  }

  // Build a FlashAttentionMeta with dp_reqs populated from extend/prefix arrays.
  void BuildFlashInput(const std::vector<int>& extend_lens, const std::vector<int>& prefix_lens) {
    flash_meta_.dp_reqs.clear();
    flash_meta_.total_dp_input_ids_len = 0;
    reqs_.clear();
    for (size_t i = 0; i < extend_lens.size(); ++i) {
      const int seq_len = extend_lens[i] + prefix_lens[i];
      ForwardRequest req;
      req.forwarding_tokens = std::make_shared<std::vector<int>>(seq_len, 0);
      req.kv_cached_token_num = prefix_lens[i];
      req.prefix_cache_len = prefix_lens[i];
      reqs_.push_back(req);
    }
    for (size_t i = 0; i < reqs_.size(); ++i) {
      flash_meta_.dp_reqs.push_back(&reqs_[i]);
      flash_meta_.total_dp_input_ids_len += extend_lens[i];
    }
  }

  IndexerSharedAttentionMeta indexer_shared_;
  IndexerAttentionMeta indexer_meta_;
  FlashAttentionMeta flash_meta_;
  std::vector<ForwardRequest> reqs_;
  std::shared_ptr<Context> context_;
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  std::vector<std::string> kv_cache_types_;
  std::unordered_map<std::string, std::vector<KvCacheLayerOffsets>> kv_cache_layer_offsets_;
  Event kvcache_offset_event_;
  Event rotary_embedding_event_;
};

// ── PrepareChunk (exercised via PrepareFlashIndexer) ──────────────────────────

TEST_F(IndexerAttentionMetaTest, PrepareChunk_SingleReqNoPrefix) {
  // Single req, no prefix → one chunk.
  BuildFlashInput(/*extend_lens=*/{5}, /*prefix_lens=*/{0});
  PrepareFlashIndexer(indexer_meta_, indexer_shared_, flash_meta_, MakeConfig());

  const int* token_starts = indexer_shared_.chunk_token_starts_host.GetPtr<int>();
  const int* kv_ranges = indexer_shared_.chunk_kv_token_ranges_host.GetPtr<int>();

  EXPECT_EQ(indexer_shared_.chunk_token_starts_host.shape[0], 2);
  EXPECT_EQ(indexer_shared_.chunk_kv_token_ranges_host.shape[0], 2);
  EXPECT_EQ(token_starts[0], 0);
  EXPECT_EQ(token_starts[1], 5);
  EXPECT_EQ(kv_ranges[0], 0);
  EXPECT_EQ(kv_ranges[1], 5);
}

TEST_F(IndexerAttentionMetaTest, PrepareChunk_TwoReqsNoPrefix_SingleChunk) {
  // Two reqs, no prefix → single chunk covering both.
  BuildFlashInput(/*extend_lens=*/{5, 3}, /*prefix_lens=*/{0, 0});
  PrepareFlashIndexer(indexer_meta_, indexer_shared_, flash_meta_, MakeConfig());

  const int* token_starts = indexer_shared_.chunk_token_starts_host.GetPtr<int>();
  const int* kv_ranges = indexer_shared_.chunk_kv_token_ranges_host.GetPtr<int>();

  EXPECT_EQ(indexer_shared_.chunk_token_starts_host.shape[0], 2);
  EXPECT_EQ(indexer_shared_.chunk_kv_token_ranges_host.shape[0], 2);
  EXPECT_EQ(token_starts[0], 0);
  EXPECT_EQ(token_starts[1], 8);
  EXPECT_EQ(kv_ranges[0], 0);
  EXPECT_EQ(kv_ranges[1], 8);
}

TEST_F(IndexerAttentionMetaTest, PrepareChunk_PrefixCausesKVSplit) {
  // Prefix causes split when accumulated KV exceeds max_chunked_kv_tokens.
  runtime_config_.max_chunked_kv_tokens = 10;
  InitIndexerShared(/*compress_ratio=*/1, runtime_config_);
  // req0: extend=5, prefix=0 → KV=5. req1: extend=3, prefix=6 → prefix KV (6) + req0 KV (5) = 11 ≥ 10 → split.
  BuildFlashInput(/*extend_lens=*/{5, 3}, /*prefix_lens=*/{0, 6});
  PrepareFlashIndexer(indexer_meta_, indexer_shared_, flash_meta_, MakeConfig());

  const int* token_starts = indexer_shared_.chunk_token_starts_host.GetPtr<int>();
  const int* kv_ranges = indexer_shared_.chunk_kv_token_ranges_host.GetPtr<int>();

  EXPECT_EQ(indexer_shared_.chunk_token_starts_host.shape[0], 3);
  EXPECT_EQ(indexer_shared_.chunk_kv_token_ranges_host.shape[0], 4);
  EXPECT_EQ(token_starts[0], 0);
  EXPECT_EQ(token_starts[1], 5);
  EXPECT_EQ(token_starts[2], 8);
  EXPECT_EQ(kv_ranges[0], 0);
  EXPECT_EQ(kv_ranges[1], 5);
  EXPECT_EQ(kv_ranges[2], 5);
  EXPECT_EQ(kv_ranges[3], 14);
}

TEST_F(IndexerAttentionMetaTest, PrepareChunk_QTokenLimitCausesSplit) {
  // Q token limit causes split even when KV budget is ample.
  // Allocate tensors large enough for 20 tokens, then override max_chunked_q_tokens
  // to 12 to trigger the split — the two values must be kept separate because the
  // device tensors must be >= total token count while the chunk limit can be smaller.
  runtime_config_.max_step_token_num = 64;
  runtime_config_.max_seq_len = 256;
  runtime_config_.max_chunked_kv_tokens = 100;
  InitIndexerShared(/*compress_ratio=*/1, runtime_config_);
  indexer_shared_.max_chunked_q_tokens = 12;  // force Q-limit split on 20-token request
  // One request with 20 tokens, max Q per chunk = 12 → split into two chunks.
  BuildFlashInput(/*extend_lens=*/{20}, /*prefix_lens=*/{0});
  PrepareFlashIndexer(indexer_meta_, indexer_shared_, flash_meta_, MakeConfig());

  const int* token_starts = indexer_shared_.chunk_token_starts_host.GetPtr<int>();
  const int* kv_ranges = indexer_shared_.chunk_kv_token_ranges_host.GetPtr<int>();

  EXPECT_EQ(indexer_shared_.chunk_token_starts_host.shape[0], 3);
  EXPECT_EQ(indexer_shared_.chunk_kv_token_ranges_host.shape[0], 4);
  EXPECT_EQ(token_starts[0], 0);
  EXPECT_EQ(token_starts[1], 12);
  EXPECT_EQ(token_starts[2], 20);
  EXPECT_EQ(kv_ranges[0], 0);
  EXPECT_EQ(kv_ranges[1], 12);
  EXPECT_EQ(kv_ranges[2], 0);
  EXPECT_EQ(kv_ranges[3], 20);
}

// ── PrepareBatchIndices (exercised via PrepareFlashIndexer) ───────────────────

TEST_F(IndexerAttentionMetaTest, PrepareBatchIndices_SingleReq) {
  ResetSharedOffsets();
  // Single request → batch_indices are all 0.
  BuildFlashInput(/*extend_lens=*/{5}, /*prefix_lens=*/{0});
  PrepareFlashIndexer(indexer_meta_, indexer_shared_, flash_meta_, MakeConfig());

  const auto& batch_indices = indexer_meta_.batch_indices_host;
  ASSERT_EQ(batch_indices.size(), 5u);
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(batch_indices[i], 0);
  }
}

TEST_F(IndexerAttentionMetaTest, PrepareBatchIndices_MultiReqs) {
  ResetSharedOffsets();
  // Multiple requests → batch_indices match each token's owning request.
  BuildFlashInput(/*extend_lens=*/{3, 2, 4}, /*prefix_lens=*/{0, 0, 0});
  PrepareFlashIndexer(indexer_meta_, indexer_shared_, flash_meta_, MakeConfig());

  const auto& batch_indices = indexer_meta_.batch_indices_host;
  ASSERT_EQ(batch_indices.size(), 9u);
  EXPECT_EQ(batch_indices[0], 0);
  EXPECT_EQ(batch_indices[1], 0);
  EXPECT_EQ(batch_indices[2], 0);
  EXPECT_EQ(batch_indices[3], 1);
  EXPECT_EQ(batch_indices[4], 1);
  EXPECT_EQ(batch_indices[5], 2);
  EXPECT_EQ(batch_indices[6], 2);
  EXPECT_EQ(batch_indices[7], 2);
  EXPECT_EQ(batch_indices[8], 2);
}

// ── PrepareCuSeqLen flash path (exercised via PrepareFlashIndexer) ─────────────

TEST_F(IndexerAttentionMetaTest, PrepareCuSeqLen_Flash_NoChunking) {
  ResetSharedOffsets();
  // Flash, no chunking → each token's cu_seq_len starts at 0, end = position within request.
  BuildFlashInput(/*extend_lens=*/{3, 2}, /*prefix_lens=*/{0, 0});
  PrepareFlashIndexer(indexer_meta_, indexer_shared_, flash_meta_, MakeConfig());

  const auto& start = indexer_meta_.cu_seq_len_start_host;
  const auto& end = indexer_meta_.cu_seq_len_end_host;
  ASSERT_EQ(start.size(), 5u);
  ASSERT_EQ(end.size(), 5u);
  std::vector<int> expected_end = {1, 2, 3, 1, 2};
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(start[i], 0) << "start i=" << i;
    EXPECT_EQ(end[i], expected_end[i]) << "end i=" << i;
  }
}

TEST_F(IndexerAttentionMetaTest, PrepareCuSeqLen_Flash_WithChunking) {
  // Flash with chunking → tokens after a split continue from their prefix position.
  runtime_config_.max_chunked_kv_tokens = 10;
  InitIndexerShared(/*compress_ratio=*/1, runtime_config_);
  ResetSharedOffsets();
  // req0: extend=3, prefix=0 → KV=3. req1: extend=2, prefix=8 → prefix alone (8) + req0 KV (3) = 11 > 10 → split.
  BuildFlashInput(/*extend_lens=*/{3, 2}, /*prefix_lens=*/{0, 8});
  PrepareFlashIndexer(indexer_meta_, indexer_shared_, flash_meta_, MakeConfig());

  const auto& start = indexer_meta_.cu_seq_len_start_host;
  const auto& end = indexer_meta_.cu_seq_len_end_host;
  ASSERT_EQ(start.size(), 5u);
  ASSERT_EQ(end.size(), 5u);
  std::vector<int> expected_end = {1, 2, 3, 9, 10};
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(start[i], 0) << "start i=" << i;
    EXPECT_EQ(end[i], expected_end[i]) << "end i=" << i;
  }
}

// ── PrepareCuSeqLen paged path (exercised via PreparePagedIndexer) ─────────────

TEST_F(IndexerAttentionMetaTest, PrepareCuSeqLen_Paged) {
  ResetSharedOffsets();
  // Paged (decode) path: two requests with the same q_seq_len (=2).
  // Requests in a paged batch all have the same q_seq_len.
  ForwardRequest req0, req1;
  req0.forwarding_tokens = std::make_shared<std::vector<int>>(5, 0);  // seq=5, extend=2, prefix=3
  req0.kv_cached_token_num = 3;
  req0.prefix_cache_len = 3;
  req1.forwarding_tokens = std::make_shared<std::vector<int>>(11, 0);  // seq=11, extend=2, prefix=9
  req1.kv_cached_token_num = 9;
  req1.prefix_cache_len = 9;

  PagedAttentionMeta page_meta;
  page_meta.dp_reqs = {&req0, &req1};
  page_meta.q_seq_len = 2;
  page_meta.total_dp_input_ids_len = 4;  // 2 per request

  PreparePagedIndexer(indexer_meta_, indexer_shared_, page_meta, MakeConfig());

  const auto& start = indexer_meta_.cu_seq_len_start_host;
  const auto& end = indexer_meta_.cu_seq_len_end_host;
  ASSERT_EQ(start.size(), 4u);
  ASSERT_EQ(end.size(), 4u);
  // req0: prefix=3, tokens i=[1,2] → end=[4, 5]
  // req1: prefix=9, tokens i=[1,2] → end=[10, 11]
  std::vector<int> expected_end = {4, 5, 10, 11};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(start[i], 0) << "start i=" << i;
    EXPECT_EQ(end[i], expected_end[i]) << "end i=" << i;
  }
  EXPECT_EQ(indexer_shared_.dp_decode_tokens, 4u);
}

}  // namespace ksana_llm
