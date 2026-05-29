/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <filesystem>

#ifdef ENABLE_CUDA
#  include "device/nvidia/nccl_utils.h"
#endif
#include "configure/global_context.h"
#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"
#include "models/base/forward_meta/indexer_meta.h"
#include "tests/test.h"
#include "utils/singleton.h"

namespace ksana_llm {

class IndexerMetaTest : public testing::Test {
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

    ModelConfig model_config;
    env->GetModelConfig(model_config);

    InitPipelineConfigForTest(env, model_config);

    auto block_cache_layout = GetTestModelCacheLayout(env);
    env->InitializeBlockManagerConfig(block_cache_layout);
    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.block_host_memory_factor = 0.0;
    block_manager_config.device_allocator_config.blocks_num = 10;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;
    env->SetBlockManagerConfig(block_manager_config);

    RuntimeConfig runtime_config;
    env->GetRuntimeConfig(runtime_config);

    runtime_config.max_chunked_kv_tokens = 100;
    runtime_config.max_batch_size = 16;
    runtime_config.max_step_token_num = 256;
    runtime_config.max_seq_len = 8192;
    runtime_config.parallel_basic_config.attn_tensor_parallel_size = 1;

    runtime_config_ = runtime_config;
    context_ = context;

    CreateIndexerMeta(/*compress_ratio=*/1);
  }

  void CreateIndexerMeta(int compress_ratio) {
    indexer_meta_ = std::make_unique<IndexerMeta>(runtime_config_, /*rank=*/0, context_, compress_ratio);
  }

  void BuildFlashInput(const std::vector<int>& extend_lens, const std::vector<int>& prefix_lens) {
    indexer_meta_->flash_meta.Reset();
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
      indexer_meta_->flash_meta.dp_reqs.push_back(&reqs_[i]);
      indexer_meta_->flash_meta.total_dp_input_ids_len += extend_lens[i];
    }
  }

  std::unique_ptr<IndexerMeta> indexer_meta_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<Context> context_;
  std::vector<ForwardRequest> reqs_;
};

TEST_F(IndexerMetaTest, PrepareChunk) {
  const int* token_starts;
  const int* kv_ranges;

  // Single req, no prefix → one chunk.
  BuildFlashInput(/*extend_lens=*/{5}, /*prefix_lens=*/{0});
  indexer_meta_->PrepareChunk(indexer_meta_->flash_meta);
  token_starts = indexer_meta_->chunk_token_starts_host.GetPtr<int>();
  kv_ranges = indexer_meta_->chunk_kv_token_ranges_host.GetPtr<int>();

  EXPECT_EQ(indexer_meta_->chunk_token_starts_host.shape[0], 2);
  EXPECT_EQ(indexer_meta_->chunk_kv_token_ranges_host.shape[0], 2);
  EXPECT_EQ(token_starts[0], 0);
  EXPECT_EQ(token_starts[1], 5);
  EXPECT_EQ(kv_ranges[0], 0);
  EXPECT_EQ(kv_ranges[1], 5);

  // Two reqs, no prefix → single chunk covering both.
  BuildFlashInput(/*extend_lens=*/{5, 3}, /*prefix_lens=*/{0, 0});
  indexer_meta_->PrepareChunk(indexer_meta_->flash_meta);
  token_starts = indexer_meta_->chunk_token_starts_host.GetPtr<int>();
  kv_ranges = indexer_meta_->chunk_kv_token_ranges_host.GetPtr<int>();

  EXPECT_EQ(indexer_meta_->chunk_token_starts_host.shape[0], 2);
  EXPECT_EQ(indexer_meta_->chunk_kv_token_ranges_host.shape[0], 2);
  EXPECT_EQ(token_starts[0], 0);
  EXPECT_EQ(token_starts[1], 8);
  EXPECT_EQ(kv_ranges[0], 0);
  EXPECT_EQ(kv_ranges[1], 8);

  // Prefix causes split when accumulated KV exceeds max_chunked_kv_tokens.
  runtime_config_.max_chunked_kv_tokens = 10;
  CreateIndexerMeta(/*compress_ratio=*/1);
  // req0: extend=5, prefix=0 → KV=5. req1: extend=3, prefix=6 → prefix=6 + req0 KV=5 = 11 ≥ 10 → split.
  BuildFlashInput(/*extend_lens=*/{5, 3}, /*prefix_lens=*/{0, 6});
  indexer_meta_->PrepareChunk(indexer_meta_->flash_meta);
  token_starts = indexer_meta_->chunk_token_starts_host.GetPtr<int>();
  kv_ranges = indexer_meta_->chunk_kv_token_ranges_host.GetPtr<int>();

  EXPECT_EQ(indexer_meta_->chunk_token_starts_host.shape[0], 3);
  EXPECT_EQ(indexer_meta_->chunk_kv_token_ranges_host.shape[0], 4);
  EXPECT_EQ(token_starts[0], 0);
  EXPECT_EQ(token_starts[1], 5);
  EXPECT_EQ(token_starts[2], 8);
  EXPECT_EQ(kv_ranges[0], 0);
  EXPECT_EQ(kv_ranges[1], 5);
  EXPECT_EQ(kv_ranges[2], 5);
  EXPECT_EQ(kv_ranges[3], 14);

  // Q token limit causes split even when KV budget is ample.
  runtime_config_.max_step_token_num = 12;
  runtime_config_.max_seq_len = 256;
  runtime_config_.max_chunked_kv_tokens = 100;
  CreateIndexerMeta(/*compress_ratio=*/1);
  // One request with 20 tokens, max Q per chunk = 12 → split into two chunks.
  BuildFlashInput(/*extend_lens=*/{20}, /*prefix_lens=*/{0});
  indexer_meta_->PrepareChunk(indexer_meta_->flash_meta);
  token_starts = indexer_meta_->chunk_token_starts_host.GetPtr<int>();
  kv_ranges = indexer_meta_->chunk_kv_token_ranges_host.GetPtr<int>();

  EXPECT_EQ(indexer_meta_->chunk_token_starts_host.shape[0], 3);
  EXPECT_EQ(indexer_meta_->chunk_kv_token_ranges_host.shape[0], 4);
  EXPECT_EQ(token_starts[0], 0);
  EXPECT_EQ(token_starts[1], 12);
  EXPECT_EQ(token_starts[2], 20);
  EXPECT_EQ(kv_ranges[0], 0);
  EXPECT_EQ(kv_ranges[1], 12);
  EXPECT_EQ(kv_ranges[2], 0);
  EXPECT_EQ(kv_ranges[3], 20);
}

TEST_F(IndexerMetaTest, PrepareBatchIndices) {
  // Shared device tensors accumulate write offsets across calls; reset to start from 0.
  indexer_meta_->batch_indices_tensor.shape = {0};

  // Single request → batch_indices are all 0.
  BuildFlashInput(/*extend_lens=*/{5}, /*prefix_lens=*/{0});
  indexer_meta_->PrepareBatchIndices(indexer_meta_->flash_meta);

  const auto& batch_indices = indexer_meta_->flash_meta.batch_indices_host;
  ASSERT_EQ(batch_indices.size(), 5);
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(batch_indices[i], 0);
  }

  // Multiple requests → batch_indices match each token's owning request.
  BuildFlashInput(/*extend_lens=*/{3, 2, 4}, /*prefix_lens=*/{0, 0, 0});
  indexer_meta_->PrepareBatchIndices(indexer_meta_->flash_meta);

  ASSERT_EQ(batch_indices.size(), 9);
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

TEST_F(IndexerMetaTest, PrepareCuSeqLen) {
  {
    // Flash, no chunking → each token's cu_seq_len starts at 0, end = position within request.
    indexer_meta_->cu_seq_len_start_tensor.shape = {0};
    indexer_meta_->cu_seq_len_end_tensor.shape = {0};
    BuildFlashInput(/*extend_lens=*/{3, 2}, /*prefix_lens=*/{0, 0});
    indexer_meta_->PrepareChunk(indexer_meta_->flash_meta);
    indexer_meta_->PrepareCuSeqLen(indexer_meta_->flash_meta);

    const auto& start = indexer_meta_->flash_meta.cu_seq_len_start_host;
    const auto& end = indexer_meta_->flash_meta.cu_seq_len_end_host;
    ASSERT_EQ(start.size(), 5);
    ASSERT_EQ(end.size(), 5);
    std::vector<int> expected_end = {1, 2, 3, 1, 2};
    for (size_t i = 0; i < 5; ++i) {
      EXPECT_EQ(start[i], 0) << "start i=" << i;
      EXPECT_EQ(end[i], expected_end[i]) << "end i=" << i;
    }
  }

  {
    // Flash with chunking → tokens after a split continue from their prefix position.
    runtime_config_.max_chunked_kv_tokens = 10;
    CreateIndexerMeta(/*compress_ratio=*/1);
    indexer_meta_->cu_seq_len_start_tensor.shape = {0};
    indexer_meta_->cu_seq_len_end_tensor.shape = {0};
    // req0: extend=3, prefix=0 → KV=3. req1: extend=2, prefix=8 → prefix alone (8) + req0 KV (3) = 11 > 10 → split.
    BuildFlashInput(/*extend_lens=*/{3, 2}, /*prefix_lens=*/{0, 8});
    indexer_meta_->PrepareChunk(indexer_meta_->flash_meta);
    indexer_meta_->PrepareCuSeqLen(indexer_meta_->flash_meta);

    const auto& start = indexer_meta_->flash_meta.cu_seq_len_start_host;
    const auto& end = indexer_meta_->flash_meta.cu_seq_len_end_host;
    ASSERT_EQ(start.size(), 5);
    ASSERT_EQ(end.size(), 5);
    std::vector<int> expected_end = {1, 2, 3, 9, 10};
    for (size_t i = 0; i < 5; ++i) {
      EXPECT_EQ(start[i], 0) << "start i=" << i;
      EXPECT_EQ(end[i], expected_end[i]) << "end i=" << i;
    }
  }

  // Paged (decode) path → cu_seq_len starts at 0, end = prefix + token position.
  {
    ForwardRequest req0, req1;
    req0.forwarding_tokens = std::make_shared<std::vector<int>>(5, 0);
    req0.kv_cached_token_num = 3;
    req0.prefix_cache_len = 3;
    req1.forwarding_tokens = std::make_shared<std::vector<int>>(10, 0);
    req1.kv_cached_token_num = 7;
    req1.prefix_cache_len = 7;

    IndexerMeta::input_meta page_meta;
    page_meta.dp_reqs = {&req0, &req1};
    page_meta.total_dp_input_ids_len = 5;

    indexer_meta_->PrepareCuSeqLen(page_meta, /*is_paged=*/true);

    const auto& paged_start = page_meta.cu_seq_len_start_host;
    const auto& paged_end = page_meta.cu_seq_len_end_host;
    ASSERT_EQ(paged_start.size(), 5);
    ASSERT_EQ(paged_end.size(), 5);
    std::vector<int> expected_paged_end = {4, 5, 8, 9, 10};
    for (size_t i = 0; i < 5; ++i) {
      EXPECT_EQ(paged_start[i], 0) << "start i=" << i;
      EXPECT_EQ(paged_end[i], expected_paged_end[i]) << "end i=" << i;
    }
  }
}

}  // namespace ksana_llm
