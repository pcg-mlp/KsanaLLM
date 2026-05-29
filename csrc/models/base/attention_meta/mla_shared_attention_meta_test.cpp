/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <filesystem>

#ifdef ENABLE_CUDA
#  include "device/nvidia/nccl_utils.h"
#endif
#include "configure/global_context.h"
#include "device/device_utils.h"
#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"
#include "models/base/attention_meta/mla_attention_meta_builder.h"
#include "tests/test.h"
#include "utils/singleton.h"

namespace ksana_llm {

class PrepareChunkKVTest : public testing::Test {
 protected:
  void SetUp() override {
    int rank = 0;

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
    model_config_.mla_config.kv_lora_rank = 512;
    model_config_.use_mla = true;
    env->SetModelConfig(model_config_);

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
    env->schedule_config_parser_.runtime_config_ = runtime_config_;
    runtime_config_.max_chunked_kv_tokens = 10;

    context_ = context;
    rank_ = rank;

    // Pre-allocate MLASharedAttentionMeta tensors (mirrors MlaMeta constructor).
    const size_t max_batch_size = runtime_config_.max_batch_size;
    mla_shared_.max_chunked_kv_tokens = static_cast<int>(runtime_config_.max_chunked_kv_tokens);
    mla_shared_.chunk_kv_batch_starts_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {max_batch_size + 1});
    mla_shared_.chunk_kv_cu_extend_lens_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {2 * max_batch_size});
    mla_shared_.chunk_kv_cu_seq_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {2 * max_batch_size});

    EventCreateWithFlags(&kvcache_offset_event_, EVENT_DISABLE_TIMING);
    EventCreateWithFlags(&rotary_embedding_event_, EVENT_DISABLE_TIMING);
  }

  AttentionBuilderConfig MakeConfig() {
    return AttentionBuilderConfig{model_config_,
                                  runtime_config_,
                                  /*enable_blocked_multi_token_forwarding_kv=*/false,
                                  /*use_flashinfer_for_decode=*/false,
                                  /*rank=*/rank_,
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

  void BuildFlashInput(const std::vector<int>& extend_lens, const std::vector<int>& prefix_lens) {
    flash_meta_.dp_reqs.clear();
    reqs_.clear();

    for (size_t i = 0; i < extend_lens.size(); ++i) {
      const int seq_len = extend_lens[i] + prefix_lens[i];
      ForwardRequest req;
      req.forwarding_tokens = std::make_shared<std::vector<int>>(seq_len, 0);
      req.prefix_cache_len = prefix_lens[i];
      reqs_.push_back(req);
    }

    for (size_t i = 0; i < reqs_.size(); ++i) {
      flash_meta_.dp_reqs.push_back(&reqs_[i]);
    }
  }

  MLASharedAttentionMeta mla_shared_;
  FlashAttentionMeta flash_meta_;
  std::shared_ptr<Context> context_;
  std::vector<ForwardRequest> reqs_;
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  std::vector<std::string> kv_cache_types_;
  std::unordered_map<std::string, std::vector<KvCacheLayerOffsets>> kv_cache_layer_offsets_;
  Event kvcache_offset_event_;
  Event rotary_embedding_event_;
  int rank_ = 0;
};

TEST_F(PrepareChunkKVTest, PrepareChunkKVTest) {
  auto config = MakeConfig();

  // Single req exactly at threshold fits in one chunk (no spurious split).
  BuildFlashInput(/*extend_lens=*/{10}, /*prefix_lens=*/{0});
  PrepareChunkKV(mla_shared_, flash_meta_, config);
  StreamSynchronize(context_->device->GetH2DStream());

  EXPECT_EQ(mla_shared_.chunk_kv_batch_starts_host.shape[0], 2);
  const int* batch_start_ptr = mla_shared_.chunk_kv_batch_starts_host.GetPtr<int>();
  EXPECT_EQ(batch_start_ptr[0], 0);
  EXPECT_EQ(batch_start_ptr[1], 1);

  EXPECT_EQ(mla_shared_.chunk_kv_cu_extend_lens_list, (std::vector<int>{0, 10}));
  EXPECT_EQ(mla_shared_.chunk_kv_cu_seq_lens_list, (std::vector<int>{0, 10}));
  EXPECT_EQ(mla_shared_.chunk_kv_cu_extend_lens_tensor.shape, (std::vector<size_t>{2}));
  EXPECT_EQ(mla_shared_.chunk_kv_cu_seq_lens_tensor.shape, (std::vector<size_t>{2}));

  // Two reqs summing exactly to threshold fit in the same chunk.
  BuildFlashInput(/*extend_lens=*/{5, 5}, /*prefix_lens=*/{0, 0});
  PrepareChunkKV(mla_shared_, flash_meta_, config);
  StreamSynchronize(context_->device->GetH2DStream());

  EXPECT_EQ(mla_shared_.chunk_kv_batch_starts_host.shape[0], 2);
  batch_start_ptr = mla_shared_.chunk_kv_batch_starts_host.GetPtr<int>();
  EXPECT_EQ(batch_start_ptr[0], 0);
  EXPECT_EQ(batch_start_ptr[1], 2);

  EXPECT_EQ(mla_shared_.chunk_kv_cu_extend_lens_list, (std::vector<int>{0, 5, 10}));
  EXPECT_EQ(mla_shared_.chunk_kv_cu_seq_lens_list, (std::vector<int>{0, 5, 10}));
  EXPECT_EQ(mla_shared_.chunk_kv_cu_extend_lens_tensor.shape, (std::vector<size_t>{3}));
  EXPECT_EQ(mla_shared_.chunk_kv_cu_seq_lens_tensor.shape, (std::vector<size_t>{3}));

  // Greedy packing with reqs [4,6,8,5,7] and prefixes [0,1,0,2,1].
  // KV tokens per req: 4, 7, 8, 7, 8. None can be packed within max_chunked_kv_tokens=10.
  BuildFlashInput(/*extend_lens=*/{4, 6, 8, 5, 7}, /*prefix_lens=*/{0, 1, 0, 2, 1});
  PrepareChunkKV(mla_shared_, flash_meta_, config);
  StreamSynchronize(context_->device->GetH2DStream());

  EXPECT_EQ(mla_shared_.chunk_kv_batch_starts_host.shape[0], 6);
  batch_start_ptr = mla_shared_.chunk_kv_batch_starts_host.GetPtr<int>();
  EXPECT_EQ(batch_start_ptr[0], 0);
  EXPECT_EQ(batch_start_ptr[1], 1);
  EXPECT_EQ(batch_start_ptr[2], 2);
  EXPECT_EQ(batch_start_ptr[3], 3);
  EXPECT_EQ(batch_start_ptr[4], 4);
  EXPECT_EQ(batch_start_ptr[5], 5);

  EXPECT_EQ(mla_shared_.chunk_kv_cu_extend_lens_list, (std::vector<int>{0, 4, 0, 6, 0, 8, 0, 5, 0, 7}));
  EXPECT_EQ(mla_shared_.chunk_kv_cu_seq_lens_list, (std::vector<int>{0, 4, 0, 7, 0, 8, 0, 7, 0, 8}));
  EXPECT_EQ(mla_shared_.chunk_kv_cu_extend_lens_tensor.shape, (std::vector<size_t>{10}));
  EXPECT_EQ(mla_shared_.chunk_kv_cu_seq_lens_tensor.shape, (std::vector<size_t>{10}));
}

}  // namespace ksana_llm
