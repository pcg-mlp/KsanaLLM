/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <stdlib.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <vector>

#include "3rdparty/half/include/half.hpp"
#include "cache_manager/prefix_cache_manager.h"
#include "configure/json_config_utils.h"
#include "configure/schedule_config_parser.h"
#include "models/chatglm/chatglm_config.h"
#include "models/chatglm/chatglm_model.h"
#include "nlohmann/json.hpp"
#include "runtime/hidden_unit_buffer.h"
#include "runtime/weight_instance.h"
#include "samplers/sampler.h"
#include "test.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/memory_allocator.h"
#include "utils/singleton.h"

#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"

using namespace ksana_llm;

// ============================================================================
// ChatGLM (GLM-4-9B-Chat) Test Fixture
// ============================================================================

class ChatglmTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    AttentionBackendManager::GetInstance()->Initialize();
    env_ = Singleton<Environment>::GetInstance();
    context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(1, 1, 1, env_));
    context->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context->device = std::make_shared<DeviceContext>(1);
    context->device->ext->InitNcclParam(nccl_uid);

    std::string model_path = "/model/GLM-4-9B-Chat";
    std::string yaml_path = "../../../examples/llama7b/ksana_llm.yaml";

    const char* skip_env = std::getenv("SKIP_TEST_IF_MODEL_NOT_FOUND");
    if (skip_env != nullptr && std::string(skip_env) == "1" && !std::filesystem::exists(model_path)) {
      GTEST_SKIP() << "Model path " << model_path << " not found, skipping test.";
    }

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / yaml_path;
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    env_->ParseConfig(config_path, model_path);
    env_->UpdateModelConfig();
    env_->GetModelConfig(model_config);
    InitPipelineConfigForTest(env_, model_config);

    BatchSchedulerConfig batch_scheduler_config;
    env_->GetBatchSchedulerConfig(batch_scheduler_config);
    batch_scheduler_config.max_batch_size = 4;
    batch_scheduler_config.max_step_token_num = 128;
    batch_scheduler_config.max_token_len = 128;
    env_->SetBatchSchedulerConfig(batch_scheduler_config);

    env_->SetModelConfig(model_config);
    env_->UpdateModelConfig();

    block_cache_layout_ = GetTestModelCacheLayout(env_);
    env_->InitializeBlockManagerConfig(block_cache_layout_);

    BlockManagerConfig block_manager_config;
    env_->GetBlockManagerConfig(block_manager_config);
    block_manager_config.device_allocator_config.blocks_num = 32;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;
    env_->SetBlockManagerConfig(block_manager_config);

    env_->GetRuntimeConfig(runtime_config);
    env_->GetPipelineConfig(pipeline_config);

    BlockAllocatorGroupConfig group_0_config;
    group_0_config.devices = {0};
    group_0_config.device_block_num = block_manager_config.device_allocator_config.blocks_num;
    group_0_config.host_block_num = block_manager_config.host_allocator_config.blocks_num;
    group_0_config.block_size = block_manager_config.device_allocator_config.block_size;

    BlockAllocatorManagerConfig block_allocator_manager_config;
    block_allocator_manager_config[0] = group_0_config;

    std::shared_ptr<MemoryAllocatorInterface> memory_allocator = std::make_shared<MemoryAllocator>();
    engine_block_manager =
        std::make_shared<BlockAllocatorManager>(true, block_allocator_manager_config, memory_allocator, context);
    engine_block_group = engine_block_manager->GetBlockAllocatorGroup(0);
    executor_block_manager =
        std::make_shared<BlockAllocatorManager>(false, block_allocator_manager_config, memory_allocator, context);
    executor_block_group = executor_block_manager->GetBlockAllocatorGroup(0);

    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = 1;
    cache_manager_config.swap_threadpool_size = 2;
    cache_manager_config.enable_prefix_caching = false;
    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, engine_block_group);

    hidden_unit = new HiddenUnitDeviceBuffer();
    hidden_unit->tensor = Tensor(MemoryLocation::LOCATION_DEVICE, model_config.weight_data_type,
                                 {runtime_config.max_step_token_num, static_cast<size_t>(model_config.hidden_units)});
  }

  void TearDown() override {
    delete hidden_unit;
    hidden_unit = nullptr;
  }

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  PipelineConfig pipeline_config;
  std::shared_ptr<BlockAllocatorManager> engine_block_manager;
  std::shared_ptr<BlockAllocatorManager> executor_block_manager;
  std::shared_ptr<BlockAllocatorGroupInterface> engine_block_group;
  std::shared_ptr<BlockAllocatorGroupInterface> executor_block_group;
  std::shared_ptr<Context> context{nullptr};
  HiddenUnitDeviceBuffer* hidden_unit = nullptr;

  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;

  CacheManagerConfig cache_manager_config;
  std::shared_ptr<CacheManagerInterface> cache_manager = nullptr;
  std::shared_ptr<Environment> env_;

  // Helper: build a ForwardRequest with the given input tokens
  std::unique_ptr<ForwardRequest> BuildForwardRequest(const std::vector<int>& input_ids,
                                                      std::vector<FlexibleCachedCopyTask>& flexible_tasks,
                                                      EmbeddingSlice& embedding_slice) {
    auto forward = std::make_unique<ForwardRequest>();
    forward->block_allocator_manager = executor_block_manager;
    forward->forwarding_tokens = std::make_shared<std::vector<int>>(input_ids);
    forward->flexible_cached_copy_tasks = &flexible_tasks;
    forward->input_refit_embedding = &embedding_slice;
    return forward;
  }

  // Helper: allocate KV cache blocks for a ForwardRequest
  void AllocateKVCacheBlocks(ForwardRequest& forward, size_t token_count) {
    std::vector<int> block_ids;
    int use_block_num = (token_count + runtime_config.attn_backend_config.block_token_num - 1) /
                        runtime_config.attn_backend_config.block_token_num;
    engine_block_group->GetDeviceBlockAllocator(0)->AllocateBlocks(use_block_num, block_ids);
    executor_block_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, forward.kv_cache_ptrs);

    forward.atb_kv_cache_base_blk_ids.assign(1, {});
    BuildFlatKVCacheBlockIds(GetTotalLayerNum(model_config, pipeline_config), block_ids,
                             forward.atb_kv_cache_base_blk_ids, executor_block_group->GetDeviceBlockAllocator(0));
    for (int block_idx = 0; block_idx < use_block_num; block_idx++) {
      Memset(forward.kv_cache_ptrs[block_idx], 0, runtime_config.attn_backend_config.block_size);
    }
  }

  // Forward test with real model weights loaded from disk
  void TestChatglmForward() {
    int device_id = 0;
    SetDevice(device_id);

    const std::string real_model_path = "/model/GLM-4-9B-Chat";
    if (!std::filesystem::exists(real_model_path)) {
      GTEST_SKIP() << "Model path does not exist: " << real_model_path;
    }

    // Load real weights from disk
    std::unique_ptr<WeightInstance> weight_instance =
        std::make_unique<WeightInstance>(model_config, runtime_config, context);
    weight_instance->Load();
    std::shared_ptr<BaseWeight> model_weight = weight_instance->GetWeight();

    std::shared_ptr<ChatglmModel> model = std::make_shared<ChatglmModel>(model_config, runtime_config, device_id,
                                                                         context, model_weight, block_cache_layout_);

    // ---- ContextDecode forward ----
    std::vector<int> input_ids = {233, 1203};
    std::vector<FlexibleCachedCopyTask> flexible_tasks;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    EmbeddingSlice embedding_slice;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;

    auto forward = BuildForwardRequest(input_ids, flexible_tasks, embedding_slice);
    AllocateKVCacheBlocks(*forward, input_ids.size());

    // Build a decode request sharing the same KV cache
    auto decode_forward = std::make_unique<ForwardRequest>(*forward);
    decode_forward->block_allocator_manager = executor_block_manager;
    decode_forward->forwarding_tokens = std::make_shared<std::vector<int>>(input_ids);
    decode_forward->infer_stage = InferStage::kDecode;
    decode_forward->kv_cached_token_num = decode_forward->forwarding_tokens->size() - 1;

    std::vector<ForwardRequest> forward_reqs = {*forward, *decode_forward};
    EXPECT_TRUE(model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    // ---- Sampling step 1 ----
    SamplingRequest sample_req;
    NgramDict ngram_dict;
    auto logprobs = std::make_shared<std::vector<std::vector<std::pair<int, float>>>>();
    std::vector<int> generated_tokens0, generated_tokens1;
    auto request_target = std::make_shared<std::unordered_map<std::string, TargetDescribe>>();

    sample_req.input_tokens = std::make_shared<std::vector<int>>(input_ids);
    sample_req.sampling_token_num = 1;
    sample_req.logits_offset = forward_reqs[0].logits_offset;
    sample_req.sampling_result_tokens = &generated_tokens0;
    sample_req.logprobs = logprobs;
    sample_req.ngram_dict = &ngram_dict;
    sample_req.logits_buf = model->GetLogitsPtr();

    SamplingConfig sample_config;
    sample_config.num_beams = 1;
    sample_config.topk = 1;
    sample_config.topp = 0;
    sample_config.temperature = 0;
    sample_config.repetition_penalty = 1;
    sample_config.no_repeat_ngram_size = 0;
    sample_config.encoder_no_repeat_ngram_size = 0;
    sample_req.sampling_config = &sample_config;
    sample_req.request_target = request_target;

    SamplingRequest decode_sample_req = sample_req;
    decode_sample_req.sampling_result_tokens = &generated_tokens1;
    decode_sample_req.logits_offset = forward_reqs[1].logits_offset;
    decode_sample_req.logits_buf = model->GetLogitsPtr();

    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);

    std::vector<SamplingRequest> sample_reqs = {sample_req, decode_sample_req};
    std::shared_ptr<Sampler> sampler = std::make_shared<Sampler>(batch_scheduler_config, device_id, context);
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());

    std::cout << fmt::format("ChatGLM step1 generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0],
                             generated_tokens1[0])
              << std::endl;

    ASSERT_EQ(generated_tokens0.size(), 1u);
    ASSERT_EQ(generated_tokens1.size(), 1u);
    EXPECT_GE(generated_tokens0[0], 0);
    EXPECT_LT(generated_tokens0[0], static_cast<int>(model_config.vocab_size));
    EXPECT_GE(generated_tokens1[0], 0);
    EXPECT_LT(generated_tokens1[0], static_cast<int>(model_config.vocab_size));

    // ---- Sampling step 2: auto-regressive decode ----
    (*forward_reqs[0].forwarding_tokens).emplace_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).emplace_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto& fwd : forward_reqs) {
      fwd.infer_stage = InferStage::kDecode;
      fwd.kv_cached_token_num = fwd.forwarding_tokens->size() - 1;
    }
    EXPECT_TRUE(model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());

    std::cout << fmt::format("ChatGLM step2 generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0],
                             generated_tokens1[0])
              << std::endl;

    ASSERT_EQ(generated_tokens0.size(), 1u);
    ASSERT_EQ(generated_tokens1.size(), 1u);
    EXPECT_GE(generated_tokens0[0], 0);
    EXPECT_LT(generated_tokens0[0], static_cast<int>(model_config.vocab_size));
    EXPECT_GE(generated_tokens1[0], 0);
    EXPECT_LT(generated_tokens1[0], static_cast<int>(model_config.vocab_size));

    // ---- Performance benchmark: ContextDecode ----
    Event start, stop;
    float milliseconds = 0;
    int rounds = 10;
    EventCreate(&start);
    EventCreate(&stop);

    auto fwd_perf1 = std::make_unique<ForwardRequest>(*forward);
    auto fwd_perf2 = std::make_unique<ForwardRequest>(*forward);
    std::vector<ForwardRequest> perf_reqs = {*fwd_perf1, *fwd_perf2};

    // Warmup
    for (int i = 0; i < rounds; ++i) {
      model->Forward(model_weight, perf_reqs, hidden_unit, RunMode::kMain);
      StreamSynchronize(context->device->GetPrimaryComputeStream());
    }

    EventRecord(start, context->device->GetPrimaryComputeStream());
    for (int i = 0; i < rounds; ++i) {
      model->Forward(model_weight, perf_reqs, hidden_unit, RunMode::kMain);
      StreamSynchronize(context->device->GetPrimaryComputeStream());
    }
    EventRecord(stop, context->device->GetPrimaryComputeStream());
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "ChatGLM ContextDecode avg ms / " << rounds << " is: " << milliseconds / rounds << std::endl;

    // ---- Performance benchmark: Decode ----
    for (auto& fwd : perf_reqs) {
      fwd.infer_stage = InferStage::kDecode;
      fwd.kv_cached_token_num = fwd.forwarding_tokens->size() - 1;
    }
    EventRecord(start, context->device->GetPrimaryComputeStream());
    for (int i = 0; i < rounds; ++i) {
      model->Forward(model_weight, perf_reqs, hidden_unit, RunMode::kMain);
    }
    EventRecord(stop, context->device->GetPrimaryComputeStream());
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "ChatGLM Decode avg ms / " << rounds << " is: " << milliseconds / rounds << std::endl;

    model.reset();
    model_weight.reset();

    StreamSynchronize(context->device->GetMemoryManageStream());
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
  }
};

// ============================================================================
// 1. Forward Test — end-to-end with real GLM-4-9B-Chat weights
// ============================================================================

TEST_F(ChatglmTest, ForwardTest) {
#ifdef ENABLE_CUDA
  TestChatglmForward();
#endif
}

// ============================================================================
// 2. Config Parsing — GLM-4-9B-Chat config.json fields
// ============================================================================

TEST_F(ChatglmTest, ConfigParseGlm4Test) {
#ifdef ENABLE_CUDA
  // Simulates the GLM-4-9B-Chat config.json
  nlohmann::json config_json = {{"num_attention_heads", 32},
                                {"multi_query_group_num", 2},
                                {"ffn_hidden_size", 13696},
                                {"hidden_size", 4096},
                                {"padded_vocab_size", 151552},
                                {"num_layers", 40},
                                {"layernorm_epsilon", 1.5625e-07},
                                {"rope_ratio", 500},
                                {"bos_token_id", 1},
                                {"eos_token_id", {151329, 151336, 151338}},
                                {"pad_token_id", 151329},
                                {"seq_length", 131072},
                                {"tie_word_embeddings", false},
                                {"add_qkv_bias", true}};

  ModelConfig mc;
  PrepareChatglmAttributes(config_json, mc);

  EXPECT_EQ(mc.head_num, 32u);
  EXPECT_EQ(mc.num_key_value_heads, 2u);  // multi_query_group_num
  EXPECT_EQ(mc.inter_size, 13696u);       // ffn_hidden_size
  EXPECT_EQ(mc.hidden_units, 4096u);
  EXPECT_EQ(mc.vocab_size, 151552u);  // padded_vocab_size
  EXPECT_EQ(mc.num_layer, 40u);
  EXPECT_FLOAT_EQ(mc.layernorm_eps, 1.5625e-07);
  EXPECT_FLOAT_EQ(mc.rope_theta, 500.0f * 10000.0f);  // rope_ratio * 10000
  EXPECT_EQ(mc.start_id, 1);
  EXPECT_EQ(mc.pad_id, 151329);
  EXPECT_EQ(mc.max_position_embeddings, 131072u);
  EXPECT_FALSE(mc.tie_word_embeddings);
  EXPECT_FALSE(mc.is_visual);

  // Derived fields
  EXPECT_EQ(mc.size_per_head, 128u);    // 4096 / 32
  EXPECT_EQ(mc.rotary_embedding, 64u);  // 128 / 2

  // EOS token array
  ASSERT_EQ(mc.end_ids.size(), 3u);
  EXPECT_EQ(mc.end_ids[0], 151329u);
  EXPECT_EQ(mc.end_ids[1], 151336u);
  EXPECT_EQ(mc.end_ids[2], 151338u);
#endif
}

// ============================================================================
// 3. Config Parsing — ChatGLM2/3 style (single eos, no padded_vocab_size)
// ============================================================================

TEST_F(ChatglmTest, ConfigParseChatglm2StyleTest) {
#ifdef ENABLE_CUDA
  nlohmann::json config_json = {{"num_attention_heads", 32}, {"multi_query_group_num", 2}, {"ffn_hidden_size", 13696},
                                {"hidden_size", 4096},       {"vocab_size", 65024},        {"num_layers", 28},
                                {"layernorm_epsilon", 1e-5}, {"eos_token_id", 2},          {"seq_length", 32768}};

  ModelConfig mc;
  PrepareChatglmAttributes(config_json, mc);

  // vocab_size should be 65024 (no padded_vocab_size override)
  EXPECT_EQ(mc.vocab_size, 65024u);
  EXPECT_EQ(mc.num_layer, 28u);
  // Single eos token
  ASSERT_EQ(mc.end_ids.size(), 1u);
  EXPECT_EQ(mc.end_ids[0], 2u);
  // rope_theta: default rope_ratio=1.0 * 10000
  EXPECT_FLOAT_EQ(mc.rope_theta, 10000.0f);
  EXPECT_EQ(mc.max_position_embeddings, 32768u);
#endif
}

// ============================================================================
// 4. Config Parsing — defaults when optional fields are missing
// ============================================================================

TEST_F(ChatglmTest, ConfigParseDefaultsTest) {
#ifdef ENABLE_CUDA
  // Minimal config: only required fields
  nlohmann::json config_json = {{"num_attention_heads", 16}, {"ffn_hidden_size", 8192}, {"hidden_size", 2048}};

  ModelConfig mc;
  PrepareChatglmAttributes(config_json, mc);

  // Default values when fields are missing
  EXPECT_EQ(mc.num_key_value_heads, 16u);         // defaults to head_num when multi_query_group_num absent
  EXPECT_EQ(mc.vocab_size, 65024u);               // default vocab_size
  EXPECT_EQ(mc.num_layer, 28u);                   // default num_layers
  EXPECT_FLOAT_EQ(mc.layernorm_eps, 1e-5);        // default layernorm_epsilon
  EXPECT_FLOAT_EQ(mc.rope_theta, 10000.0f);       // default rope_ratio=1.0 * 10000
  EXPECT_EQ(mc.start_id, 1);                      // default bos_token_id
  EXPECT_EQ(mc.pad_id, 0);                        // default pad_token_id
  EXPECT_EQ(mc.max_position_embeddings, 32768u);  // default seq_length
  EXPECT_FALSE(mc.tie_word_embeddings);           // default
  EXPECT_FALSE(mc.is_visual);                     // no "visual" key

  // Default eos when eos_token_id is absent
  ASSERT_EQ(mc.end_ids.size(), 1u);
  EXPECT_EQ(mc.end_ids[0], 2u);

  // Derived dimensions: 2048 / 16 = 128, 128 / 2 = 64
  EXPECT_EQ(mc.size_per_head, 128u);
  EXPECT_EQ(mc.rotary_embedding, 64u);
#endif
}

// ============================================================================
// 5. Model Config Consistency — verify parsed config from real model path
// ============================================================================

TEST_F(ChatglmTest, ModelConfigFromEnvironmentTest) {
#ifdef ENABLE_CUDA
  const std::string real_model_path = "/model/GLM-4-9B-Chat";
  if (!std::filesystem::exists(real_model_path)) {
    GTEST_SKIP() << "Model path does not exist: " << real_model_path;
  }

  // Verify model_config was correctly parsed from config.json via Environment
  EXPECT_EQ(model_config.head_num, 32u);
  EXPECT_EQ(model_config.num_key_value_heads, 2u);
  EXPECT_EQ(model_config.inter_size, 13696u);
  EXPECT_EQ(model_config.hidden_units, 4096u);
  EXPECT_EQ(model_config.vocab_size, 151552u);
  EXPECT_EQ(model_config.num_layer, 40u);
  EXPECT_EQ(model_config.size_per_head, 128u);
  EXPECT_EQ(model_config.rotary_embedding, 64u);
  EXPECT_EQ(model_config.max_position_embeddings, 131072u);
  EXPECT_FALSE(model_config.tie_word_embeddings);

  // EOS token IDs array (GLM-4 specific)
  ASSERT_EQ(model_config.end_ids.size(), 3u);
  EXPECT_EQ(model_config.end_ids[0], 151329u);
  EXPECT_EQ(model_config.end_ids[1], 151336u);
  EXPECT_EQ(model_config.end_ids[2], 151338u);

  // rope_theta: rope_ratio=500 * 10000 = 5000000
  EXPECT_FLOAT_EQ(model_config.rope_theta, 5000000.0f);
#endif
}

// ============================================================================
// 6. Multi-Step Autoregressive Generation — 3 consecutive decode steps
// ============================================================================

TEST_F(ChatglmTest, MultiStepGenerationTest) {
#ifdef ENABLE_CUDA
  const std::string real_model_path = "/model/GLM-4-9B-Chat";
  if (!std::filesystem::exists(real_model_path)) {
    GTEST_SKIP() << "Model path does not exist: " << real_model_path;
  }

  int device_id = 0;
  SetDevice(device_id);

  std::unique_ptr<WeightInstance> weight_instance =
      std::make_unique<WeightInstance>(model_config, runtime_config, context);
  weight_instance->Load();
  std::shared_ptr<BaseWeight> model_weight = weight_instance->GetWeight();

  auto model = std::make_shared<ChatglmModel>(model_config, runtime_config, device_id, context, model_weight,
                                              block_cache_layout_);

  std::vector<int> input_ids = {233, 1203};
  std::vector<FlexibleCachedCopyTask> flexible_tasks;
  std::vector<int> refit_pos;
  std::vector<std::vector<float>> refit_embed;
  EmbeddingSlice embedding_slice;
  embedding_slice.pos = refit_pos;
  embedding_slice.embeddings = refit_embed;

  auto forward = BuildForwardRequest(input_ids, flexible_tasks, embedding_slice);
  AllocateKVCacheBlocks(*forward, input_ids.size() + 5);  // extra space for generated tokens

  // Step 0: context decode
  std::vector<ForwardRequest> forward_reqs = {*forward};
  EXPECT_TRUE(model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
  StreamSynchronize(context->device->GetPrimaryComputeStream());

  // Set up sampler
  BatchSchedulerConfig batch_scheduler_config;
  Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
  auto sampler = std::make_shared<Sampler>(batch_scheduler_config, device_id, context);

  // Steps 1-3: autoregressive decode
  int num_decode_steps = 3;
  std::vector<int> all_generated;
  for (int step = 0; step < num_decode_steps; ++step) {
    SamplingRequest sample_req;
    NgramDict ngram_dict;
    auto logprobs = std::make_shared<std::vector<std::vector<std::pair<int, float>>>>();
    std::vector<int> generated_tokens;
    auto request_target = std::make_shared<std::unordered_map<std::string, TargetDescribe>>();

    sample_req.input_tokens = forward_reqs[0].forwarding_tokens;
    sample_req.sampling_token_num = 1;
    sample_req.logits_offset = forward_reqs[0].logits_offset;
    sample_req.sampling_result_tokens = &generated_tokens;
    sample_req.logprobs = logprobs;
    sample_req.ngram_dict = &ngram_dict;
    sample_req.logits_buf = model->GetLogitsPtr();
    sample_req.request_target = request_target;

    SamplingConfig sample_config;
    sample_config.num_beams = 1;
    sample_config.topk = 1;
    sample_config.topp = 0;
    sample_config.temperature = 0;
    sample_config.repetition_penalty = 1;
    sample_config.no_repeat_ngram_size = 0;
    sample_config.encoder_no_repeat_ngram_size = 0;
    sample_req.sampling_config = &sample_config;

    std::vector<SamplingRequest> sample_reqs = {sample_req};
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());

    ASSERT_EQ(generated_tokens.size(), 1u);
    EXPECT_GE(generated_tokens[0], 0);
    EXPECT_LT(generated_tokens[0], static_cast<int>(model_config.vocab_size));
    all_generated.push_back(generated_tokens[0]);

    std::cout << fmt::format("ChatGLM step {} generated token: {}", step + 1, generated_tokens[0]) << std::endl;

    // Prepare next decode forward
    forward_reqs[0].forwarding_tokens->push_back(generated_tokens[0]);
    forward_reqs[0].infer_stage = InferStage::kDecode;
    forward_reqs[0].kv_cached_token_num = forward_reqs[0].forwarding_tokens->size() - 1;
    EXPECT_TRUE(model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    StreamSynchronize(context->device->GetPrimaryComputeStream());
  }

  EXPECT_EQ(all_generated.size(), static_cast<size_t>(num_decode_steps));

  model.reset();
  model_weight.reset();
  DeviceSynchronize();
#endif
}
