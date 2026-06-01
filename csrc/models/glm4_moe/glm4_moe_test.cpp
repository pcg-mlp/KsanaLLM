/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <stdlib.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <set>
#include <vector>

#include "3rdparty/half/include/half.hpp"
#include "cache_manager/prefix_cache_manager.h"
#include "configure/json_config_utils.h"
#include "configure/schedule_config_parser.h"
#include "models/glm4_moe/glm4_moe_config.h"
#include "models/glm4_moe/glm4_moe_config_parser.h"
#include "models/glm4_moe/glm4_moe_model.h"
#include "models/glm4_moe/glm4_moe_weight_loader.h"
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
// GLM4-MoE Test Fixture
// ============================================================================

class Glm4MoeTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    AttentionBackendManager::GetInstance()->Initialize();
    env_ = Singleton<Environment>::GetInstance();
    const auto& env = env_;
    SetupWeightLoaderConfig();
    context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(1, 1, 1, env));
    context->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context->device = std::make_shared<DeviceContext>(1);
    context->device->ext->InitNcclParam(nccl_uid);

    std::string model_path = "/model/GLM4-MoE-Fake";
    std::string yaml_path = "../../../examples/llama7b/ksana_llm.yaml";

    const char* skip_env = std::getenv("SKIP_TEST_IF_MODEL_NOT_FOUND");
    if (skip_env != nullptr && std::string(skip_env) == "1" && !std::filesystem::exists(model_path)) {
      GTEST_SKIP() << "Model path " << model_path << " not found, skipping test.";
    }

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / yaml_path;
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    env->ParseConfig(config_path, model_path);
    env->UpdateModelConfig();
    env->GetModelConfig(model_config);

    BatchSchedulerConfig batch_scheduler_config;
    env->GetBatchSchedulerConfig(batch_scheduler_config);
    batch_scheduler_config.max_batch_size = 4;
    batch_scheduler_config.max_step_token_num = 128;
    batch_scheduler_config.max_token_len = 128;
    env->SetBatchSchedulerConfig(batch_scheduler_config);

    env->SetModelConfig(model_config);
    env->UpdateModelConfig();

    env->GetModelConfig(model_config);
    InitPipelineConfigForTest(env, model_config, /* enable_mtp */ false);

    block_cache_layout_ = GetTestModelCacheLayout(env_);
    env->InitializeBlockManagerConfig(block_cache_layout_);

    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.device_allocator_config.blocks_num = 32;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;
    env->SetBlockManagerConfig(block_manager_config);

    env->GetRuntimeConfig(runtime_config);
    env->GetPipelineConfig(pipeline_config);

    BlockAllocatorGroupConfig group_0_config;
    group_0_config.devices = {0};
    group_0_config.device_block_num = block_manager_config.device_allocator_config.blocks_num;
    group_0_config.host_block_num = block_manager_config.host_allocator_config.blocks_num;
    group_0_config.block_size = block_manager_config.device_allocator_config.block_size;

    BlockAllocatorManagerConfig block_allocator_manager_config;
    block_allocator_manager_config[0] = group_0_config;

    std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = std::make_shared<MemoryAllocator>();
    engine_block_manager =
        std::make_shared<BlockAllocatorManager>(true, block_allocator_manager_config, memory_allocator_, context);
    engine_block_group = engine_block_manager->GetBlockAllocatorGroup(0);
    executor_block_manager =
        std::make_shared<BlockAllocatorManager>(false, block_allocator_manager_config, memory_allocator_, context);
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
  std::shared_ptr<Glm4MoeConfig> weight_loader_config_;
  std::shared_ptr<Environment> env_;

  void SetupWeightLoaderConfig() {
    weight_loader_config_ = std::make_shared<Glm4MoeConfig>();
    weight_loader_config_->model_dir = "/model/test";
    weight_loader_config_->head_num = 8;
    weight_loader_config_->num_key_value_heads = 2;
    weight_loader_config_->hidden_units = 512;
    weight_loader_config_->size_per_head = 64;
    weight_loader_config_->num_layer = 2;
    weight_loader_config_->inter_size = 1024;
    weight_loader_config_->vocab_size = 2048;
    weight_loader_config_->is_moe = true;
    weight_loader_config_->has_shared_experts = true;
    weight_loader_config_->load_bias = true;
    weight_loader_config_->use_qk_norm = true;
    weight_loader_config_->enable_add_qkv_bias = true;
    weight_loader_config_->partial_rotary_factor = 0.5f;
    weight_loader_config_->moe_config.num_experts = 4;
    weight_loader_config_->moe_config.experts_topk = 2;
    weight_loader_config_->moe_config.moe_inter_size = 256;
    weight_loader_config_->moe_config.first_k_dense_replace = 1;
    weight_loader_config_->moe_config.num_shared_experts = 1;
    weight_loader_config_->moe_config.shared_expert_inter_size = 256;
    weight_loader_config_->moe_config.scoring_func = "sigmoid";
    weight_loader_config_->moe_config.topk_method = "fast";
    weight_loader_config_->moe_config.use_e_score_correction_bias = true;
    weight_loader_config_->moe_config.use_vllm_moe = true;
  }

  // Forward test with real model weights loaded from disk
  void TestGlm4MoeForward() {
    int device_id = 0;
    SetDevice(device_id);

    const std::string real_model_path = "/model/GLM4-MoE-Fake";
    if (!std::filesystem::exists(real_model_path)) {
      GTEST_SKIP() << "Model path does not exist: " << real_model_path;
    }

    Event start;
    Event stop;
    float milliseconds = 0;
    int rounds = 10;
    EventCreate(&start);
    EventCreate(&stop);

    // Load real weights from disk
    std::unique_ptr<WeightInstance> weight_instance =
        std::make_unique<WeightInstance>(model_config, runtime_config, context);
    weight_instance->Load();
    std::shared_ptr<BaseWeight> model_weight = weight_instance->GetWeight();
    std::shared_ptr<Glm4MoeModel> model = std::make_shared<Glm4MoeModel>(model_config, runtime_config, device_id,
                                                                         context, model_weight, block_cache_layout_);

    // ContextDecode forward
    auto forward = std::make_unique<ForwardRequest>();
    std::vector<int> input_ids = {233, 1203};
    forward->block_allocator_manager = executor_block_manager;
    forward->forwarding_tokens = std::make_shared<std::vector<int>>(input_ids);
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
    forward->flexible_cached_copy_tasks = &flexible_cached_copy_tasks;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    EmbeddingSlice embedding_slice;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;
    forward->input_refit_embedding = &embedding_slice;

    std::vector<int> block_ids;
    int use_block_num = (input_ids.size() + runtime_config.attn_backend_config.block_token_num - 1) /
                        runtime_config.attn_backend_config.block_token_num;
    engine_block_group->GetDeviceBlockAllocator(0)->AllocateBlocks(use_block_num, block_ids);
    executor_block_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, forward->kv_cache_ptrs);

    forward->atb_kv_cache_base_blk_ids.assign(1, {});
    BuildFlatKVCacheBlockIds(GetTotalLayerNum(model_config, pipeline_config), {block_ids},
                             forward->atb_kv_cache_base_blk_ids, executor_block_group->GetDeviceBlockAllocator(0));
    for (int block_idx = 0; block_idx < use_block_num; block_idx++) {
      Memset(forward->kv_cache_ptrs[block_idx], 0, runtime_config.attn_backend_config.block_size);
    }

    auto decode_forward = std::make_unique<ForwardRequest>(*forward);
    std::vector<int> decode_ids = input_ids;
    decode_forward->block_allocator_manager = executor_block_manager;
    decode_forward->forwarding_tokens = std::make_shared<std::vector<int>>(decode_ids);
    decode_forward->infer_stage = InferStage::kDecode;
    decode_forward->kv_cached_token_num = decode_forward->forwarding_tokens->size() - 1;
    std::vector<ForwardRequest> forward_reqs = {*forward, *decode_forward};
    EXPECT_TRUE(model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    // Performance test: ContextDecode
    auto forward_1 = std::make_unique<ForwardRequest>(*forward);
    auto forward_2 = std::make_unique<ForwardRequest>(*forward);
    std::vector<ForwardRequest> multi_forward_reqs = {*forward_1, *forward_2};
    for (int i = 0; i < rounds; ++i) {
      model->Forward(model_weight, multi_forward_reqs, hidden_unit, RunMode::kMain);
      StreamSynchronize(context->device->GetPrimaryComputeStream());
    }
    EventRecord(start, context->device->GetPrimaryComputeStream());
    for (int i = 0; i < rounds; ++i) {
      model->Forward(model_weight, multi_forward_reqs, hidden_unit, RunMode::kMain);
      StreamSynchronize(context->device->GetPrimaryComputeStream());
    }
    EventRecord(stop, context->device->GetPrimaryComputeStream());
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "GLM4MoE ContextDecode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

    // Sampling step 1
    SamplingRequest sample_req;
    NgramDict ngram_dict;
    auto logprobs = std::make_shared<std::vector<std::vector<std::pair<int, float>>>>();
    std::vector<float> prompt_probs;
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
    std::cout << fmt::format("GLM4MoE step1 generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0],
                             generated_tokens1[0])
              << std::endl;

    // Verify step 1 tokens (multiple valid values due to GPU differences)
    ASSERT_EQ(generated_tokens0.size(), 1u);
    ASSERT_EQ(generated_tokens1.size(), 1u);
    std::vector<int> valid_step1_tokens = {24923, 5595};
    EXPECT_TRUE(std::find(valid_step1_tokens.begin(), valid_step1_tokens.end(), generated_tokens0[0]) !=
                valid_step1_tokens.end())
        << "step1 generated_tokens0[0] = " << generated_tokens0[0] << " not in valid set";
    EXPECT_TRUE(std::find(valid_step1_tokens.begin(), valid_step1_tokens.end(), generated_tokens1[0]) !=
                valid_step1_tokens.end())
        << "step1 generated_tokens1[0] = " << generated_tokens1[0] << " not in valid set";

    // Sampling step 2: decode
    (*forward_reqs[0].forwarding_tokens).emplace_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).emplace_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto& forward_req : forward_reqs) {
      forward_req.infer_stage = InferStage::kDecode;
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }
    EXPECT_TRUE(model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());
    std::cout << fmt::format("GLM4MoE step2 generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0],
                             generated_tokens1[0])
              << std::endl;

    // Verify step 2 tokens
    ASSERT_EQ(generated_tokens0.size(), 1u);
    ASSERT_EQ(generated_tokens1.size(), 1u);
    std::vector<int> valid_step2_tokens = {4281, 7919};
    EXPECT_TRUE(std::find(valid_step2_tokens.begin(), valid_step2_tokens.end(), generated_tokens0[0]) !=
                valid_step2_tokens.end())
        << "step2 generated_tokens0[0] = " << generated_tokens0[0] << " not in valid set";
    EXPECT_TRUE(std::find(valid_step2_tokens.begin(), valid_step2_tokens.end(), generated_tokens1[0]) !=
                valid_step2_tokens.end())
        << "step2 generated_tokens1[0] = " << generated_tokens1[0] << " not in valid set";

    // Performance test: Decode
    EventRecord(start, context->device->GetPrimaryComputeStream());
    for (auto& forward_req : multi_forward_reqs) {
      forward_req.infer_stage = InferStage::kDecode;
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }
    for (int i = 0; i < rounds; ++i) {
      model->Forward(model_weight, multi_forward_reqs, hidden_unit, RunMode::kMain);
    }
    EventRecord(stop, context->device->GetPrimaryComputeStream());
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "GLM4MoE Decode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

    model.reset();
    model_weight.reset();

    StreamSynchronize(context->device->GetMemoryManageStream());
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
  }
};

// ============================================================================
// 1. Forward Test — end-to-end with real weights
// ============================================================================

TEST_F(Glm4MoeTest, ForwardTest) {
#ifdef ENABLE_CUDA
  TestGlm4MoeForward();
#endif
}

// ============================================================================
// 2. ConfigParser — full config parsing
// ============================================================================

TEST_F(Glm4MoeTest, ConfigParserBasicTest) {
#ifdef ENABLE_CUDA
  nlohmann::json config_json = {{"architectures", {"Glm4MoeForCausalLM"}},
                                {"bos_token_id", 1},
                                {"eos_token_id", {151329, 151336, 151338}},
                                {"pad_token_id", 151329},
                                {"hidden_size", 5120},
                                {"intermediate_size", 12288},
                                {"max_position_embeddings", 202752},
                                {"num_attention_heads", 96},
                                {"num_hidden_layers", 92},
                                {"num_key_value_heads", 8},
                                {"head_dim", 128},
                                {"rms_norm_eps", 1e-5},
                                {"rope_theta", 1000000.0},
                                {"tie_word_embeddings", false},
                                {"vocab_size", 151552},
                                {"partial_rotary_factor", 0.5},
                                {"use_qk_norm", true},
                                {"attention_bias", true},
                                {"n_routed_experts", 160},
                                {"num_experts_per_tok", 8},
                                {"moe_intermediate_size", 1536},
                                {"first_k_dense_replace", 3},
                                {"n_shared_experts", 1},
                                {"routed_scaling_factor", 2.5},
                                {"model_type", "glm4_moe"}};

  Glm4MoeConfigParser parser;
  ParallelismBasicConfig parallel_config;
  parallel_config.tensor_parallel_size = 1;
  std::shared_ptr<BaseModelConfig> base_config;
  Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());

  auto config = std::dynamic_pointer_cast<Glm4MoeConfig>(base_config);
  ASSERT_NE(config, nullptr);

  // Basic model params
  EXPECT_EQ(config->head_num, 96);
  EXPECT_EQ(config->num_key_value_heads, 8);
  EXPECT_EQ(config->hidden_units, 5120);
  EXPECT_EQ(config->inter_size, 12288);
  EXPECT_EQ(config->num_layer, 92);
  EXPECT_EQ(config->vocab_size, 151552);
  EXPECT_EQ(config->size_per_head, 128);
  EXPECT_FLOAT_EQ(config->rope_theta, 1000000.0f);
  EXPECT_FLOAT_EQ(config->layernorm_eps, 1e-5);
  EXPECT_EQ(config->start_id, 1);
  EXPECT_EQ(config->pad_id, 151329);
  EXPECT_EQ(config->max_position_embeddings, 202752);
  EXPECT_FALSE(config->tie_word_embeddings);

  // GLM4-MoE specific features
  EXPECT_TRUE(config->use_qk_norm);
  EXPECT_TRUE(config->enable_add_qkv_bias);
  EXPECT_TRUE(config->load_bias);
  EXPECT_FLOAT_EQ(config->partial_rotary_factor, 0.5f);
  EXPECT_EQ(config->rotary_embedding, 64);  // 128 * 0.5

  // MoE config
  EXPECT_TRUE(config->is_moe);
  EXPECT_TRUE(config->has_shared_experts);
  EXPECT_EQ(config->moe_config.num_experts, 160);
  EXPECT_EQ(config->moe_config.experts_topk, 8);
  EXPECT_EQ(config->moe_config.moe_inter_size, 1536);
  EXPECT_EQ(config->moe_config.first_k_dense_replace, 3);
  EXPECT_EQ(config->moe_config.num_shared_experts, 1);
  EXPECT_EQ(config->moe_config.scoring_func, "sigmoid");
  EXPECT_FLOAT_EQ(config->moe_config.routed_scaling_factor, 2.5f);
  EXPECT_TRUE(config->moe_config.use_e_score_correction_bias);
  EXPECT_TRUE(config->moe_config.use_vllm_moe);

  // Eos token IDs (array)
  ASSERT_EQ(config->end_ids.size(), 3u);
  EXPECT_EQ(config->end_ids[0], 151329);
  EXPECT_EQ(config->end_ids[1], 151336);
  EXPECT_EQ(config->end_ids[2], 151338);
  EXPECT_EQ(config->end_id, 151329);
#endif
}

// ============================================================================
// 3. ConfigParser — quantization configs (W4A8 + FP8 + NoQuant)
// ============================================================================

TEST_F(Glm4MoeTest, ConfigParserQuantTest) {
#ifdef ENABLE_CUDA
  Glm4MoeConfigParser parser;
  ParallelismBasicConfig parallel_config;
  parallel_config.tensor_parallel_size = 1;
  std::shared_ptr<BaseModelConfig> base_config;

  // W4A8_AWQ: RTN with input_scale=true
  {
    nlohmann::json config_json = {
        {"hidden_size", 5120},
        {"num_attention_heads", 96},
        {"num_hidden_layers", 92},
        {"vocab_size", 151552},
        {"quantization_config", {{"quant_method", "rtn"}, {"bits", 4}, {"group_size", 128}, {"input_scale", true}}}};
    Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
    EXPECT_TRUE(status.OK());
    auto config = std::dynamic_pointer_cast<Glm4MoeConfig>(base_config);
    ASSERT_NE(config, nullptr);
    EXPECT_TRUE(config->is_quant);
    EXPECT_EQ(config->quant_config.method, QUANT_W4A8_AWQ);
  }

  // Block FP8
  {
    nlohmann::json config_json = {
        {"hidden_size", 5120},
        {"num_attention_heads", 96},
        {"quantization_config",
         {{"quant_method", "fp8"}, {"activation_scheme", "dynamic"}, {"weight_block_size", {128, 128}}}}};
    Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
    EXPECT_TRUE(status.OK());
    auto config = std::dynamic_pointer_cast<Glm4MoeConfig>(base_config);
    ASSERT_NE(config, nullptr);
    EXPECT_TRUE(config->is_quant);
    EXPECT_EQ(config->quant_config.method, QUANT_BLOCK_FP8_E4M3);
  }

  // No quantization
  {
    nlohmann::json config_json = {
        {"hidden_size", 5120}, {"num_attention_heads", 96}, {"num_hidden_layers", 92}, {"vocab_size", 151552}};
    Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
    EXPECT_TRUE(status.OK());
    auto config = std::dynamic_pointer_cast<Glm4MoeConfig>(base_config);
    ASSERT_NE(config, nullptr);
    EXPECT_FALSE(config->is_quant);
  }
#endif
}

// ============================================================================
// 4. ConfigParser — GLM4-MoE specific fields (MTP, shared_expert_inter_size, eos edge cases)
// ============================================================================

TEST_F(Glm4MoeTest, ConfigParserMoeSpecificTest) {
#ifdef ENABLE_CUDA
  Glm4MoeConfigParser parser;
  ParallelismBasicConfig parallel_config;
  parallel_config.tensor_parallel_size = 1;
  std::shared_ptr<BaseModelConfig> base_config;

  // MTP layers
  {
    nlohmann::json config_json = {
        {"hidden_size", 5120}, {"num_attention_heads", 96}, {"num_hidden_layers", 92}, {"num_nextn_predict_layers", 1}};
    Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
    EXPECT_TRUE(status.OK());
    auto config = std::dynamic_pointer_cast<Glm4MoeConfig>(base_config);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->num_nextn_predict_layers, 1);
  }

  // shared_expert_inter_size = n_shared_experts * moe_intermediate_size
  {
    nlohmann::json config_json = {{"hidden_size", 5120},
                                  {"num_attention_heads", 96},
                                  {"n_routed_experts", 160},
                                  {"n_shared_experts", 1},
                                  {"moe_intermediate_size", 1536}};
    Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
    EXPECT_TRUE(status.OK());
    auto config = std::dynamic_pointer_cast<Glm4MoeConfig>(base_config);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->shared_expert_intermediate_size, 1536);
    EXPECT_EQ(config->moe_config.shared_expert_inter_size, 1536);
  }

  // eos_token_id: single value
  {
    nlohmann::json config_json = {
        {"hidden_size", 4096}, {"num_attention_heads", 32}, {"eos_token_id", 2}, {"model_type", "glm4_moe"}};
    Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
    EXPECT_TRUE(status.OK());
    auto config = std::dynamic_pointer_cast<Glm4MoeConfig>(base_config);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->end_id, 2);
    ASSERT_EQ(config->end_ids.size(), 1u);
  }

  // eos_token_id: missing (default)
  {
    nlohmann::json config_json = {{"hidden_size", 4096}, {"num_attention_heads", 32}, {"model_type", "glm4_moe"}};
    Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
    EXPECT_TRUE(status.OK());
    auto config = std::dynamic_pointer_cast<Glm4MoeConfig>(base_config);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->end_id, 2);  // default
  }
#endif
}

// ============================================================================
// 5. Config W4A8 helper methods (IsWeightMatchW4A8 + ContainW4A8Weights + IsW4A8Contain)
// ============================================================================

TEST_F(Glm4MoeTest, ConfigW4A8HelperTest) {
#ifdef ENABLE_CUDA
  // IsWeightMatchW4A8: primary W4A8
  {
    Glm4MoeConfig config;
    config.quant_config.method = QUANT_W4A8_AWQ;
    EXPECT_TRUE(config.IsWeightMatchW4A8("model.layers.0.mlp.experts.0.gate_proj.weight"));
    EXPECT_TRUE(config.IsWeightMatchW4A8("model.layers.0.self_attn.q_proj.weight"));
  }

  // IsWeightMatchW4A8: mixed quantization (FP8 primary + W4A8 sub for experts)
  {
    Glm4MoeConfig config;
    config.quant_config.method = QUANT_FP8_E4M3;
    QuantConfig sub_config;
    sub_config.method = QUANT_W4A8_AWQ;
    sub_config.pattern_layers = {".mlp.experts."};
    sub_config.ignored_layers = {"model.layers.92"};
    config.sub_quant_configs.push_back(sub_config);

    EXPECT_TRUE(config.IsWeightMatchW4A8("model.layers.0.mlp.experts.0.gate_proj.weight"));
    EXPECT_FALSE(config.IsWeightMatchW4A8("model.layers.92.mlp.experts.0.gate_proj.weight"));
    EXPECT_FALSE(config.IsWeightMatchW4A8("model.layers.0.self_attn.q_proj.weight"));
  }

  // ContainW4A8Weights
  {
    Glm4MoeConfig config;
    config.quant_config.method = QUANT_FP8_E4M3;
    EXPECT_FALSE(config.ContainW4A8Weights());
    config.quant_config.method = QUANT_W4A8_AWQ;
    EXPECT_TRUE(config.ContainW4A8Weights());

    Glm4MoeConfig config2;
    config2.quant_config.method = QUANT_FP8_E4M3;
    QuantConfig sub_config;
    sub_config.method = QUANT_GPTQ;
    config2.sub_quant_configs.push_back(sub_config);
    EXPECT_TRUE(config2.ContainW4A8Weights());
  }

  // IsW4A8Contain: weight name suffix matching
  {
    Glm4MoeConfig config;
    config.quant_config.method = QUANT_W4A8_AWQ;
    config.quant_config.input_scale = true;

    EXPECT_TRUE(config.IsW4A8Contain("model.layers.0.mlp.experts.0.gate_proj.qweight"));
    EXPECT_TRUE(config.IsW4A8Contain("model.layers.0.mlp.experts.0.gate_proj.scales"));
    EXPECT_TRUE(config.IsW4A8Contain("model.layers.0.mlp.experts.0.gate_proj.qzeros"));
    EXPECT_TRUE(config.IsW4A8Contain("model.layers.0.mlp.experts.0.gate_proj.input_scale"));
    EXPECT_TRUE(config.IsW4A8Contain("model.layers.0.mlp.experts.0.gate_proj.weight_scale"));
    EXPECT_FALSE(config.IsW4A8Contain("model.layers.0.mlp.experts.0.gate_proj.weight"));
  }
#endif
}

// ============================================================================
// 6. WeightLoader — filter weight names (skip_list + layer_range + keep_valid)
// ============================================================================

TEST_F(Glm4MoeTest, WeightLoaderFilterTest) {
#ifdef ENABLE_CUDA
  // Test skip list: shared_head.* should be filtered out
  {
    Glm4MoeWeightLoader weight_loader(weight_loader_config_, env_, context);
    std::vector<std::string> weight_names = {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate.weight",
        "shared_head.head.weight",
        "shared_head.norm.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    };
    EXPECT_TRUE(weight_loader.FilterWeightNames(weight_names).OK());
    for (const auto& name : weight_names) {
      EXPECT_EQ(name.find("shared_head"), std::string::npos) << "should filter: " << name;
    }
  }

  // Test layer range: only layers 0-1 kept, layers 2-3 filtered
  {
    PipelineConfig wl_pipeline_config;
    wl_pipeline_config.lower_layer_idx = 0;
    wl_pipeline_config.upper_layer_idx = 1;
    wl_pipeline_config.lower_nextn_layer_idx = -1;
    wl_pipeline_config.upper_nextn_layer_idx = -1;
    env_->SetPipelineConfig(wl_pipeline_config);

    Glm4MoeWeightLoader weight_loader(weight_loader_config_, env_, context);
    std::vector<std::string> weight_names = {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
    };
    EXPECT_TRUE(weight_loader.FilterWeightNames(weight_names).OK());

    bool found_layer_0 = false, found_layer_1 = false, found_layer_2 = false, found_embed = false;
    for (const auto& name : weight_names) {
      if (name.find("layers.0.") != std::string::npos) found_layer_0 = true;
      if (name.find("layers.1.") != std::string::npos) found_layer_1 = true;
      if (name.find("layers.2.") != std::string::npos) found_layer_2 = true;
      if (name.find("embed_tokens") != std::string::npos) found_embed = true;
    }
    EXPECT_TRUE(found_layer_0);
    EXPECT_TRUE(found_layer_1);
    EXPECT_FALSE(found_layer_2);
    EXPECT_TRUE(found_embed);
  }
#endif
}

// ============================================================================
// 7. WeightLoader — PreProcess (common + FP8)
// ============================================================================

TEST_F(Glm4MoeTest, WeightLoaderPreProcessTest) {
#ifdef ENABLE_CUDA
  // Common (non-quantized)
  {
    weight_loader_config_->is_quant = false;
    Glm4MoeWeightLoader weight_loader(weight_loader_config_, env_, context);
    std::unordered_map<std::string, Tensor> host_model_weights;
    EXPECT_TRUE(weight_loader.PreProcessModelWeights(host_model_weights).OK());
  }

  // FP8 quantized
  {
    weight_loader_config_->is_quant = true;
    weight_loader_config_->quant_config.method = QUANT_FP8_E4M3;
    Glm4MoeWeightLoader weight_loader(weight_loader_config_, env_, context);
    std::unordered_map<std::string, Tensor> host_model_weights;
    EXPECT_TRUE(weight_loader.PreProcessModelWeights(host_model_weights).OK());
  }
#endif
}

// ============================================================================
// 8. WeightLoader — base weights (norm + embed_tokens + lm_head + layernorms)
// ============================================================================

TEST_F(Glm4MoeTest, WeightLoaderBaseWeightsTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t vocab_size = 1024;

  weight_loader_config_->vocab_size = vocab_size;
  weight_loader_config_->hidden_units = hidden_units;

  Glm4MoeWeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  // model.norm.weight
  Tensor norm_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units});
  std::vector<half_float::half> norm_data(hidden_units, half_float::half(1.0f));
  std::memcpy(norm_tensor.GetPtr<void>(), norm_data.data(), norm_tensor.GetTotalBytes());
  host_model_weights["model.norm.weight"] = norm_tensor;

  // model.embed_tokens.weight
  Tensor embed_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {vocab_size, hidden_units});
  std::vector<half_float::half> embed_data(vocab_size * hidden_units, half_float::half(0.1f));
  std::memcpy(embed_tensor.GetPtr<void>(), embed_data.data(), embed_tensor.GetTotalBytes());
  host_model_weights["model.embed_tokens.weight"] = embed_tensor;

  // lm_head.weight
  Tensor lm_head_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {vocab_size, hidden_units});
  std::vector<half_float::half> lm_head_data(vocab_size * hidden_units, half_float::half(0.1f));
  std::memcpy(lm_head_tensor.GetPtr<void>(), lm_head_data.data(), lm_head_tensor.GetTotalBytes());
  host_model_weights["lm_head.weight"] = lm_head_tensor;

  // input_layernorm + post_attention_layernorm
  for (const auto& suffix : {".input_layernorm.weight", ".post_attention_layernorm.weight"}) {
    Tensor ln(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units});
    std::vector<half_float::half> ln_data(hidden_units, half_float::half(1.0f));
    std::memcpy(ln.GetPtr<void>(), ln_data.data(), ln.GetTotalBytes());
    host_model_weights[std::string("model.layers.0") + suffix] = ln;
  }

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(device_model_weights.find("model.norm.weight") != device_model_weights.end());
#endif
}

// ============================================================================
// 9. WeightLoader — attention weights (QKV proj + bias + O_proj + QK norm + dense MLP)
// ============================================================================

TEST_F(Glm4MoeTest, WeightLoaderAttentionWeightsTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t head_num = 8;
  const size_t head_dim = 64;
  const size_t kv_heads = 2;
  const size_t q_size = head_num * head_dim;
  const size_t kv_size = kv_heads * head_dim;
  const size_t inter_size = 1024;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->head_num = head_num;
  weight_loader_config_->num_key_value_heads = kv_heads;
  weight_loader_config_->size_per_head = head_dim;
  weight_loader_config_->inter_size = inter_size;

  Glm4MoeWeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  auto make_tensor = [](MemoryLocation loc, DataType dtype, std::vector<size_t> shape, float val) {
    Tensor t(loc, dtype, shape);
    size_t n = 1;
    for (auto s : shape) n *= s;
    std::vector<half_float::half> data(n, half_float::half(val));
    std::memcpy(t.GetPtr<void>(), data.data(), t.GetTotalBytes());
    return t;
  };

  const std::string lp = "model.layers.0";

  // QKV projections
  host_model_weights[lp + ".self_attn.q_proj.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {q_size, hidden_units}, 0.1f);
  host_model_weights[lp + ".self_attn.k_proj.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {kv_size, hidden_units}, 0.1f);
  host_model_weights[lp + ".self_attn.v_proj.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {kv_size, hidden_units}, 0.1f);

  // QKV biases
  host_model_weights[lp + ".self_attn.q_proj.bias"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {q_size}, 0.01f);
  host_model_weights[lp + ".self_attn.k_proj.bias"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {kv_size}, 0.01f);
  host_model_weights[lp + ".self_attn.v_proj.bias"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {kv_size}, 0.01f);

  // O_proj
  host_model_weights[lp + ".self_attn.o_proj.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, hidden_units}, 0.1f);

  // QK norm
  host_model_weights[lp + ".self_attn.query_layernorm.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {head_dim}, 1.0f);
  host_model_weights[lp + ".self_attn.key_layernorm.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {head_dim}, 1.0f);

  // Dense MLP (layer 0 < first_k_dense_replace=1)
  host_model_weights[lp + ".mlp.gate_proj.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {inter_size, hidden_units}, 0.1f);
  host_model_weights[lp + ".mlp.up_proj.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {inter_size, hidden_units}, 0.1f);
  host_model_weights[lp + ".mlp.down_proj.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, inter_size}, 0.1f);

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

// ============================================================================
// 10. WeightLoader — MoE weights (gate + routed experts + shared expert + e_score_correction_bias)
// ============================================================================

TEST_F(Glm4MoeTest, WeightLoaderMoeWeightsTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t moe_inter = 256;
  const size_t num_experts = 4;
  const size_t shared_expert_inter = 256;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->moe_config.moe_inter_size = moe_inter;
  weight_loader_config_->moe_config.num_experts = num_experts;
  weight_loader_config_->moe_config.shared_expert_inter_size = shared_expert_inter;

  Glm4MoeWeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  auto make_tensor = [](MemoryLocation loc, DataType dtype, std::vector<size_t> shape, float val) {
    Tensor t(loc, dtype, shape);
    size_t n = 1;
    for (auto s : shape) n *= s;
    if (dtype == TYPE_FP32) {
      std::vector<float> data(n, val);
      std::memcpy(t.GetPtr<void>(), data.data(), t.GetTotalBytes());
    } else {
      std::vector<half_float::half> data(n, half_float::half(val));
      std::memcpy(t.GetPtr<void>(), data.data(), t.GetTotalBytes());
    }
    return t;
  };

  const std::string lp = "model.layers.1";  // layer 1 is MoE (first_k_dense_replace=1)

  // Router gate
  host_model_weights[lp + ".mlp.gate.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, num_experts}, 0.1f);

  // e_score_correction_bias (FP32)
  host_model_weights[lp + ".mlp.gate.e_score_correction_bias"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {num_experts}, 0.1f);

  // Routed experts
  for (size_t e = 0; e < num_experts; ++e) {
    std::string ep = fmt::format("{}.mlp.experts.{}", lp, e);
    host_model_weights[ep + ".gate_proj.weight"] =
        make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {moe_inter, hidden_units}, 0.1f);
    host_model_weights[ep + ".up_proj.weight"] =
        make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {moe_inter, hidden_units}, 0.1f);
    host_model_weights[ep + ".down_proj.weight"] =
        make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, moe_inter}, 0.1f);
  }

  // Shared expert
  host_model_weights[lp + ".mlp.shared_expert.gate_proj.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {shared_expert_inter, hidden_units}, 0.1f);
  host_model_weights[lp + ".mlp.shared_expert.up_proj.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {shared_expert_inter, hidden_units}, 0.1f);
  host_model_weights[lp + ".mlp.shared_expert.down_proj.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, shared_expert_inter}, 0.1f);

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(device_model_weights.find(lp + ".mlp.gate.e_score_correction_bias") != device_model_weights.end());
#endif
}

// ============================================================================
// 11. WeightLoader — Constructor paths (W4A8, BlockFP8, enable_full_shared_expert)
// ============================================================================

TEST_F(Glm4MoeTest, WeightLoaderConstructorQuantPathsTest) {
#ifdef ENABLE_CUDA
  std::unordered_map<std::string, Tensor> host_model_weights;

  // W4A8 quant constructor + PreProcess
  {
    weight_loader_config_->is_quant = true;
    weight_loader_config_->quant_config.method = QUANT_W4A8_AWQ;
    weight_loader_config_->quant_config.enable_moe_int4 = true;
    Glm4MoeWeightLoader wl(weight_loader_config_, env_, context);
    EXPECT_TRUE(wl.PreProcessModelWeights(host_model_weights).OK());
  }

  // Block FP8 constructor
  {
    weight_loader_config_->quant_config.method = QUANT_BLOCK_FP8_E4M3;
    weight_loader_config_->quant_config.weight_block_size = {128, 128};
    Glm4MoeWeightLoader wl(weight_loader_config_, env_, context);
    EXPECT_TRUE(wl.PreProcessModelWeights(host_model_weights).OK());
  }

#endif
}

// ============================================================================
// 12. FilterWeightNames — nextn layer filtering and MTP embed_tokens skip
// ============================================================================

TEST_F(Glm4MoeTest, WeightLoaderFilterNextnLayersTest) {
#ifdef ENABLE_CUDA
  PipelineConfig wl_pipeline_config;
  wl_pipeline_config.lower_layer_idx = 0;
  wl_pipeline_config.upper_layer_idx = 1;
  wl_pipeline_config.lower_nextn_layer_idx = 2;
  wl_pipeline_config.upper_nextn_layer_idx = 2;
  env_->SetPipelineConfig(wl_pipeline_config);

  Glm4MoeWeightLoader weight_loader(weight_loader_config_, env_, context);

  std::vector<std::string> weight_names = {
      "model.layers.0.self_attn.q_proj.weight",  // keep: within [0,1]
      "model.layers.1.self_attn.q_proj.weight",  // keep: within [0,1]
      "model.layers.2.self_attn.q_proj.weight",  // keep: within nextn [2,2]
      "model.layers.2.embed_tokens.weight",      // skip: MTP embed_tokens
      "model.layers.3.self_attn.q_proj.weight",  // skip: outside both ranges
      "model.embed_tokens.weight",               // keep: main embed_tokens (layer_idx=-1)
  };

  EXPECT_TRUE(weight_loader.FilterWeightNames(weight_names).OK());

  std::set<std::string> remaining(weight_names.begin(), weight_names.end());
  EXPECT_TRUE(remaining.count("model.layers.0.self_attn.q_proj.weight"));
  EXPECT_TRUE(remaining.count("model.layers.1.self_attn.q_proj.weight"));
  EXPECT_TRUE(remaining.count("model.layers.2.self_attn.q_proj.weight"));
  EXPECT_FALSE(remaining.count("model.layers.2.embed_tokens.weight")) << "MTP embed_tokens should be filtered";
  EXPECT_FALSE(remaining.count("model.layers.3.self_attn.q_proj.weight")) << "layer 3 outside range should be filtered";
  EXPECT_TRUE(remaining.count("model.embed_tokens.weight")) << "main embed_tokens should NOT be filtered";
#endif
}

// ============================================================================
// 13. ProcessModelWeights — MTP eh_proj weight and scale loading
// ============================================================================

TEST_F(Glm4MoeTest, WeightLoaderMtpEhProjTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->num_nextn_predict_layers = 1;
  weight_loader_config_->num_layer = 2;

  Glm4MoeWeightLoader weight_loader(weight_loader_config_, env_, context);
  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  auto make_tensor = [](MemoryLocation loc, DataType dtype, std::vector<size_t> shape, float val) {
    Tensor t(loc, dtype, shape);
    size_t n = 1;
    for (auto s : shape) n *= s;
    if (dtype == TYPE_FP32) {
      std::vector<float> data(n, val);
      std::memcpy(t.GetPtr<void>(), data.data(), t.GetTotalBytes());
    } else {
      std::vector<half_float::half> data(n, half_float::half(val));
      std::memcpy(t.GetPtr<void>(), data.data(), t.GetTotalBytes());
    }
    return t;
  };

  // MTP layer index = num_layer = 2
  const std::string mtp_prefix = "model.layers.2";

  // eh_proj.weight (shape: [hidden_size, 2*hidden_size] for ColumnParallelLinear)
  host_model_weights[mtp_prefix + ".eh_proj.weight"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, 2 * hidden_units}, 0.1f);

  // eh_proj scales
  host_model_weights[mtp_prefix + ".eh_proj.weight_scale"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {1}, 1.0f);
  host_model_weights[mtp_prefix + ".eh_proj.input_scale"] =
      make_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {1}, 1.0f);

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());

  // eh_proj weight should be loaded with SplitOptTrans (ColumnParallel + transpose)
  EXPECT_TRUE(device_model_weights.count(mtp_prefix + ".eh_proj.weight")) << "MTP eh_proj.weight should be loaded";
  // eh_proj scales should be directly loaded
  EXPECT_TRUE(device_model_weights.count(mtp_prefix + ".eh_proj.weight_scale"))
      << "MTP eh_proj.weight_scale should be loaded";
  EXPECT_TRUE(device_model_weights.count(mtp_prefix + ".eh_proj.input_scale"))
      << "MTP eh_proj.input_scale should be loaded";
#endif
}

// ============================================================================
// 15. Config — W4A8 helpers (IsW4A8Method, GetW4A8QuantConfig, IsW4A8Contain)
// ============================================================================

TEST_F(Glm4MoeTest, ConfigW4A8HelpersTest) {
#ifdef ENABLE_CUDA
  // IsW4A8Method
  EXPECT_TRUE(Glm4MoeConfig::IsW4A8Method(QUANT_W4A8_AWQ));
  EXPECT_TRUE(Glm4MoeConfig::IsW4A8Method(QUANT_GPTQ));
  EXPECT_FALSE(Glm4MoeConfig::IsW4A8Method(QUANT_FP8_E4M3));
  EXPECT_FALSE(Glm4MoeConfig::IsW4A8Method(QUANT_BLOCK_FP8_E4M3));
  EXPECT_FALSE(Glm4MoeConfig::IsW4A8Method(QUANT_NONE));

  // GetW4A8QuantConfig — primary W4A8
  {
    Glm4MoeConfig config;
    config.quant_config.method = QUANT_W4A8_AWQ;
    config.quant_config.bits = 4;
    QuantConfig result = config.GetW4A8QuantConfig();
    EXPECT_EQ(result.method, QUANT_W4A8_AWQ);
  }

  // GetW4A8QuantConfig — from sub_quant_configs
  {
    Glm4MoeConfig config;
    config.quant_config.method = QUANT_FP8_E4M3;
    QuantConfig sub;
    sub.method = QUANT_GPTQ;
    config.sub_quant_configs.push_back(sub);
    EXPECT_EQ(config.GetW4A8QuantConfig().method, QUANT_GPTQ);
  }

  // GetW4A8QuantConfig — no W4A8 anywhere
  {
    Glm4MoeConfig config;
    config.quant_config.method = QUANT_FP8_E4M3;
    EXPECT_EQ(config.GetW4A8QuantConfig().method, QUANT_NONE);
  }

  // IsW4A8Contain — with input_scale=false
  {
    Glm4MoeConfig config;
    config.quant_config.method = QUANT_W4A8_AWQ;
    config.quant_config.input_scale = false;
    EXPECT_TRUE(config.IsW4A8Contain("model.layers.0.mlp.experts.0.gate_proj.qweight"));
    EXPECT_TRUE(config.IsW4A8Contain("model.layers.0.mlp.experts.0.gate_proj.scales"));
    EXPECT_TRUE(config.IsW4A8Contain("model.layers.0.mlp.experts.0.gate_proj.qzeros"));
    EXPECT_FALSE(config.IsW4A8Contain("model.layers.0.mlp.experts.0.gate_proj.input_scale"));
    EXPECT_FALSE(config.IsW4A8Contain("model.layers.0.mlp.experts.0.gate_proj.weight_scale"));
    EXPECT_FALSE(config.IsW4A8Contain("model.layers.0.mlp.experts.0.gate_proj.weight"));
  }
#endif
}
