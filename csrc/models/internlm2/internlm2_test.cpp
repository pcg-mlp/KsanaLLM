/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <stdlib.h>

#include <filesystem>
#include <memory>
#include <sstream>
#include <string>

#include "cache_manager/prefix_cache_manager.h"
#include "configure/json_config_utils.h"
#include "configure/schedule_config_parser.h"
#include "models/internlm2/internlm2_config.h"
#include "models/internlm2/internlm2_model.h"
#include "models/internlm2/new_internlm2_config_parser.h"
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

class Internlm2Test : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    parser_ = std::make_unique<Internlm2ConfigParser>();
    std::string model_path = "/model/internlm-2.5-7B-Chat";
    std::string yaml_path = "../../../examples/llama7b/ksana_llm.yaml";

    const char *skip_env = std::getenv("SKIP_TEST_IF_MODEL_NOT_FOUND");
    if (skip_env != nullptr && std::string(skip_env) == "1" && !std::filesystem::exists(model_path)) {
      GTEST_SKIP() << "Model path " << model_path << " not found, skipping test.";
    }

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / yaml_path;
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    AttentionBackendManager::GetInstance()->Initialize();
    const auto &env = Singleton<Environment>::GetInstance();
    context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(1, 1, 1, env));
    context->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context->device = std::make_shared<DeviceContext>(1);
    context->device->ext->InitNcclParam(nccl_uid);

    env->ParseConfig(config_path, model_path);
    env->UpdateModelConfig();
    env->GetModelConfig(model_config);
    InitPipelineConfigForTest(env, model_config);

    block_cache_layout_ = GetTestModelCacheLayout(env);
    env->InitializeBlockManagerConfig(block_cache_layout_);

    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);

    block_manager_config.device_allocator_config.blocks_num = 10;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;

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
    cache_manager_config.enable_prefix_caching = false;
    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, engine_block_group);

    hidden_unit_buffer_helper = std::make_shared<HiddenUnitBufferHelper>(env);
    hidden_unit = hidden_unit_buffer_helper->CreateDeviceBuffer();
  }

  void TearDown() override {}

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  PipelineConfig pipeline_config;
  std::shared_ptr<BlockAllocatorManager> engine_block_manager;
  std::shared_ptr<BlockAllocatorManager> executor_block_manager;
  std::shared_ptr<BlockAllocatorGroupInterface> engine_block_group;
  std::shared_ptr<BlockAllocatorGroupInterface> executor_block_group;
  std::shared_ptr<Context> context{nullptr};
  std::shared_ptr<HiddenUnitBufferHelper> hidden_unit_buffer_helper = nullptr;
  HiddenUnitDeviceBuffer *hidden_unit = nullptr;

  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;

  CacheManagerConfig cache_manager_config;
  std::shared_ptr<CacheManagerInterface> cache_manager = nullptr;
  std::unique_ptr<Internlm2ConfigParser> parser_;

  nlohmann::json CreateBaseConfig() {
    return {{"architectures", {"InternLM2ForCausalLM"}},
            {"bos_token_id", 1},
            {"eos_token_id", 2},
            {"pad_token_id", 2},
            {"hidden_size", 4096},
            {"intermediate_size", 14336},
            {"max_position_embeddings", 32768},
            {"num_attention_heads", 32},
            {"num_hidden_layers", 32},
            {"num_key_value_heads", 8},
            {"rms_norm_eps", 1e-5},
            {"rope_theta", 1000000},
            {"tie_word_embeddings", false},
            {"vocab_size", 92544},
            {"model_type", "internlm2"}};
  }

  std::shared_ptr<Internlm2Config> ParseConfig(const nlohmann::json &config_json) {
    ParallelismBasicConfig parallel_basic_config;
    parallel_basic_config.tensor_parallel_size = 1;
    std::shared_ptr<BaseModelConfig> base_model_config;
    Status status = parser_->ParseModelConfig(config_json, parallel_basic_config, "", base_model_config);
    EXPECT_TRUE(status.OK());
    return std::dynamic_pointer_cast<Internlm2Config>(base_model_config);
  }

  void TestInternlm2Forward() {
    int device_id = 0;
    SetDevice(device_id);
    std::filesystem::path model_path(model_config.path);
    if (!std::filesystem::exists(model_path)) {
      KLLM_LOG_ERROR << fmt::format("The given model path {} does not exist.", model_config.path);
      EXPECT_TRUE(std::filesystem::exists(model_path));
    }
    Event start;
    Event stop;
    float milliseconds = 0;
    int rounds = 10;
    EventCreate(&start);
    EventCreate(&stop);

    std::unique_ptr<WeightInstance> weight_instance =
        std::make_unique<WeightInstance>(model_config, runtime_config, context);
    weight_instance->Load();
    std::shared_ptr<BaseWeight> internlm2_weight = weight_instance->GetWeight();
    std::shared_ptr<Internlm2Model> internlm2 = std::make_shared<Internlm2Model>(
        model_config, runtime_config, device_id, context, internlm2_weight, block_cache_layout_);

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
      KLLM_LOG_DEBUG << fmt::format(
          "kv_cache_ptrs {} end {}", forward->kv_cache_ptrs[block_idx],
          forward->kv_cache_ptrs[block_idx] + (runtime_config.attn_backend_config.block_size));
    }

    auto decode_forward = std::make_unique<ForwardRequest>(*forward);
    std::vector<int> decode_ids = input_ids;
    decode_forward->block_allocator_manager = executor_block_manager;
    decode_forward->forwarding_tokens = std::make_shared<std::vector<int>>(decode_ids);
    decode_forward->infer_stage = InferStage::kDecode;
    decode_forward->kv_cached_token_num = decode_forward->forwarding_tokens->size() - 1;
    std::vector<ForwardRequest> forward_reqs = {*forward, *decode_forward};
    EXPECT_TRUE(internlm2->Forward(internlm2_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    auto forward_1 = std::make_unique<ForwardRequest>(*forward);
    auto forward_2 = std::make_unique<ForwardRequest>(*forward);
    std::vector<ForwardRequest> multi_forward_reqs = {*forward_1, *forward_2};
    for (int i = 0; i < rounds; ++i) {
      internlm2->Forward(internlm2_weight, multi_forward_reqs, hidden_unit, RunMode::kMain);
      StreamSynchronize(context->device->GetPrimaryComputeStream());
    }
    EventRecord(start, context->device->GetPrimaryComputeStream());
    for (int i = 0; i < rounds; ++i) {
      internlm2->Forward(internlm2_weight, multi_forward_reqs, hidden_unit, RunMode::kMain);
      StreamSynchronize(context->device->GetPrimaryComputeStream());
    }
    EventRecord(stop, context->device->GetPrimaryComputeStream());
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    KLLM_LOG_INFO << fmt::format("ContextDecode ms/iter ({} rounds): {}", rounds, milliseconds / rounds);

#ifdef ENABLE_CUDA
    EXPECT_TRUE((milliseconds / rounds) < 50);
#else
    EXPECT_TRUE((milliseconds / rounds) < 300) << "milliseconds / " << rounds << " is: " << milliseconds / rounds;
#endif

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
    sample_req.logits_buf = internlm2->GetLogitsPtr();
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
    decode_sample_req.logits_buf = internlm2->GetLogitsPtr();

    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);

    std::vector<SamplingRequest> sample_reqs = {sample_req, decode_sample_req};
    std::shared_ptr<Sampler> sampler = std::make_shared<Sampler>(batch_scheduler_config, device_id, context);
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());
    KLLM_LOG_INFO << fmt::format("generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0],
                                 generated_tokens1[0]);

    (*forward_reqs[0].forwarding_tokens).emplace_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).emplace_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req.infer_stage = InferStage::kDecode;
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }
    EXPECT_TRUE(internlm2->Forward(internlm2_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());
    KLLM_LOG_INFO << fmt::format("generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0],
                                 generated_tokens1[0]);

    (*forward_reqs[0].forwarding_tokens).emplace_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).emplace_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }

#ifdef ENABLE_CUDA
    EXPECT_TRUE(internlm2->Forward(internlm2_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());
    ASSERT_EQ(generated_tokens0[0], 300);
    ASSERT_EQ(generated_tokens1[0], 300);
    generated_tokens0.clear();
    generated_tokens1.clear();
#endif

    EventRecord(start, context->device->GetPrimaryComputeStream());
    for (auto &forward_req : multi_forward_reqs) {
      forward_req.infer_stage = InferStage::kDecode;
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }
    for (int i = 0; i < rounds; ++i) {
      internlm2->Forward(internlm2_weight, multi_forward_reqs, hidden_unit, RunMode::kMain);
    }
    EventRecord(stop, context->device->GetPrimaryComputeStream());
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    KLLM_LOG_INFO << fmt::format("Decode ms/iter ({} rounds): {}", rounds, milliseconds / rounds);

    internlm2.reset();
    internlm2_weight.reset();

    StreamSynchronize(context->device->GetMemoryManageStream());
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
  }
};

TEST_F(Internlm2Test, Internlm2ForwardTest) {
#ifdef ENABLE_CUDA
  TestInternlm2Forward();
#endif
}

TEST_F(Internlm2Test, ParseModelConfigBasic) {
  nlohmann::json config_with_quant = CreateBaseConfig();
  config_with_quant["quantization_config"] = {{"quant_method", "modelopt"},
                                              {"quant_algo", "W4A8_AWQ"},
                                              {"producer", {{"name", "modelopt"}, {"version", "0.33.1"}}}};

  auto internlm2_config = ParseConfig(config_with_quant);
  ASSERT_NE(internlm2_config, nullptr);

  EXPECT_EQ(internlm2_config->head_num, 32);
  EXPECT_EQ(internlm2_config->num_key_value_heads, 8);
  EXPECT_EQ(internlm2_config->inter_size, 14336);
  EXPECT_EQ(internlm2_config->vocab_size, 92544);
  EXPECT_EQ(internlm2_config->num_layer, 32);
  EXPECT_EQ(internlm2_config->hidden_units, 4096);
  EXPECT_FLOAT_EQ(internlm2_config->rope_theta, 1000000.0f);
  EXPECT_FLOAT_EQ(internlm2_config->layernorm_eps, 1e-5);
  EXPECT_EQ(internlm2_config->start_id, 1);
  EXPECT_EQ(internlm2_config->end_id, 2);
  EXPECT_EQ(internlm2_config->max_position_embeddings, 32768);
  EXPECT_FALSE(internlm2_config->tie_word_embeddings);

  EXPECT_TRUE(internlm2_config->is_quant);
  EXPECT_EQ(internlm2_config->quant_config.method, QUANT_W4A8_AWQ);

  nlohmann::json config_no_quant = CreateBaseConfig();
  auto internlm2_config_no_quant = ParseConfig(config_no_quant);
  EXPECT_FALSE(internlm2_config_no_quant->is_quant);
}

TEST_F(Internlm2Test, ParseModelConfigSpecialFields) {
  nlohmann::json config = CreateBaseConfig();
  config["head_dim"] = 128;
  auto internlm2_config = ParseConfig(config);
  EXPECT_EQ(internlm2_config->size_per_head, 128);

  config = CreateBaseConfig();
  internlm2_config = ParseConfig(config);
  EXPECT_EQ(internlm2_config->size_per_head, 4096 / 32);

  config = CreateBaseConfig();
  config["tie_word_embeddings"] = true;
  internlm2_config = ParseConfig(config);
  EXPECT_TRUE(internlm2_config->tie_word_embeddings);
  EXPECT_TRUE(internlm2_config->exist_tie_embeddings_param);

  config = CreateBaseConfig();
  config.erase("tie_word_embeddings");
  internlm2_config = ParseConfig(config);
  EXPECT_FALSE(internlm2_config->exist_tie_embeddings_param);
  EXPECT_FALSE(internlm2_config->tie_word_embeddings);
}

TEST_F(Internlm2Test, ParseQuantConfig) {
  nlohmann::json config_w4a8 = CreateBaseConfig();
  config_w4a8["quantization_config"] = {{"quant_method", "modelopt"}, {"quant_algo", "W4A8_AWQ"}};
  auto internlm2_config = ParseConfig(config_w4a8);
  EXPECT_TRUE(internlm2_config->is_quant);
  EXPECT_EQ(internlm2_config->quant_config.method, QUANT_W4A8_AWQ);

  nlohmann::json config_no_quant = CreateBaseConfig();
  auto config_no_quant_parsed = ParseConfig(config_no_quant);
  EXPECT_FALSE(config_no_quant_parsed->is_quant);

  nlohmann::json config_unsupported = CreateBaseConfig();
  config_unsupported["quantization_config"] = {{"quant_method", "unsupported"}};
  auto config_unsupported_parsed = ParseConfig(config_unsupported);
  EXPECT_TRUE(config_unsupported_parsed->is_quant);

  nlohmann::json config_bad_algo = CreateBaseConfig();
  config_bad_algo["quantization_config"] = {{"quant_method", "modelopt"}, {"quant_algo", "UNSUPPORTED"}};
  auto config_bad_algo_parsed = ParseConfig(config_bad_algo);
  EXPECT_TRUE(config_bad_algo_parsed->is_quant);
}

TEST_F(Internlm2Test, ParseModelConfigDefaultsAndComprehensive) {
  nlohmann::json minimal_config = {{"model_type", "internlm2"}};
  auto internlm2_config = ParseConfig(minimal_config);
  EXPECT_EQ(internlm2_config->head_num, 32);
  EXPECT_EQ(internlm2_config->vocab_size, 92544);
  EXPECT_FLOAT_EQ(internlm2_config->rope_theta, 1000000.0f);

  nlohmann::json comprehensive_config = {
      {"bos_token_id", 100},
      {"eos_token_id", 101},
      {"pad_token_id", 102},
      {"head_dim", 256},
      {"hidden_size", 2048},
      {"intermediate_size", 8192},
      {"max_position_embeddings", 16384},
      {"num_attention_heads", 16},
      {"num_hidden_layers", 24},
      {"num_key_value_heads", 4},
      {"rms_norm_eps", 1e-6},
      {"rope_theta", 10000.0},
      {"tie_word_embeddings", true},
      {"vocab_size", 50000},
      {"quantization_config", {{"quant_method", "modelopt"}, {"quant_algo", "W4A8_AWQ"}}}};

  internlm2_config = ParseConfig(comprehensive_config);
  EXPECT_EQ(internlm2_config->start_id, 100);
  EXPECT_EQ(internlm2_config->end_id, 101);
  EXPECT_EQ(internlm2_config->pad_id, 102);
  EXPECT_EQ(internlm2_config->size_per_head, 256);
  EXPECT_EQ(internlm2_config->hidden_units, 2048);
  EXPECT_EQ(internlm2_config->inter_size, 8192);
  EXPECT_EQ(internlm2_config->max_position_embeddings, 16384);
  EXPECT_EQ(internlm2_config->head_num, 16);
  EXPECT_EQ(internlm2_config->num_layer, 24);
  EXPECT_EQ(internlm2_config->num_key_value_heads, 4);
  EXPECT_FLOAT_EQ(internlm2_config->layernorm_eps, 1e-6);
  EXPECT_FLOAT_EQ(internlm2_config->rope_theta, 10000.0f);
  EXPECT_TRUE(internlm2_config->tie_word_embeddings);
  EXPECT_EQ(internlm2_config->vocab_size, 50000);
  EXPECT_TRUE(internlm2_config->is_quant);
  EXPECT_EQ(internlm2_config->quant_config.method, QUANT_W4A8_AWQ);
}
