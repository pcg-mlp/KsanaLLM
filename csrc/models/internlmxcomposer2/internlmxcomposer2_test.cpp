/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <stdlib.h>

#include <filesystem>
#include <memory>
#include <random>
#include <vector>

#include "3rdparty/half/include/half.hpp"
#include "cache_manager/prefix_cache_manager.h"
#include "configure/json_config_utils.h"
#include "configure/schedule_config_parser.h"
#include "models/base/fake_weight_for_test.h"
#include "models/internlmxcomposer2/internlmxcomposer2_config.h"
#include "models/internlmxcomposer2/internlmxcomposer2_model.h"
#include "models/internlmxcomposer2/new_internlmxcomposer2_config_parser.h"
#include "models/internlmxcomposer2/new_internlmxcomposer2_weight_loader.h"
#include "nlohmann/json.hpp"
#include "runtime/hidden_unit_buffer.h"
#include "samplers/sampler.h"
#include "test.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/memory_allocator.h"
#include "utils/singleton.h"

#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"

using namespace ksana_llm;

class InternlmxComposer2Test : public testing::Test {
 protected:
  static void BuildMockInternlmxComposer2Weights(FakeWeight& weight, const ModelConfig& model_config) {
    const DataType dtype = model_config.weight_data_type;
    const MemoryLocation location = MemoryLocation::LOCATION_DEVICE;

    const size_t hidden = model_config.hidden_units;
    const size_t inter = model_config.inter_size;
    const size_t vocab = model_config.vocab_size;
    const size_t head_num = model_config.head_num;
    const size_t num_kv_heads = model_config.num_key_value_heads;
    const size_t head_dim = model_config.size_per_head;
    const size_t qkv_out = (head_num + 2 * num_kv_heads) * head_dim;

    const size_t plora_rank = 8;

    weight.CreateWeight("model.embed_tokens.weight", {vocab, hidden}, dtype, location);
    weight.CreateWeight("lm_head.weight", {hidden, vocab}, dtype, location);
    weight.CreateWeight("model.norm.weight", {hidden}, dtype, location);

    for (uint32_t layer_idx = 0; layer_idx < model_config.num_layer; ++layer_idx) {
      const std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);
      weight.CreateWeight(layer_prefix + ".input_layernorm.weight", {hidden}, dtype, location);
      weight.CreateWeight(layer_prefix + ".post_attention_layernorm.weight", {hidden}, dtype, location);
      weight.CreateWeight(layer_prefix + ".self_attn.query_key_value.weight", {hidden, qkv_out}, dtype, location);
      weight.CreateWeight(layer_prefix + ".self_attn.o_proj.weight", {hidden, hidden}, dtype, location);
      weight.CreateWeight(layer_prefix + ".self_attn.query_layernorm.weight", {head_dim}, dtype, location);
      weight.CreateWeight(layer_prefix + ".self_attn.key_layernorm.weight", {head_dim}, dtype, location);
      weight.CreateWeight(layer_prefix + ".mlp.gate_proj.weight", {hidden, inter}, dtype, location);
      weight.CreateWeight(layer_prefix + ".mlp.up_proj.weight", {hidden, inter}, dtype, location);
      weight.CreateWeight(layer_prefix + ".mlp.down_proj.weight", {inter, hidden}, dtype, location);
      weight.CreateWeight(layer_prefix + ".mlp.gate_proj.Plora_A.weight", {hidden, plora_rank}, dtype, location);
      weight.CreateWeight(layer_prefix + ".mlp.gate_proj.Plora_B.weight", {plora_rank, inter}, dtype, location);
      weight.CreateWeight(layer_prefix + ".mlp.up_proj.Plora_A.weight", {hidden, plora_rank}, dtype, location);
      weight.CreateWeight(layer_prefix + ".mlp.up_proj.Plora_B.weight", {plora_rank, inter}, dtype, location);
      weight.CreateWeight(layer_prefix + ".mlp.down_proj.Plora_A.weight", {hidden, plora_rank}, dtype, location);
      weight.CreateWeight(layer_prefix + ".mlp.down_proj.Plora_B.weight", {plora_rank, hidden}, dtype, location);
      weight.CreateWeight(layer_prefix + ".self_attn.qkv_proj.Plora_A.weight", {hidden, plora_rank}, dtype, location);
      weight.CreateWeight(layer_prefix + ".self_attn.qkv_rproj.Plora_B.weight", {plora_rank, qkv_out}, dtype, location);
      weight.CreateWeight(layer_prefix + ".self_attn.o_proj.Plora_A.weight", {hidden, plora_rank}, dtype, location);
      weight.CreateWeight(layer_prefix + ".self_attn.o_proj.Plora_B.weight", {plora_rank, hidden}, dtype, location);
    }
  }

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

    std::string model_path = "/model/IXC-2.5-7B";
    std::string yaml_path = "../../../examples/llama7b/ksana_llm.yaml";

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / yaml_path;
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    env->ParseConfig(config_path, model_path);
    env->UpdateModelConfig();
    env->GetModelConfig(model_config);
    env->GetPipelineConfig(pipeline_config);

    nlohmann::json test_config_json = {{"architectures", {"InternLMXComposer2ForCausalLM"}},
                                       {"bos_token_id", 1},
                                       {"eos_token_id", 2},
                                       {"pad_token_id", 2},
                                       {"hidden_size", 512},
                                       {"intermediate_size", 2048},
                                       {"max_position_embeddings", 4096},
                                       {"num_attention_heads", 8},
                                       {"num_hidden_layers", 1},
                                       {"num_key_value_heads", 8},
                                       {"rms_norm_eps", 1e-5},
                                       {"rope_theta", 1000000},
                                       {"tie_word_embeddings", false},
                                       {"vocab_size", 2048},
                                       {"model_type", "internlm2"}};

    InternlmxComposer2ConfigParser config_parser;
    ParallelismBasicConfig parallel_basic_config;
    parallel_basic_config.tensor_parallel_size = 1;
    std::shared_ptr<BaseModelConfig> parsed_config;
    Status parse_status = config_parser.ParseModelConfig(test_config_json, parallel_basic_config, "", parsed_config);
    EXPECT_TRUE(parse_status.OK());

    auto ixc2_config = std::dynamic_pointer_cast<InternlmxComposer2Config>(parsed_config);
    EXPECT_NE(ixc2_config, nullptr);
    EXPECT_TRUE(ixc2_config->is_visual);
    EXPECT_TRUE(ixc2_config->has_plora);

    model_config.type = "internlmxcomposer2";
    model_config.num_layer = ixc2_config->num_layer;
    model_config.hidden_units = ixc2_config->hidden_units;
    model_config.head_num = ixc2_config->head_num;
    model_config.num_key_value_heads = ixc2_config->num_key_value_heads;
    model_config.size_per_head = ixc2_config->size_per_head;
    model_config.inter_size = ixc2_config->inter_size;
    model_config.vocab_size = ixc2_config->vocab_size;
    model_config.weight_data_type = TYPE_FP16;
    model_config.layernorm_eps = ixc2_config->layernorm_eps;
    InitPipelineConfigForTest(env, model_config);

    BatchSchedulerConfig batch_scheduler_config;
    env->GetBatchSchedulerConfig(batch_scheduler_config);
    batch_scheduler_config.max_batch_size = 4;
    batch_scheduler_config.max_step_token_num = 128;
    batch_scheduler_config.max_token_len = 128;
    env->SetBatchSchedulerConfig(batch_scheduler_config);

    env->SetModelConfig(model_config);
    env->UpdateModelConfig();

    block_cache_layout_ = GetTestModelCacheLayout(env);
    env->InitializeBlockManagerConfig(block_cache_layout_);

    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.device_allocator_config.blocks_num = 10;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;
    env->SetBlockManagerConfig(block_manager_config);

    env->GetRuntimeConfig(runtime_config);
    env->GetPipelineConfig(pipeline_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", runtime_config.attn_backend_config.block_size);

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
  std::shared_ptr<InternlmxComposer2Config> weight_loader_config_;
  std::shared_ptr<Environment> env_;

  void SetupWeightLoaderConfig() {
    weight_loader_config_ = std::make_shared<InternlmxComposer2Config>();
    weight_loader_config_->model_dir = "/model/test";
    weight_loader_config_->head_num = 32;
    weight_loader_config_->num_key_value_heads = 8;
    weight_loader_config_->hidden_units = 4096;
    weight_loader_config_->size_per_head = 128;
    weight_loader_config_->num_layer = 32;
    weight_loader_config_->inter_size = 14336;
    weight_loader_config_->vocab_size = 92544;
    weight_loader_config_->is_visual = true;
    weight_loader_config_->has_plora = true;
  }

  void TestInternlmxComposer2Forward() {
    int device_id = 0;
    SetDevice(device_id);
    Event start;
    Event stop;
    float milliseconds = 0;
    int rounds = 10;
    EventCreate(&start);
    EventCreate(&stop);

    auto fake_weight = std::make_shared<FakeWeight>(device_id);
    BuildMockInternlmxComposer2Weights(*fake_weight, model_config);
    std::shared_ptr<BaseWeight> model_weight = fake_weight;
    std::shared_ptr<InternlmxComposer2Model> model = std::make_shared<InternlmxComposer2Model>(
        model_config, runtime_config, device_id, context, model_weight, block_cache_layout_);

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
    EXPECT_TRUE(model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    StreamSynchronize(context->device->GetPrimaryComputeStream());

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
    std::cout << "ContextDecode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

#ifdef ENABLE_CUDA
    EXPECT_TRUE((milliseconds / rounds) < 10);
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
    std::cout << fmt::format("generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0], generated_tokens1[0])
              << std::endl;

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
    std::cout << fmt::format("generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0], generated_tokens1[0])
              << std::endl;

    (*forward_reqs[0].forwarding_tokens).emplace_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).emplace_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto& forward_req : forward_reqs) {
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }

#ifdef ENABLE_CUDA
    EXPECT_TRUE(model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());
    ASSERT_EQ(generated_tokens0.size(), 1u);
    ASSERT_EQ(generated_tokens0[0], 0);
    ASSERT_EQ(generated_tokens1[0], 0);
    std::cout << fmt::format("generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0], generated_tokens1[0])
              << std::endl;
    (*forward_reqs[0].forwarding_tokens).emplace_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).emplace_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
#endif

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
    std::cout << "Decode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

    model.reset();
    model_weight.reset();

    StreamSynchronize(context->device->GetMemoryManageStream());
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
  }
};

TEST_F(InternlmxComposer2Test, InternlmxComposer2ForwardTest) {
#ifdef ENABLE_CUDA
  TestInternlmxComposer2Forward();
#endif
}

TEST_F(InternlmxComposer2Test, ConfigParserBasicTest) {
#ifdef ENABLE_CUDA
  nlohmann::json config_json = {{"architectures", {"InternLMXComposer2ForCausalLM"}},
                                {"bos_token_id", 1},
                                {"eos_token_id", 2},
                                {"pad_token_id", 2},
                                {"hidden_size", 4096},
                                {"intermediate_size", 14336},
                                {"max_position_embeddings", 24576},
                                {"num_attention_heads", 32},
                                {"num_hidden_layers", 32},
                                {"num_key_value_heads", 8},
                                {"rms_norm_eps", 1e-5},
                                {"rope_theta", 1000000},
                                {"tie_word_embeddings", false},
                                {"vocab_size", 92544},
                                {"model_type", "internlm2"}};

  InternlmxComposer2ConfigParser parser;
  ParallelismBasicConfig parallel_config;
  parallel_config.tensor_parallel_size = 1;
  std::shared_ptr<BaseModelConfig> base_config;
  Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());

  auto config = std::dynamic_pointer_cast<InternlmxComposer2Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->head_num, 32);
  EXPECT_EQ(config->num_key_value_heads, 8);
  EXPECT_EQ(config->inter_size, 14336);
  EXPECT_EQ(config->vocab_size, 92544);
  EXPECT_EQ(config->num_layer, 32);
  EXPECT_EQ(config->hidden_units, 4096);
  EXPECT_FLOAT_EQ(config->rope_theta, 1000000.0f);
  EXPECT_FLOAT_EQ(config->layernorm_eps, 1e-5);
  EXPECT_EQ(config->start_id, 1);
  EXPECT_EQ(config->end_id, 2);
  EXPECT_EQ(config->pad_id, 2);
  EXPECT_EQ(config->max_position_embeddings, 24576);
  EXPECT_FALSE(config->tie_word_embeddings);
  EXPECT_TRUE(config->is_visual);
  EXPECT_TRUE(config->has_plora);
#endif
}

TEST_F(InternlmxComposer2Test, ConfigParserQuantW4A8AWQTest) {
#ifdef ENABLE_CUDA
  nlohmann::json config_json = {
      {"hidden_size", 4096},     {"num_attention_heads", 32},
      {"num_hidden_layers", 32}, {"num_key_value_heads", 8},
      {"vocab_size", 92544},     {"quantization_config", {{"quant_method", "modelopt"}, {"quant_algo", "W4A8_AWQ"}}}};

  InternlmxComposer2ConfigParser parser;
  ParallelismBasicConfig parallel_config;
  parallel_config.tensor_parallel_size = 1;
  std::shared_ptr<BaseModelConfig> base_config;
  Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());

  auto config = std::dynamic_pointer_cast<InternlmxComposer2Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_TRUE(config->is_quant);
  EXPECT_EQ(config->quant_config.method, QUANT_W4A8_AWQ);
#endif
}

TEST_F(InternlmxComposer2Test, ConfigParserNoQuantTest) {
#ifdef ENABLE_CUDA
  nlohmann::json config_json = {{"hidden_size", 4096},
                                {"num_attention_heads", 32},
                                {"num_hidden_layers", 32},
                                {"num_key_value_heads", 8},
                                {"vocab_size", 92544}};

  InternlmxComposer2ConfigParser parser;
  ParallelismBasicConfig parallel_config;
  parallel_config.tensor_parallel_size = 1;
  std::shared_ptr<BaseModelConfig> base_config;
  Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());

  auto config = std::dynamic_pointer_cast<InternlmxComposer2Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_FALSE(config->is_quant);
#endif
}

TEST_F(InternlmxComposer2Test, ConfigParserUnsupportedQuantMethodTest) {
#ifdef ENABLE_CUDA
  nlohmann::json config_json = {
      {"hidden_size", 4096}, {"num_attention_heads", 32}, {"quantization_config", {{"quant_method", "unsupported"}}}};

  InternlmxComposer2ConfigParser parser;
  ParallelismBasicConfig parallel_config;
  parallel_config.tensor_parallel_size = 1;
  std::shared_ptr<BaseModelConfig> base_config;
  Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());

  auto config = std::dynamic_pointer_cast<InternlmxComposer2Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_TRUE(config->is_quant);
#endif
}

TEST_F(InternlmxComposer2Test, ConfigParserUnsupportedQuantAlgoTest) {
#ifdef ENABLE_CUDA
  nlohmann::json config_json = {{"hidden_size", 4096},
                                {"num_attention_heads", 32},
                                {"quantization_config", {{"quant_method", "modelopt"}, {"quant_algo", "UNSUPPORTED"}}}};

  InternlmxComposer2ConfigParser parser;
  ParallelismBasicConfig parallel_config;
  parallel_config.tensor_parallel_size = 1;
  std::shared_ptr<BaseModelConfig> base_config;
  Status status = parser.ParseModelConfig(config_json, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());

  auto config = std::dynamic_pointer_cast<InternlmxComposer2Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_TRUE(config->is_quant);
#endif
}

TEST_F(InternlmxComposer2Test, ConfigParserHeadDimTest) {
#ifdef ENABLE_CUDA
  nlohmann::json config_with_head_dim = {{"hidden_size", 4096}, {"num_attention_heads", 32}, {"head_dim", 128}};

  InternlmxComposer2ConfigParser parser;
  ParallelismBasicConfig parallel_config;
  parallel_config.tensor_parallel_size = 1;
  std::shared_ptr<BaseModelConfig> base_config;
  Status status = parser.ParseModelConfig(config_with_head_dim, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());

  auto config = std::dynamic_pointer_cast<InternlmxComposer2Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->size_per_head, 128);

  nlohmann::json config_without_head_dim = {{"hidden_size", 4096}, {"num_attention_heads", 32}};
  status = parser.ParseModelConfig(config_without_head_dim, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());
  config = std::dynamic_pointer_cast<InternlmxComposer2Config>(base_config);
  EXPECT_EQ(config->size_per_head, 4096 / 32);
#endif
}

TEST_F(InternlmxComposer2Test, ConfigParserTieWordEmbeddingsTest) {
#ifdef ENABLE_CUDA
  InternlmxComposer2ConfigParser parser;
  ParallelismBasicConfig parallel_config;
  parallel_config.tensor_parallel_size = 1;
  std::shared_ptr<BaseModelConfig> base_config;

  nlohmann::json config_tie_true = {{"hidden_size", 4096}, {"num_attention_heads", 32}, {"tie_word_embeddings", true}};
  Status status = parser.ParseModelConfig(config_tie_true, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());
  auto config = std::dynamic_pointer_cast<InternlmxComposer2Config>(base_config);
  EXPECT_TRUE(config->tie_word_embeddings);
  EXPECT_TRUE(config->exist_tie_embeddings_param);

  nlohmann::json config_tie_false = {
      {"hidden_size", 4096}, {"num_attention_heads", 32}, {"tie_word_embeddings", false}};
  status = parser.ParseModelConfig(config_tie_false, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());
  config = std::dynamic_pointer_cast<InternlmxComposer2Config>(base_config);
  EXPECT_FALSE(config->tie_word_embeddings);

  nlohmann::json config_no_tie = {{"hidden_size", 4096}, {"num_attention_heads", 32}};
  status = parser.ParseModelConfig(config_no_tie, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());
  config = std::dynamic_pointer_cast<InternlmxComposer2Config>(base_config);
  EXPECT_FALSE(config->exist_tie_embeddings_param);
  EXPECT_FALSE(config->tie_word_embeddings);
#endif
}

TEST_F(InternlmxComposer2Test, ConfigParserDefaultsTest) {
#ifdef ENABLE_CUDA
  nlohmann::json minimal_config = {{"model_type", "internlm2"}};

  InternlmxComposer2ConfigParser parser;
  ParallelismBasicConfig parallel_config;
  parallel_config.tensor_parallel_size = 1;
  std::shared_ptr<BaseModelConfig> base_config;
  Status status = parser.ParseModelConfig(minimal_config, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());

  auto config = std::dynamic_pointer_cast<InternlmxComposer2Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->head_num, 32);
  EXPECT_EQ(config->vocab_size, 92544);
  EXPECT_FLOAT_EQ(config->rope_theta, 1000000.0f);
  EXPECT_TRUE(config->is_visual);
  EXPECT_TRUE(config->has_plora);
#endif
}

TEST_F(InternlmxComposer2Test, ConfigParserComprehensiveTest) {
#ifdef ENABLE_CUDA
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

  InternlmxComposer2ConfigParser parser;
  ParallelismBasicConfig parallel_config;
  parallel_config.tensor_parallel_size = 1;
  std::shared_ptr<BaseModelConfig> base_config;
  Status status = parser.ParseModelConfig(comprehensive_config, parallel_config, "", base_config);
  EXPECT_TRUE(status.OK());

  auto config = std::dynamic_pointer_cast<InternlmxComposer2Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->start_id, 100);
  EXPECT_EQ(config->end_id, 101);
  EXPECT_EQ(config->pad_id, 102);
  EXPECT_EQ(config->size_per_head, 256);
  EXPECT_EQ(config->hidden_units, 2048);
  EXPECT_EQ(config->inter_size, 8192);
  EXPECT_EQ(config->max_position_embeddings, 16384);
  EXPECT_EQ(config->head_num, 16);
  EXPECT_EQ(config->num_layer, 24);
  EXPECT_EQ(config->num_key_value_heads, 4);
  EXPECT_FLOAT_EQ(config->layernorm_eps, 1e-6);
  EXPECT_FLOAT_EQ(config->rope_theta, 10000.0f);
  EXPECT_TRUE(config->tie_word_embeddings);
  EXPECT_EQ(config->vocab_size, 50000);
  EXPECT_TRUE(config->is_quant);
  EXPECT_EQ(config->quant_config.method, QUANT_W4A8_AWQ);
  EXPECT_TRUE(config->is_visual);
  EXPECT_TRUE(config->has_plora);
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderFilterWeightNamesSkipListTest) {
#ifdef ENABLE_CUDA
  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::vector<std::string> weight_names = {
      "model.layers.0.self_attn.q_proj.weight",
      "model.layers.0.self_attn.k_proj.weight",
      "model.layers.0.self_attn.v_proj.weight",
      "model.layers.0.self_attn.rotary_emb.inv_freq",
      "model.layers.0.mlp.gate_proj.weight",
      "model.embed_tokens.weight",
      "lm_head.weight",
  };

  Status status = weight_loader.FilterWeightNames(weight_names);
  EXPECT_TRUE(status.OK());

  bool found_inv_freq = false;
  for (const auto& name : weight_names) {
    if (name.find("rotary_emb.inv_freq") != std::string::npos) {
      found_inv_freq = true;
      break;
    }
  }
  EXPECT_FALSE(found_inv_freq);
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderFilterWeightNamesLayerRangeTest) {
#ifdef ENABLE_CUDA
  PipelineConfig wl_pipeline_config;
  wl_pipeline_config.lower_layer_idx = 0;
  wl_pipeline_config.upper_layer_idx = 1;
  wl_pipeline_config.lower_nextn_layer_idx = -1;
  wl_pipeline_config.upper_nextn_layer_idx = -1;
  env_->SetPipelineConfig(wl_pipeline_config);

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::vector<std::string> weight_names = {
      "model.layers.0.self_attn.q_proj.weight",
      "model.layers.1.self_attn.q_proj.weight",
      "model.layers.2.self_attn.q_proj.weight",
      "model.layers.31.self_attn.q_proj.weight",
      "model.embed_tokens.weight",
      "lm_head.weight",
  };

  Status status = weight_loader.FilterWeightNames(weight_names);
  EXPECT_TRUE(status.OK());

  bool found_layer_2 = false;
  bool found_layer_31 = false;
  for (const auto& name : weight_names) {
    if (name.find("layers.2.") != std::string::npos) {
      found_layer_2 = true;
    }
    if (name.find("layers.31.") != std::string::npos) {
      found_layer_31 = true;
    }
  }
  EXPECT_FALSE(found_layer_2);
  EXPECT_FALSE(found_layer_31);
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderFilterWeightNamesKeepValidLayersTest) {
#ifdef ENABLE_CUDA
  PipelineConfig wl_pipeline_config;
  wl_pipeline_config.lower_layer_idx = 0;
  wl_pipeline_config.upper_layer_idx = 1;
  wl_pipeline_config.lower_nextn_layer_idx = -1;
  wl_pipeline_config.upper_nextn_layer_idx = -1;
  env_->SetPipelineConfig(wl_pipeline_config);

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::vector<std::string> weight_names = {
      "model.layers.0.self_attn.q_proj.weight",
      "model.layers.1.self_attn.q_proj.weight",
      "model.embed_tokens.weight",
  };

  Status status = weight_loader.FilterWeightNames(weight_names);
  EXPECT_TRUE(status.OK());

  bool found_layer_0 = false;
  bool found_layer_1 = false;
  bool found_embed = false;
  for (const auto& name : weight_names) {
    if (name.find("layers.0.") != std::string::npos) found_layer_0 = true;
    if (name.find("layers.1.") != std::string::npos) found_layer_1 = true;
    if (name.find("embed_tokens") != std::string::npos) found_embed = true;
  }
  EXPECT_TRUE(found_layer_0);
  EXPECT_TRUE(found_layer_1);
  EXPECT_TRUE(found_embed);
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderPreProcessModelWeightsCommonTest) {
#ifdef ENABLE_CUDA
  weight_loader_config_->is_quant = false;
  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  Status status = weight_loader.PreProcessModelWeights(host_model_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderPreProcessModelWeightsQuantW4A8AWQTest) {
#ifdef ENABLE_CUDA
  weight_loader_config_->is_quant = true;
  weight_loader_config_->quant_config.method = QUANT_W4A8_AWQ;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  Status status = weight_loader.PreProcessModelWeights(host_model_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsNormTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 4096;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor norm_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units});
  std::vector<half_float::half> norm_data(hidden_units, half_float::half(1.0f));
  std::memcpy(norm_tensor.GetPtr<void>(), norm_data.data(), norm_tensor.GetTotalBytes());
  host_model_weights["model.norm.weight"] = norm_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(device_model_weights.find("model.norm.weight") != device_model_weights.end());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsEmbedTokensTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t vocab_size = 1024;
  const size_t hidden_units = 512;

  weight_loader_config_->vocab_size = vocab_size;
  weight_loader_config_->hidden_units = hidden_units;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor embed_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {vocab_size, hidden_units});
  std::vector<half_float::half> embed_data(vocab_size * hidden_units, half_float::half(0.1f));
  std::memcpy(embed_tensor.GetPtr<void>(), embed_data.data(), embed_tensor.GetTotalBytes());
  host_model_weights["model.embed_tokens.weight"] = embed_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsTokEmbeddingsTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t vocab_size = 1024;
  const size_t hidden_units = 512;

  weight_loader_config_->vocab_size = vocab_size;
  weight_loader_config_->hidden_units = hidden_units;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor embed_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {vocab_size, hidden_units});
  std::vector<half_float::half> embed_data(vocab_size * hidden_units, half_float::half(0.1f));
  std::memcpy(embed_tensor.GetPtr<void>(), embed_data.data(), embed_tensor.GetTotalBytes());
  host_model_weights["model.tok_embeddings.weight"] = embed_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsLmHeadTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t vocab_size = 1024;
  const size_t hidden_units = 512;

  weight_loader_config_->vocab_size = vocab_size;
  weight_loader_config_->hidden_units = hidden_units;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor lm_head_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {vocab_size, hidden_units});
  std::vector<half_float::half> lm_head_data(vocab_size * hidden_units, half_float::half(0.1f));
  std::memcpy(lm_head_tensor.GetPtr<void>(), lm_head_data.data(), lm_head_tensor.GetTotalBytes());
  host_model_weights["lm_head.weight"] = lm_head_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsOutputWeightTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t vocab_size = 1024;
  const size_t hidden_units = 512;

  weight_loader_config_->vocab_size = vocab_size;
  weight_loader_config_->hidden_units = hidden_units;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor output_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {vocab_size, hidden_units});
  std::vector<half_float::half> output_data(vocab_size * hidden_units, half_float::half(0.1f));
  std::memcpy(output_tensor.GetPtr<void>(), output_data.data(), output_tensor.GetTotalBytes());
  host_model_weights["output.weight"] = output_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsPLoraATest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t plora_rank = 8;

  weight_loader_config_->hidden_units = hidden_units;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor plora_a_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, plora_rank});
  std::vector<half_float::half> plora_a_data(hidden_units * plora_rank, half_float::half(0.1f));
  std::memcpy(plora_a_tensor.GetPtr<void>(), plora_a_data.data(), plora_a_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.mlp.gate_proj.Plora_A.weight"] = plora_a_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(device_model_weights.find("model.layers.0.mlp.gate_proj.Plora_A.weight") != device_model_weights.end());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsPLoraBTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t plora_rank = 8;

  weight_loader_config_->hidden_units = hidden_units;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor plora_b_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {plora_rank, hidden_units});
  std::vector<half_float::half> plora_b_data(plora_rank * hidden_units, half_float::half(0.1f));
  std::memcpy(plora_b_tensor.GetPtr<void>(), plora_b_data.data(), plora_b_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.mlp.gate_proj.Plora_B.weight"] = plora_b_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(device_model_weights.find("model.layers.0.mlp.gate_proj.Plora_B.weight") != device_model_weights.end());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsAttnNormTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t head_dim = 128;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor query_norm_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {head_dim});
  std::vector<half_float::half> query_norm_data(head_dim, half_float::half(1.0f));
  std::memcpy(query_norm_tensor.GetPtr<void>(), query_norm_data.data(), query_norm_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.self_attn.query_layernorm.weight"] = query_norm_tensor;

  Tensor key_norm_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {head_dim});
  std::vector<half_float::half> key_norm_data(head_dim, half_float::half(1.0f));
  std::memcpy(key_norm_tensor.GetPtr<void>(), key_norm_data.data(), key_norm_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.self_attn.key_layernorm.weight"] = key_norm_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsInputLayerNormTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 4096;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor input_norm_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units});
  std::vector<half_float::half> input_norm_data(hidden_units, half_float::half(1.0f));
  std::memcpy(input_norm_tensor.GetPtr<void>(), input_norm_data.data(), input_norm_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.input_layernorm.weight"] = input_norm_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsPostAttnNormTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 4096;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor post_norm_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units});
  std::vector<half_float::half> post_norm_data(hidden_units, half_float::half(1.0f));
  std::memcpy(post_norm_tensor.GetPtr<void>(), post_norm_data.data(), post_norm_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.post_attention_layernorm.weight"] = post_norm_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsAttentionNormTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 4096;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor attn_norm_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units});
  std::vector<half_float::half> attn_norm_data(hidden_units, half_float::half(1.0f));
  std::memcpy(attn_norm_tensor.GetPtr<void>(), attn_norm_data.data(), attn_norm_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.attention_norm.weight"] = attn_norm_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsFfnNormTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 4096;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor ffn_norm_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units});
  std::vector<half_float::half> ffn_norm_data(hidden_units, half_float::half(1.0f));
  std::memcpy(ffn_norm_tensor.GetPtr<void>(), ffn_norm_data.data(), ffn_norm_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.ffn_norm.weight"] = ffn_norm_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsOProjTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->head_num = 8;
  weight_loader_config_->num_key_value_heads = 8;
  weight_loader_config_->size_per_head = 64;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor o_proj_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, hidden_units});
  std::vector<half_float::half> o_proj_data(hidden_units * hidden_units, half_float::half(0.1f));
  std::memcpy(o_proj_tensor.GetPtr<void>(), o_proj_data.data(), o_proj_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.self_attn.o_proj.weight"] = o_proj_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsWoTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->head_num = 8;
  weight_loader_config_->num_key_value_heads = 8;
  weight_loader_config_->size_per_head = 64;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor wo_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, hidden_units});
  std::vector<half_float::half> wo_data(hidden_units * hidden_units, half_float::half(0.1f));
  std::memcpy(wo_tensor.GetPtr<void>(), wo_data.data(), wo_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.self_attn.wo.weight"] = wo_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsMlpGateProjTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t inter_size = 2048;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->inter_size = inter_size;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor gate_proj_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {inter_size, hidden_units});
  std::vector<half_float::half> gate_proj_data(inter_size * hidden_units, half_float::half(0.1f));
  std::memcpy(gate_proj_tensor.GetPtr<void>(), gate_proj_data.data(), gate_proj_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.mlp.gate_proj.weight"] = gate_proj_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsMlpUpProjTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t inter_size = 2048;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->inter_size = inter_size;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor up_proj_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {inter_size, hidden_units});
  std::vector<half_float::half> up_proj_data(inter_size * hidden_units, half_float::half(0.1f));
  std::memcpy(up_proj_tensor.GetPtr<void>(), up_proj_data.data(), up_proj_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.mlp.up_proj.weight"] = up_proj_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsMlpDownProjTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t inter_size = 2048;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->inter_size = inter_size;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor down_proj_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, inter_size});
  std::vector<half_float::half> down_proj_data(hidden_units * inter_size, half_float::half(0.1f));
  std::memcpy(down_proj_tensor.GetPtr<void>(), down_proj_data.data(), down_proj_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.mlp.down_proj.weight"] = down_proj_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsMlpW1Test) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t inter_size = 2048;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->inter_size = inter_size;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor w1_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {inter_size, hidden_units});
  std::vector<half_float::half> w1_data(inter_size * hidden_units, half_float::half(0.1f));
  std::memcpy(w1_tensor.GetPtr<void>(), w1_data.data(), w1_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.mlp.w1.weight"] = w1_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsMlpW2Test) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t inter_size = 2048;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->inter_size = inter_size;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor w2_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, inter_size});
  std::vector<half_float::half> w2_data(hidden_units * inter_size, half_float::half(0.1f));
  std::memcpy(w2_tensor.GetPtr<void>(), w2_data.data(), w2_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.mlp.w2.weight"] = w2_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsMlpW3Test) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t inter_size = 2048;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->inter_size = inter_size;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor w3_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {inter_size, hidden_units});
  std::vector<half_float::half> w3_data(inter_size * hidden_units, half_float::half(0.1f));
  std::memcpy(w3_tensor.GetPtr<void>(), w3_data.data(), w3_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.mlp.w3.weight"] = w3_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsQProjTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t head_num = 8;
  const size_t head_dim = 64;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->head_num = head_num;
  weight_loader_config_->num_key_value_heads = head_num;
  weight_loader_config_->size_per_head = head_dim;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  const size_t q_size = head_num * head_dim;
  Tensor q_proj_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {q_size, hidden_units});
  std::vector<half_float::half> q_proj_data(q_size * hidden_units, half_float::half(0.1f));
  std::memcpy(q_proj_tensor.GetPtr<void>(), q_proj_data.data(), q_proj_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.self_attn.q_proj.weight"] = q_proj_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsKProjTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t num_kv_heads = 8;
  const size_t head_dim = 64;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->head_num = 8;
  weight_loader_config_->num_key_value_heads = num_kv_heads;
  weight_loader_config_->size_per_head = head_dim;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  const size_t kv_size = num_kv_heads * head_dim;
  Tensor k_proj_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {kv_size, hidden_units});
  std::vector<half_float::half> k_proj_data(kv_size * hidden_units, half_float::half(0.1f));
  std::memcpy(k_proj_tensor.GetPtr<void>(), k_proj_data.data(), k_proj_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.self_attn.k_proj.weight"] = k_proj_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsVProjTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 512;
  const size_t num_kv_heads = 8;
  const size_t head_dim = 64;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->head_num = 8;
  weight_loader_config_->num_key_value_heads = num_kv_heads;
  weight_loader_config_->size_per_head = head_dim;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  const size_t kv_size = num_kv_heads * head_dim;
  Tensor v_proj_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {kv_size, hidden_units});
  std::vector<half_float::half> v_proj_data(kv_size * hidden_units, half_float::half(0.1f));
  std::memcpy(v_proj_tensor.GetPtr<void>(), v_proj_data.data(), v_proj_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.self_attn.v_proj.weight"] = v_proj_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsInterleavedQKVTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 128;
  const size_t head_num = 4;
  const size_t num_kv_heads = 4;
  const size_t head_dim = hidden_units / head_num;
  const size_t qkv_out = (head_num + 2 * num_kv_heads) * head_dim;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->head_num = head_num;
  weight_loader_config_->num_key_value_heads = num_kv_heads;
  weight_loader_config_->size_per_head = head_dim;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor wqkv_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {qkv_out, hidden_units});
  std::vector<half_float::half> wqkv_data(qkv_out * hidden_units, half_float::half(0.1f));
  std::memcpy(wqkv_tensor.GetPtr<void>(), wqkv_data.data(), wqkv_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.self_attn.wqkv.weight"] = wqkv_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;
  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());

  auto it = device_model_weights.find("model.layers.0.self_attn.query_key_value.weight");
  ASSERT_TRUE(it != device_model_weights.end());
  EXPECT_EQ(it->second.shape[0], hidden_units);
  EXPECT_EQ(it->second.shape[1], qkv_out);
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderProcessModelWeightsInterleavedPLoraQKVTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 128;
  const size_t head_num = 4;
  const size_t num_kv_heads = 4;
  const size_t head_dim = hidden_units / head_num;
  const size_t qkv_out = (head_num + 2 * num_kv_heads) * head_dim;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->head_num = head_num;
  weight_loader_config_->num_key_value_heads = num_kv_heads;
  weight_loader_config_->size_per_head = head_dim;

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  const std::string weight_name = "model.layers.0.self_attn.qkv_rproj.Plora_B.weight";
  Tensor qkv_plora_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {qkv_out, hidden_units});
  std::vector<half_float::half> qkv_plora_data(qkv_out * hidden_units, half_float::half(0.1f));
  std::memcpy(qkv_plora_tensor.GetPtr<void>(), qkv_plora_data.data(), qkv_plora_tensor.GetTotalBytes());
  host_model_weights[weight_name] = qkv_plora_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;
  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());

  auto it = device_model_weights.find(weight_name);
  ASSERT_TRUE(it != device_model_weights.end());
  EXPECT_EQ(it->second.shape[0], qkv_out);
  EXPECT_EQ(it->second.shape[1], hidden_units);
#endif
}

TEST_F(InternlmxComposer2Test, WeightLoaderPostProcessModelWeightsTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  const size_t hidden_units = 128;
  const size_t inter_size = 256;
  const size_t vocab_size = 64;

  weight_loader_config_->hidden_units = hidden_units;
  weight_loader_config_->inter_size = inter_size;
  weight_loader_config_->vocab_size = vocab_size;
  weight_loader_config_->head_num = 4;
  weight_loader_config_->num_key_value_heads = 4;
  weight_loader_config_->size_per_head = hidden_units / weight_loader_config_->head_num;

  PipelineConfig wl_pipeline_config;
  env_->GetPipelineConfig(wl_pipeline_config);
  wl_pipeline_config.lower_layer_idx = 0;
  wl_pipeline_config.upper_layer_idx = 0;
  env_->SetPipelineConfig(wl_pipeline_config);

  NewInternlmxComposer2WeightLoader weight_loader(weight_loader_config_, env_, context);

  std::unordered_map<std::string, Tensor> host_model_weights;
  weight_loader.PreProcessModelWeights(host_model_weights);

  Tensor o_proj_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, hidden_units});
  std::vector<half_float::half> o_proj_data(hidden_units * hidden_units, half_float::half(0.1f));
  std::memcpy(o_proj_tensor.GetPtr<void>(), o_proj_data.data(), o_proj_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.self_attn.o_proj.weight"] = o_proj_tensor;

  Tensor gate_proj_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {inter_size, hidden_units});
  std::vector<half_float::half> gate_proj_data(inter_size * hidden_units, half_float::half(0.1f));
  std::memcpy(gate_proj_tensor.GetPtr<void>(), gate_proj_data.data(), gate_proj_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.mlp.gate_proj.weight"] = gate_proj_tensor;

  Tensor up_proj_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {inter_size, hidden_units});
  std::vector<half_float::half> up_proj_data(inter_size * hidden_units, half_float::half(0.1f));
  std::memcpy(up_proj_tensor.GetPtr<void>(), up_proj_data.data(), up_proj_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.mlp.up_proj.weight"] = up_proj_tensor;

  Tensor down_proj_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_units, inter_size});
  std::vector<half_float::half> down_proj_data(hidden_units * inter_size, half_float::half(0.1f));
  std::memcpy(down_proj_tensor.GetPtr<void>(), down_proj_data.data(), down_proj_tensor.GetTotalBytes());
  host_model_weights["model.layers.0.mlp.down_proj.weight"] = down_proj_tensor;

  Tensor lm_head_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {vocab_size, hidden_units});
  std::vector<half_float::half> lm_head_data(vocab_size * hidden_units, half_float::half(0.1f));
  std::memcpy(lm_head_tensor.GetPtr<void>(), lm_head_data.data(), lm_head_tensor.GetTotalBytes());
  host_model_weights["lm_head.weight"] = lm_head_tensor;

  std::unordered_map<std::string, Tensor> device_model_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;
  Status status =
      weight_loader.ProcessModelWeights(host_model_weights, kDeviceRank, device_model_weights, left_host_weights);
  EXPECT_TRUE(status.OK());

  Status post_status = weight_loader.PostProcessModelWeights(device_model_weights, kDeviceRank);
  EXPECT_TRUE(post_status.OK());
#endif
}
