/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <stdlib.h>
#include <filesystem>

#include "cache_manager/prefix_cache_manager.h"
#include "configure/schedule_config_parser.h"
#include "models/deepseek_v3/deepseek_v3_model.h"
#include "runtime/hidden_unit_buffer.h"
#include "runtime/weight_instance.h"
#include "samplers/sampler.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/memory_allocator.h"
#include "utils/singleton.h"

#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"
#include "test.h"

namespace ksana_llm {

class DeepSeekV3Test : public testing::Test {
 protected:
  void SetUp() override {
    setenv("W4AFP8_MOE_BACKEND", "0", 1);
    InitLoguru();
    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = test_info->name();
    std::string model_path = "/model/DeepSeek-R1-17832-fix-mtp";
    std::string yaml_path = "../../../examples/llama7b/ksana_llm.yaml";

    // Skip int4 output token check cause it's not stable.
    if (test_name.find("ForwardGPTQInt4Test") != std::string::npos) {
      model_path = "/model/DeepSeek-R1-17832-fix-mtp-bf16-w4g128-auto-gptq";
    } else if (test_name.find("ForwardMoeInt4Test") != std::string::npos) {
      model_path = "/model/DeepSeek-R1-0528-moe-int4-fix-mtp";
    } else if (test_name.find("SmallExpertsTest") != std::string::npos) {
      model_path = "/model/deepseek_v3";
      expected_tokens = {{3648, 303, 19892}};
    } else {
      if (test_name.find("ForwardW4AFP8Test") != std::string::npos) {
        model_path = "/model/DeepSeek-R1-0528-W4AFP8-mtpbfp8-venus-fix";
      }
      expected_tokens = {{5306, 13245, 15354}, /*without fastmath*/ {5306, 13245, 536}, {28570, 27932, 4180}};
    }

    const char* skip_env = std::getenv("SKIP_TEST_IF_MODEL_NOT_FOUND");
    if (skip_env != nullptr && std::string(skip_env) == "1" && !std::filesystem::exists(model_path)) {
      GTEST_SKIP() << "Model path " << model_path << " not found, skipping test.";
    }

    // 解析 config.json,初始化 ModelConfig 以及 BlockManager
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
    // TODO(robertyuan): bad style, remove later
    env->UpdateModelConfig();
    env->GetModelConfig(model_config);
    BatchSchedulerConfig batch_scheduler_config;
    env->GetBatchSchedulerConfig(batch_scheduler_config);
    batch_scheduler_config.mtp_step_num = model_config.num_nextn_predict_layers;
    batch_scheduler_config.enable_xgrammar = true;
    env->SetBatchSchedulerConfig(batch_scheduler_config);

    KLLM_LOG_INFO << "model_config.quant_config.method: " << model_config.quant_config.method;
    AttnBackendConfig attn_backend_config;
    env->GetAttnBackendConfig(attn_backend_config);
    attn_backend_config.enable_blocked_multi_token_forwarding_kv = true;
    env->SetAttnBackendConfig(attn_backend_config);
    BlockManagerConfig block_manager_config;

    InitPipelineConfigForTest(env, model_config);

    block_cache_layout_ = GetTestModelCacheLayout(env);
    env->InitializeBlockManagerConfig(block_cache_layout_);

    env->GetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);

    block_manager_config.device_allocator_config.blocks_num = 32;  // This test just need a few blocks;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;

    env->GetRuntimeConfig(runtime_config);
    runtime_config.mtp_step_num = batch_scheduler_config.mtp_step_num;

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

    CacheManagerConfig cache_manager_config;
    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = 1;
    cache_manager_config.swap_threadpool_size = 2;
    cache_manager_config.enable_prefix_caching = true;
    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, engine_block_group);

    hidden_unit_buffer_helper = std::make_shared<HiddenUnitBufferHelper>(env);
    hidden_unit = hidden_unit_buffer_helper->CreateDeviceBuffer();
  }

  void TearDown() override {
    unsetenv("W4AFP8_MOE_BACKEND");
    std::cout << "TearDown" << std::endl;
  }

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  PipelineConfig pipeline_config;
  std::shared_ptr<BlockAllocatorManager> engine_block_manager;
  std::shared_ptr<BlockAllocatorManager> executor_block_manager;
  std::shared_ptr<BlockAllocatorGroupInterface> engine_block_group;
  std::shared_ptr<BlockAllocatorGroupInterface> executor_block_group;
  std::shared_ptr<PrefixCacheManager> cache_manager = nullptr;
  std::shared_ptr<Context> context{nullptr};
  std::vector<std::vector<int>> expected_tokens;
  std::shared_ptr<HiddenUnitBufferHelper> hidden_unit_buffer_helper = nullptr;
  HiddenUnitDeviceBuffer *hidden_unit = nullptr;

  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;

  template <typename weight_data_type>
  void TestDeepSeekV3Forward() {
    int device_id = 0;
    SetDevice(device_id);
#ifdef ENABLE_FP8
    // fp8 is not supported
#endif
    std::filesystem::path model_path(model_config.path);
    if (!std::filesystem::exists(model_path)) {
      const char* skip_env = std::getenv("SKIP_TEST_IF_MODEL_NOT_FOUND");
      if (skip_env != nullptr && std::string(skip_env) == "1") {
        GTEST_SKIP() << "Model path " << model_config.path << " not found, skipping test.";
      }
      KLLM_LOG_ERROR << fmt::format("The given model path {} does not exist.", model_config.path);
      EXPECT_TRUE(std::filesystem::exists(model_path));
      return;
    }
    Event start;
    Event stop;
    float milliseconds = 0;
    constexpr int rounds = 10;
    EventCreate(&start);
    EventCreate(&stop);
    std::unique_ptr<WeightInstance> weight_instance =
        std::make_unique<WeightInstance>(model_config, runtime_config, context);
    weight_instance->Load();
    std::shared_ptr<BaseWeight> deepseek_v3_weight = weight_instance->GetWeight();
    std::shared_ptr<DeepSeekV3Model> deepseek_v3 = std::make_shared<DeepSeekV3Model>(
        model_config, runtime_config, device_id, context, deepseek_v3_weight, block_cache_layout_);

    // ContextDecode
    auto forward = std::make_unique<ForwardRequest>();
    std::vector<int> input_ids = {
        0,     0,     128803, 2788,  3655,   5979,   3099,  32200, 7624,  7524,   19,     16,     223,   1140,   2056,
        12519, 61320, 58788,  9090,  14721,  625,    303,   8040,  1612,  1049,   410,    31946,  303,   2788,   112467,
        718,   16227, 111162, 303,   1380,   32200,  8955,  7383,  10949, 20,     16,     223,    6094,  42257,  1261,
        40345, 34666, 525,    4385,  7624,   303,    13380, 41495, 718,   111162, 303,    3722,   2056,  422,    8673,
        2032,  54919, 2056,   1380,  6831,   9090,   303,   3722,  6525,  9090,   2032,   112467, 718,   16227,  111162,
        10949, 21,    16,     223,   100260, 2484,   8504,  2541,  34666, 65656,  121504, 654,    917,   2484,   9090,
        525,   19193, 34666,  7804,  303,    2541,   173,   241,   248,   548,    173,    241,    249,   36703,  902,
        34666, 65656, 21066,  4211,  34666,  7804,   303,   883,   1056,  19,     558,    34666,  64043, 173,    241,
        248,   19,    173,    241,   249,    303,    13097, 2032,  6831,  13850,  303,    23305,  19484, 1107,   50292,
        1847,  3722,  1530,   4385,  34666,  121386, 10626, 34666, 7804,  478,    22,     16,     223,   7624,   27095,
        7747,  7919,  16734,  271,   23,     16,     223,   9090,  974,   10209,  1735,   10655,  271,   122641, 7524,
        2556,  17288, 621,    4385,  34666,  271,    2792,  2130,  768,   939,    23,     15,     3425,  15,     3130,
        271,   2056,  768,    12183, 9617,   128804};
    forward->attn_dp_group_id = 0;
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
    BuildFlatKVCacheBlockIds(GetTotalLayerNum(model_config, pipeline_config), block_ids,
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
    forward->forwarding_tokens = std::make_shared<std::vector<int>>(decode_ids);
    decode_forward->infer_stage = InferStage::kDecode;
    decode_forward->kv_cached_token_num = decode_forward->forwarding_tokens->size() - 1;
    std::vector<ForwardRequest> forward_reqs = {*forward, *decode_forward};
    EXPECT_TRUE(deepseek_v3->Forward(deepseek_v3_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    auto forward_1 = std::make_unique<ForwardRequest>(*forward);
    auto forward_2 = std::make_unique<ForwardRequest>(*forward);
    std::vector<ForwardRequest> multi_forward_reqs = {*forward_1, *forward_2};
    // warmup
    for (int i = 0; i < rounds; ++i) {
      deepseek_v3->Forward(deepseek_v3_weight, multi_forward_reqs, hidden_unit, RunMode::kMain);
      StreamSynchronize(context->device->GetPrimaryComputeStream());
    }
    // test performance
    EventRecord(start, context->device->GetPrimaryComputeStream());
    for (int i = 0; i < rounds; ++i) {
      deepseek_v3->Forward(deepseek_v3_weight, multi_forward_reqs, hidden_unit, RunMode::kMain);
      StreamSynchronize(context->device->GetPrimaryComputeStream());
    }
    EventRecord(stop, context->device->GetPrimaryComputeStream());
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "ContextDecode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

    if (model_config.quant_config.method == QUANT_BLOCK_FP8_E4M3 && !model_config.quant_config.enable_moe_int4) {
      EXPECT_TRUE((milliseconds / rounds) < 20);
    }

    // Sampling
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
    sample_req.logits_buf = deepseek_v3->GetLogitsPtr();
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
    decode_sample_req.logits_buf = deepseek_v3->GetLogitsPtr();

    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);

    std::vector<SamplingRequest> sample_reqs = {sample_req, decode_sample_req};
    std::shared_ptr<Sampler> sampler = std::make_shared<Sampler>(batch_scheduler_config, device_id, context);
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());
    std::cout << fmt::format("Only sampling: generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0],
                             generated_tokens1[0])
              << std::endl;
    // Check if generated tokens match any group of expected tokens
    bool match0 = false, match1 = false;
    for (size_t i = 0; i < expected_tokens.size(); ++i) {
      if (generated_tokens0[0] == expected_tokens[i][0]) match0 = true;
      if (generated_tokens1[0] == expected_tokens[i][0]) match1 = true;
    }
    EXPECT_TRUE(match0 || expected_tokens.empty());
    EXPECT_TRUE(match1 || expected_tokens.empty());

    // Decode
    (*forward_reqs[0].forwarding_tokens).emplace_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).emplace_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req.infer_stage = InferStage::kDecode;
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }
    EXPECT_TRUE(deepseek_v3->Forward(deepseek_v3_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());
    std::cout << fmt::format("Main forward: generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0],
                             generated_tokens1[0])
              << std::endl;
    // Check if generated tokens match any group of expected tokens
    match0 = false;
    match1 = false;
    for (size_t i = 0; i < expected_tokens.size(); ++i) {
      if (generated_tokens0[0] == expected_tokens[i][1]) match0 = true;
      if (generated_tokens1[0] == expected_tokens[i][1]) match1 = true;
    }
    EXPECT_TRUE(match0 || expected_tokens.empty());
    EXPECT_TRUE(match1 || expected_tokens.empty());

    (*forward_reqs[0].forwarding_tokens).emplace_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).emplace_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }

    EventRecord(start, context->device->GetPrimaryComputeStream());
    for (auto &forward_req : multi_forward_reqs) {
      forward_req.infer_stage = InferStage::kDecode;
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }
    for (int i = 0; i < rounds; ++i) {
      deepseek_v3->Forward(deepseek_v3_weight, multi_forward_reqs, hidden_unit, RunMode::kMain);
    }
    EventRecord(stop, context->device->GetPrimaryComputeStream());
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "Decode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;
    EXPECT_TRUE((milliseconds / rounds) < 11);

    // MTP
    for (auto &forward_req : multi_forward_reqs) {
      forward_req.infer_stage = InferStage::kContext;
      forward_req.kv_cached_token_num = 0;
    }
    EXPECT_TRUE(deepseek_v3->Forward(deepseek_v3_weight, forward_reqs, hidden_unit, RunMode::kNextN).OK());
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());
    std::cout << fmt::format("NextN forward: generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0],
                             generated_tokens1[0])
              << std::endl;
    // Check if generated tokens match any group of expected tokens
    match0 = false;
    match1 = false;
    for (size_t i = 0; i < expected_tokens.size(); ++i) {
      if (generated_tokens0[0] == expected_tokens[i][2]) match0 = true;
      if (generated_tokens1[0] == expected_tokens[i][2]) match1 = true;
    }
    EXPECT_TRUE(match0 || expected_tokens.empty());
    EXPECT_TRUE(match1 || expected_tokens.empty());

    generated_tokens0.clear();
    generated_tokens1.clear();

    EventRecord(start, context->device->GetPrimaryComputeStream());
    for (int i = 0; i < rounds; ++i) {
      deepseek_v3->Forward(deepseek_v3_weight, forward_reqs, hidden_unit, RunMode::kNextN);
      StreamSynchronize(context->device->GetPrimaryComputeStream());
    }
    EventRecord(stop, context->device->GetPrimaryComputeStream());
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "prefill mtp milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;
    EXPECT_TRUE((milliseconds / rounds) < 11);

    deepseek_v3.reset();
    deepseek_v3_weight.reset();

    StreamSynchronize(context->device->GetMemoryManageStream());
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
  }
};

}  // namespace ksana_llm
