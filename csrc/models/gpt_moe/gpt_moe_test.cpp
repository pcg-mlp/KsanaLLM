/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <filesystem>

#include "configure/schedule_config_parser.h"
#include "models/gpt_moe/gpt_moe_model.h"
#include "runtime/hidden_unit_buffer.h"
#include "runtime/weight_instance.h"
#include "samplers/sampler.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/memory_allocator.h"
#include "utils/singleton.h"

#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"
#include "test.h"

using namespace ksana_llm;

class GptMoeTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    std::string model_path = "/model/gpt_moe_256";
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

    block_manager_config.device_allocator_config.blocks_num = 10;  // This test just need a few blocks;
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

    auto memory_allocator_ = std::make_shared<MemoryAllocator>();
    engine_block_manager =
        std::make_shared<BlockAllocatorManager>(true, block_allocator_manager_config, memory_allocator_, context);
    engine_block_group = engine_block_manager->GetBlockAllocatorGroup(0);
    executor_block_manager =
        std::make_shared<BlockAllocatorManager>(false, block_allocator_manager_config, memory_allocator_, context);
    executor_block_group = executor_block_manager->GetBlockAllocatorGroup(0);

    hidden_unit_buffer_helper = std::make_shared<HiddenUnitBufferHelper>(env);
    hidden_unit = hidden_unit_buffer_helper->CreateDeviceBuffer();
  }

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  PipelineConfig pipeline_config;
  std::shared_ptr<BlockAllocatorManager> engine_block_manager;
  std::shared_ptr<BlockAllocatorManager> executor_block_manager;
  std::shared_ptr<BlockAllocatorGroupInterface> engine_block_group;
  std::shared_ptr<BlockAllocatorGroupInterface> executor_block_group;
  std::shared_ptr<Context> context;
  std::shared_ptr<HiddenUnitBufferHelper> hidden_unit_buffer_helper;
  HiddenUnitDeviceBuffer *hidden_unit = nullptr;

  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_;

  void TestGptMoeForward() {
    int device_id = 0;
    SetDevice(device_id);
    std::filesystem::path model_path(model_config.path);
    if (!std::filesystem::exists(model_path)) {
      KLLM_LOG_ERROR << fmt::format("The given model path {} does not exist.", model_config.path);
      EXPECT_TRUE(std::filesystem::exists(model_path));
    }

    auto weight_instance = std::make_unique<WeightInstance>(model_config, runtime_config, context);
    weight_instance->Load();
    auto gpt_moe_weight = weight_instance->GetWeight();

    auto gpt_moe = std::make_shared<GptMoeModel>(model_config, runtime_config, device_id, context, gpt_moe_weight,
                                                 block_cache_layout_);

    // Context forward
    auto forward = std::make_unique<ForwardRequest>();
    std::vector<int> input_ids = {456, 1687};
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
    std::unordered_map<std::string, TargetDescribe> request_target;
    TargetDescribe target_describe;
    target_describe.slice_pos.push_back({0, static_cast<int>(input_ids.size()) - 1});
    target_describe.token_reduce_mode = TokenReduceMode::GATHER_TOKEN_ID;
    request_target["logits"] = target_describe;
    forward->request_target = std::make_shared<const decltype(request_target)>(request_target);
    forward->logits_custom_length = input_ids.size();
    forward->sampling_token_num = forward->logits_custom_length;
    std::unordered_map<std::string, PythonTensor> response;
    forward->response = &response;

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
    }

    std::vector<ForwardRequest> forward_reqs = {*forward};
    EXPECT_TRUE(gpt_moe->Forward(gpt_moe_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    // Sampling with GATHER_TOKEN_ID populates response["logits"] via Sampler::CopyProbsOutputToRequests.
    SamplingRequest sample_req;
    auto logprobs = std::make_shared<std::vector<std::vector<std::pair<int, float>>>>();
    std::vector<int> generated_tokens;
    sample_req.input_tokens = std::make_shared<std::vector<int>>(input_ids);
    sample_req.sampling_token_num = forward_reqs[0].sampling_token_num;
    sample_req.logits_custom_length = forward_reqs[0].logits_custom_length;
    sample_req.logits_offset = forward_reqs[0].logits_offset;
    sample_req.sampling_result_tokens = &generated_tokens;
    sample_req.logprobs = logprobs;
    sample_req.logits_buf = gpt_moe->GetLogitsPtr();
    sample_req.request_target = forward_reqs[0].request_target;
    sample_req.response = &response;
    SamplingConfig sample_config;
    sample_config.topp = 0;
    sample_config.temperature = 0;
    sample_req.sampling_config = &sample_config;

    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);

    std::vector<SamplingRequest> sample_reqs = {sample_req};
    auto sampler = std::make_shared<Sampler>(batch_scheduler_config, device_id, context);
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    // Verify response
    EXPECT_TRUE(response.size() == 1);
    EXPECT_TRUE(response.find("logits") != response.end());
    const PythonTensor &ret_tensor = response.at("logits");
    EXPECT_TRUE(ret_tensor.shape.size() == 1);
    EXPECT_TRUE(ret_tensor.shape[0] == input_ids.size());
    EXPECT_TRUE(ret_tensor.dtype == GetTypeString(TYPE_FP32));
    EXPECT_TRUE(ret_tensor.data.size() == input_ids.size() * sizeof(float));

    const float *probs = reinterpret_cast<const float *>(ret_tensor.data.data());
    EXPECT_NEAR(probs[0], 5.65879e-06f, 1e-7);
    EXPECT_NEAR(probs[1], 3.125e-05f, 1e-6);

    gpt_moe.reset();
    gpt_moe_weight.reset();
  }
};

TEST_F(GptMoeTest, FP16ForwardTest) {
#ifdef ENABLE_CUDA
  runtime_config.inter_data_type = TYPE_FP16;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_FP16 weight_data_type forward." << std::endl;
  TestGptMoeForward();
#endif
}
