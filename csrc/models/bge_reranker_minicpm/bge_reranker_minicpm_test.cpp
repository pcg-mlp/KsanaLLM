/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <filesystem>
#include <fstream>

#include "cache_manager/prefix_cache_manager.h"
#include "configure/json_config_utils.h"
#include "configure/schedule_config_parser.h"
#include "models/bge_reranker_minicpm/bge_reranker_minicpm_model.h"
#include "runtime/hidden_unit_buffer.h"
#include "runtime/weight_instance.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/memory_allocator.h"
#include "utils/singleton.h"

#include "helpers/model_test_helper.h"
#include "test.h"

using namespace ksana_llm;

class BgeRerankerMinicpmTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    std::string model_path = "/model/bge-reranker-v2-minicpm-layerwise/";
    std::string yaml_path = "../../../examples/llama7b/ksana_llm.yaml";
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

    BlockManagerConfig block_manager_config;
    env->InitializeBlockManagerConfig();
    env->GetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);

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

    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = 1;
    cache_manager_config.swap_threadpool_size = 2;
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
  std::shared_ptr<Context> context;
  std::shared_ptr<HiddenUnitBufferHelper> hidden_unit_buffer_helper;
  HiddenUnitDeviceBuffer *hidden_unit = nullptr;

  CacheManagerConfig cache_manager_config;
  std::shared_ptr<CacheManagerInterface> cache_manager;

  template <typename weight_data_type>
  void TestBgeRerankerMinicpmForward() {
    int device_id = 0;
    SetDevice(device_id);
    std::filesystem::path model_path(model_config.path);
    if (!std::filesystem::exists(model_path)) {
      KLLM_LOG_ERROR << fmt::format("The given model path {} does not exist.", model_config.path);
      EXPECT_TRUE(std::filesystem::exists(model_path));
    }

    auto weight_instance = std::make_unique<WeightInstance>(model_config, runtime_config, context);
    weight_instance->Load();
    auto bge_reranker_minicpm_weight = weight_instance->GetWeight();

    auto bge_reranker_minicpm = std::make_shared<BgeRerankerMinicpmModel>(model_config, runtime_config, device_id,
                                                                          context, bge_reranker_minicpm_weight);

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
    request_target.emplace("lm_head",
                           TargetDescribe{.cutoff_layer = {10}, .token_reduce_mode = TokenReduceMode::GATHER_ALL});
    forward->request_target = std::make_shared<const decltype(request_target)>(request_target);
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
    // AppendFlatKVCacheBlkIds(model_config.num_layer, {block_ids}, forward->atb_kv_cache_base_blk_ids, cache_manager);
    for (int block_idx = 0; block_idx < use_block_num; block_idx++) {
      Memset(forward->kv_cache_ptrs[block_idx], 0, runtime_config.attn_backend_config.block_size);
      KLLM_LOG_DEBUG << fmt::format(
          "kv_cache_ptrs {} end {}", forward->kv_cache_ptrs[block_idx],
          forward->kv_cache_ptrs[block_idx] + (runtime_config.attn_backend_config.block_size));
    }

    std::vector<ForwardRequest> forward_reqs = {*forward};
    EXPECT_TRUE(
        bge_reranker_minicpm->Forward(bge_reranker_minicpm_weight, forward_reqs, hidden_unit, RunMode::kMain).OK());

    // Verify response
    EXPECT_TRUE(response.size() == 1);
    EXPECT_TRUE(response.find("lm_head") != response.end());
    const PythonTensor &ret_tensor = response.at("lm_head");
    EXPECT_TRUE(ret_tensor.shape.size() == 2);
    EXPECT_TRUE(ret_tensor.shape[0] == input_ids.size());
    EXPECT_TRUE(ret_tensor.shape[1] == 1);
    EXPECT_TRUE(ret_tensor.dtype == "bfloat16");
    EXPECT_TRUE(ret_tensor.data.size() == 4);
    const weight_data_type *logits = reinterpret_cast<const weight_data_type *>(ret_tensor.data.data());
    if (model_config.quant_config.method == QUANT_FP8_E4M3) {
      EXPECT_NEAR(static_cast<float>(logits[0]), -9.0625, 1e-3);
      EXPECT_NEAR(static_cast<float>(logits[1]), -11.8125, 1e-3);
    } else {
      EXPECT_NEAR(static_cast<float>(logits[0]), -9.4375, 1e-3);
      EXPECT_NEAR(static_cast<float>(logits[1]), -12.125, 1e-3);
    }

    bge_reranker_minicpm.reset();
    bge_reranker_minicpm_weight.reset();
  }
};

TEST_F(BgeRerankerMinicpmTest, BF16ForwardTest) {
#ifdef ENABLE_CUDA
  runtime_config.inter_data_type = TYPE_BF16;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_BF16 weight_data_type forward." << std::endl;
  TestBgeRerankerMinicpmForward<bfloat16>();
#endif
}

TEST_F(BgeRerankerMinicpmTest, FP8ForwardTest) {
#ifdef ENABLE_CUDA
  runtime_config.inter_data_type = TYPE_BF16;
  model_config.quant_config.method = QUANT_FP8_E4M3;
  Singleton<Environment>::GetInstance()->yaml_weight_quant_method_ = "fp8_e4m3";
  std::cout << "Test TYPE_BF16 weight_data_type forward with quant fp8_e4m3." << std::endl;
  TestBgeRerankerMinicpmForward<bfloat16>();
#endif
}
