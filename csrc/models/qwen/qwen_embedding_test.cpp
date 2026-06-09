/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <stdlib.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <vector>

#include "cache_manager/prefix_cache_manager.h"
#include "configure/json_config_utils.h"
#include "configure/model_config_parser.h"
#include "configure/schedule_config_parser.h"
#include "models/qwen/qwen_embedding_model.h"
#include "runtime/hidden_unit_buffer.h"
#include "runtime/weight_instance.h"
#include "test.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/memory_allocator.h"
#include "utils/request.h"
#include "utils/singleton.h"

#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"

using namespace ksana_llm;

// ============================================================================
// Qwen3 Embedding Model Test Fixture
// ============================================================================

class QwenEmbeddingTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    std::string model_path = "/model/Qwen3-Embedding-8B";
    std::string yaml_path = "../../../examples/llama7b/ksana_llm.yaml";

    const char* skip_env = std::getenv("SKIP_TEST_IF_MODEL_NOT_FOUND");
    if (skip_env != nullptr && std::string(skip_env) == "1" && !std::filesystem::exists(model_path)) {
      GTEST_SKIP() << "Model path " << model_path << " not found, skipping test.";
    }

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / yaml_path;
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    AttentionBackendManager::GetInstance()->Initialize();
    env_ = Singleton<Environment>::GetInstance();
    context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(1, 1, 1, env_));
    context->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context->device = std::make_shared<DeviceContext>(1);
    context->device->ext->InitNcclParam(nccl_uid);

    env_->ParseConfig(config_path, model_path);
    env_->UpdateModelConfig();
    env_->GetModelConfig(model_config);
    InitPipelineConfigForTest(env_, model_config);

    // Embedding models must have these flags set.
    // In production, model_config_parser.cpp detects 1_Pooling/config.json automatically.
    // For the test, we verify they were correctly set.
    EXPECT_TRUE(model_config.is_embedding_model);
    EXPECT_EQ(model_config.pooling_type, PoolingType::LAST);
    EXPECT_TRUE(model_config.normalize_embedding);

    BatchSchedulerConfig batch_scheduler_config;
    env_->GetBatchSchedulerConfig(batch_scheduler_config);
    batch_scheduler_config.max_batch_size = 4;
    batch_scheduler_config.max_step_token_num = 256;
    batch_scheduler_config.max_token_len = 256;
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

    hidden_unit_buffer_helper = std::make_shared<HiddenUnitBufferHelper>(env_);
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
  HiddenUnitDeviceBuffer* hidden_unit = nullptr;

  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;

  CacheManagerConfig cache_manager_config;
  std::shared_ptr<CacheManagerInterface> cache_manager = nullptr;
  std::shared_ptr<Environment> env_;

  // Helper: create a ForwardRequest with embedding-specific fields.
  // request_target: shared_ptr (ForwardRequest takes shared_ptr<const ...>)
  // response: raw pointer (ForwardRequest takes raw pointer)
  // Caller must ensure response outlives the ForwardRequest.
  std::unique_ptr<ForwardRequest> MakeEmbeddingForwardRequest(
      const std::vector<int>& input_ids,
      std::shared_ptr<std::unordered_map<std::string, TargetDescribe>>& request_target,
      std::unordered_map<std::string, PythonTensor>& response,
      std::vector<FlexibleCachedCopyTask>& flexible_cached_copy_tasks, EmbeddingSlice& embedding_slice) {
    auto forward = std::make_unique<ForwardRequest>();
    forward->block_allocator_manager = executor_block_manager;
    forward->forwarding_tokens = std::make_shared<std::vector<int>>(input_ids);
    forward->flexible_cached_copy_tasks = &flexible_cached_copy_tasks;
    forward->input_refit_embedding = &embedding_slice;

    // Embedding models need request_target with "layernorm" key to trigger output.
    request_target = std::make_shared<std::unordered_map<std::string, TargetDescribe>>();
    (*request_target)["layernorm"] = TargetDescribe{};
    forward->request_target = request_target;

    // Response map where the embedding will be written (raw pointer).
    forward->response = &response;

    // Allocate KV cache blocks for this request.
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

    return forward;
  }

  // =========================================================================
  // Forward test: load real weights, run embedding forward, verify output.
  // =========================================================================
  void TestQwenEmbeddingForward() {
    int device_id = 0;
    SetDevice(device_id);

    const std::string real_model_path = "/model/Qwen3-Embedding-8B";
    if (!std::filesystem::exists(real_model_path)) {
      GTEST_SKIP() << "Model path does not exist: " << real_model_path;
    }

    // Load real weights.
    std::unique_ptr<WeightInstance> weight_instance =
        std::make_unique<WeightInstance>(model_config, runtime_config, context);
    weight_instance->Load();
    std::shared_ptr<BaseWeight> model_weight = weight_instance->GetWeight();

    // Create QwenEmbeddingModel.
    std::shared_ptr<QwenEmbeddingModel> model = std::make_shared<QwenEmbeddingModel>(
        model_config, runtime_config, device_id, context, model_weight, block_cache_layout_);

    // Prepare a single embedding request (prefill only, no decode).
    std::vector<int> input_ids = {151643, 8506, 553, 220, 16};  // "Hello world" tokenized roughly
    std::shared_ptr<std::unordered_map<std::string, TargetDescribe>> request_target;
    std::unordered_map<std::string, PythonTensor> response;
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
    EmbeddingSlice embedding_slice;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;

    auto forward =
        MakeEmbeddingForwardRequest(input_ids, request_target, response, flexible_cached_copy_tasks, embedding_slice);

    // Embedding models only run prefill (ContextDecode), no decode step.
    std::vector<ForwardRequest> forward_reqs = {*forward};
    Status status = model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain);
    EXPECT_TRUE(status.OK()) << "Forward failed: " << status.ToString();
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    // Verify: response should contain "layernorm" entry with embedding data.
    auto it = response.find("layernorm");
    ASSERT_NE(it, response.end()) << "Response should contain 'layernorm' embedding";

    const PythonTensor& embedding_tensor = it->second;
    const size_t expected_hidden_size = model_config.hidden_units;  // 4096 for Qwen3-Embedding-8B

    // Verify shape: [1, hidden_size]
    ASSERT_EQ(embedding_tensor.shape.size(), 2u);
    EXPECT_EQ(embedding_tensor.shape[0], 1u);
    EXPECT_EQ(embedding_tensor.shape[1], expected_hidden_size);

    // Verify data is not empty and has correct byte size.
    const size_t expected_bytes = expected_hidden_size * 2;  // bf16 = 2 bytes per element
    EXPECT_EQ(embedding_tensor.data.size(), expected_bytes);

    // Verify the embedding is not all zeros (sanity check).
    bool has_nonzero = false;
    for (size_t i = 0; i < embedding_tensor.data.size(); ++i) {
      if (embedding_tensor.data[i] != 0) {
        has_nonzero = true;
        break;
      }
    }
    EXPECT_TRUE(has_nonzero) << "Embedding output should not be all zeros";

    std::cout << "Embedding output shape: [" << embedding_tensor.shape[0] << ", " << embedding_tensor.shape[1]
              << "], data bytes: " << embedding_tensor.data.size() << ", dtype: " << embedding_tensor.dtype
              << std::endl;

    model.reset();
    model_weight.reset();
    StreamSynchronize(context->device->GetMemoryManageStream());
    DeviceSynchronize();
  }

  // =========================================================================
  // Batch forward test: multiple sequences in one batch.
  // =========================================================================
  void TestQwenEmbeddingBatchForward() {
    int device_id = 0;
    SetDevice(device_id);

    const std::string real_model_path = "/model/Qwen3-Embedding-8B";
    if (!std::filesystem::exists(real_model_path)) {
      GTEST_SKIP() << "Model path does not exist: " << real_model_path;
    }

    std::unique_ptr<WeightInstance> weight_instance =
        std::make_unique<WeightInstance>(model_config, runtime_config, context);
    weight_instance->Load();
    std::shared_ptr<BaseWeight> model_weight = weight_instance->GetWeight();

    std::shared_ptr<QwenEmbeddingModel> model = std::make_shared<QwenEmbeddingModel>(
        model_config, runtime_config, device_id, context, model_weight, block_cache_layout_);

    // Two different sequences.
    std::vector<int> input_ids_0 = {151643, 8506, 553, 220, 16};
    std::vector<int> input_ids_1 = {151643, 25, 1000, 3456};

    std::shared_ptr<std::unordered_map<std::string, TargetDescribe>> target_0, target_1;
    std::unordered_map<std::string, PythonTensor> response_0, response_1;
    std::vector<FlexibleCachedCopyTask> tasks_0, tasks_1;
    EmbeddingSlice slice_0, slice_1;
    std::vector<int> pos_0, pos_1;
    std::vector<std::vector<float>> emb_0, emb_1;
    slice_0.pos = pos_0;
    slice_0.embeddings = emb_0;
    slice_1.pos = pos_1;
    slice_1.embeddings = emb_1;

    auto forward_0 = MakeEmbeddingForwardRequest(input_ids_0, target_0, response_0, tasks_0, slice_0);
    auto forward_1 = MakeEmbeddingForwardRequest(input_ids_1, target_1, response_1, tasks_1, slice_1);

    std::vector<ForwardRequest> forward_reqs = {*forward_0, *forward_1};
    Status status = model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain);
    EXPECT_TRUE(status.OK()) << "Batch forward failed: " << status.ToString();
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    // Both responses should have embedding output.
    for (size_t i = 0; i < 2; ++i) {
      auto& resp = (i == 0) ? response_0 : response_1;
      auto it = resp.find("layernorm");
      ASSERT_NE(it, resp.end()) << "Response " << i << " should contain 'layernorm' embedding";

      const PythonTensor& emb = it->second;
      EXPECT_EQ(emb.shape.size(), 2u);
      EXPECT_EQ(emb.shape[0], 1u);
      EXPECT_EQ(emb.shape[1], model_config.hidden_units);
      EXPECT_GT(emb.data.size(), 0u);
    }

    // Different inputs should produce different embeddings.
    const auto& data_0 = response_0.at("layernorm").data;
    const auto& data_1 = response_1.at("layernorm").data;
    EXPECT_NE(data_0, data_1) << "Different inputs should produce different embeddings";

    std::cout << "Batch forward: 2 sequences processed successfully" << std::endl;

    model.reset();
    model_weight.reset();
    StreamSynchronize(context->device->GetMemoryManageStream());
    DeviceSynchronize();
  }

  // =========================================================================
  // Performance test: measure embedding forward latency.
  // =========================================================================
  void TestQwenEmbeddingPerformance() {
    int device_id = 0;
    SetDevice(device_id);

    const std::string real_model_path = "/model/Qwen3-Embedding-8B";
    if (!std::filesystem::exists(real_model_path)) {
      GTEST_SKIP() << "Model path does not exist: " << real_model_path;
    }

    std::unique_ptr<WeightInstance> weight_instance =
        std::make_unique<WeightInstance>(model_config, runtime_config, context);
    weight_instance->Load();
    std::shared_ptr<BaseWeight> model_weight = weight_instance->GetWeight();

    std::shared_ptr<QwenEmbeddingModel> model = std::make_shared<QwenEmbeddingModel>(
        model_config, runtime_config, device_id, context, model_weight, block_cache_layout_);

    std::vector<int> input_ids = {151643, 8506, 553, 220, 16};
    std::shared_ptr<std::unordered_map<std::string, TargetDescribe>> request_target;
    std::unordered_map<std::string, PythonTensor> response;
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
    EmbeddingSlice embedding_slice;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;

    auto forward =
        MakeEmbeddingForwardRequest(input_ids, request_target, response, flexible_cached_copy_tasks, embedding_slice);

    std::vector<ForwardRequest> forward_reqs = {*forward};

    Event start, stop;
    float milliseconds = 0;
    int rounds = 10;
    EventCreate(&start);
    EventCreate(&stop);

    // Warmup.
    for (int i = 0; i < 3; ++i) {
      model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain);
      StreamSynchronize(context->device->GetPrimaryComputeStream());
    }

    // Measure.
    EventRecord(start, context->device->GetPrimaryComputeStream());
    for (int i = 0; i < rounds; ++i) {
      model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain);
      StreamSynchronize(context->device->GetPrimaryComputeStream());
    }
    EventRecord(stop, context->device->GetPrimaryComputeStream());
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "Qwen3-Embedding-8B prefill avg latency (" << rounds << " rounds): " << milliseconds / rounds << " ms"
              << std::endl;

    model.reset();
    model_weight.reset();
    StreamSynchronize(context->device->GetMemoryManageStream());
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
  }

  // =========================================================================
  // Helper: run forward with a specific pooling_type and normalize_embedding.
  // Reuses the same real weights but overrides model config for the LmHead path.
  // =========================================================================
  void TestQwenEmbeddingForwardWithConfig(PoolingType pooling_type, bool normalize, const std::string& label) {
    int device_id = 0;
    SetDevice(device_id);

    const std::string real_model_path = "/model/Qwen3-Embedding-8B";
    if (!std::filesystem::exists(real_model_path)) {
      GTEST_SKIP() << "Model path does not exist: " << real_model_path;
    }

    // Override pooling_type and normalize_embedding in model_config.
    ModelConfig test_config = model_config;
    test_config.pooling_type = pooling_type;
    test_config.normalize_embedding = normalize;

    std::unique_ptr<WeightInstance> weight_instance =
        std::make_unique<WeightInstance>(test_config, runtime_config, context);
    weight_instance->Load();
    std::shared_ptr<BaseWeight> model_weight = weight_instance->GetWeight();

    std::shared_ptr<QwenEmbeddingModel> model = std::make_shared<QwenEmbeddingModel>(
        test_config, runtime_config, device_id, context, model_weight, block_cache_layout_);

    std::vector<int> input_ids = {151643, 8506, 553, 220, 16};
    std::shared_ptr<std::unordered_map<std::string, TargetDescribe>> request_target;
    std::unordered_map<std::string, PythonTensor> response;
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
    EmbeddingSlice embedding_slice;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;

    auto forward =
        MakeEmbeddingForwardRequest(input_ids, request_target, response, flexible_cached_copy_tasks, embedding_slice);

    std::vector<ForwardRequest> forward_reqs = {*forward};
    Status status = model->Forward(model_weight, forward_reqs, hidden_unit, RunMode::kMain);
    EXPECT_TRUE(status.OK()) << label << " forward failed: " << status.ToString();
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    auto it = response.find("layernorm");
    ASSERT_NE(it, response.end()) << label << " response missing 'layernorm'";

    const PythonTensor& emb = it->second;
    ASSERT_EQ(emb.shape.size(), 2u);
    EXPECT_EQ(emb.shape[0], 1u);
    EXPECT_EQ(emb.shape[1], model_config.hidden_units);
    EXPECT_GT(emb.data.size(), 0u);

    // Verify not all zeros.
    bool has_nonzero = false;
    for (size_t i = 0; i < emb.data.size(); ++i) {
      if (emb.data[i] != 0) {
        has_nonzero = true;
        break;
      }
    }
    EXPECT_TRUE(has_nonzero) << label << " embedding output is all zeros";

    std::cout << label << " output shape: [" << emb.shape[0] << ", " << emb.shape[1]
              << "], data bytes: " << emb.data.size() << ", dtype: " << emb.dtype << std::endl;

    model.reset();
    model_weight.reset();
    StreamSynchronize(context->device->GetMemoryManageStream());
    DeviceSynchronize();
  }
};

// ============================================================================
// 1. Forward Test — single sequence embedding with real weights
// ============================================================================

TEST_F(QwenEmbeddingTest, ForwardTest) {
#ifdef ENABLE_CUDA
  TestQwenEmbeddingForward();
#endif
}

// ============================================================================
// 2. Batch Forward Test — multiple sequences in one batch
// ============================================================================

TEST_F(QwenEmbeddingTest, BatchForwardTest) {
#ifdef ENABLE_CUDA
  TestQwenEmbeddingBatchForward();
#endif
}

// ============================================================================
// 3. Performance Test — measure prefill latency
// ============================================================================

TEST_F(QwenEmbeddingTest, PerformanceTest) {
#ifdef ENABLE_CUDA
  TestQwenEmbeddingPerformance();
#endif
}

// ============================================================================
// 3a. MEAN Pooling Forward Test — covers MeanPooling kernel path (BF16)
// ============================================================================

TEST_F(QwenEmbeddingTest, MeanPoolingForwardTest) {
#ifdef ENABLE_CUDA
  TestQwenEmbeddingForwardWithConfig(PoolingType::MEAN, true, "MEAN+normalize");
#endif
}

// ============================================================================
// 3b. CLS Pooling Forward Test — covers CLS token extraction path
// ============================================================================

TEST_F(QwenEmbeddingTest, CLSPoolingForwardTest) {
#ifdef ENABLE_CUDA
  TestQwenEmbeddingForwardWithConfig(PoolingType::CLS, true, "CLS+normalize");
#endif
}

// ============================================================================
// 3c. LAST Pooling without normalize — covers L2Normalize skip path
// ============================================================================

TEST_F(QwenEmbeddingTest, LastPoolingNoNormalizeTest) {
#ifdef ENABLE_CUDA
  TestQwenEmbeddingForwardWithConfig(PoolingType::LAST, false, "LAST+no_normalize");
#endif
}

// ============================================================================
// 3d. MEAN Pooling without normalize — covers MEAN + skip normalize
// ============================================================================

TEST_F(QwenEmbeddingTest, MeanPoolingNoNormalizeTest) {
#ifdef ENABLE_CUDA
  TestQwenEmbeddingForwardWithConfig(PoolingType::MEAN, false, "MEAN+no_normalize");
#endif
}

// ============================================================================
// 4. Config Validation Test — verify is_embedding_model detected correctly
// ============================================================================

TEST_F(QwenEmbeddingTest, ConfigDetectionTest) {
#ifdef ENABLE_CUDA
  // Verify that the environment correctly detected this as an embedding model.
  ModelConfig config;
  env_->GetModelConfig(config);

  EXPECT_TRUE(config.is_embedding_model)
      << "Qwen3-Embedding-8B should be detected as an embedding model via 1_Pooling/config.json";
  EXPECT_EQ(config.pooling_type, PoolingType::LAST) << "Qwen3-Embedding-8B uses pooling_mode_lasttoken=true";
  EXPECT_TRUE(config.normalize_embedding)
      << "modules.json contains Normalize module, so normalize_embedding should be true";
#endif
}

// ============================================================================
// 5. PoolingType enum and utility function tests
// ============================================================================

TEST(EmbeddingConfigTest, PoolingTypeEnumValues) {
  EXPECT_EQ(static_cast<int>(PoolingType::LAST), 0);
  EXPECT_EQ(static_cast<int>(PoolingType::CLS), 1);
  EXPECT_EQ(static_cast<int>(PoolingType::MEAN), 2);
}

TEST(EmbeddingConfigTest, GetPoolingTypeString) {
  EXPECT_EQ(GetPoolingTypeString(PoolingType::LAST), "last");
  EXPECT_EQ(GetPoolingTypeString(PoolingType::CLS), "cls");
  EXPECT_EQ(GetPoolingTypeString(PoolingType::MEAN), "mean");
  EXPECT_EQ(GetPoolingTypeString(static_cast<PoolingType>(99)), "Unknown");
}

TEST(EmbeddingConfigTest, ModelConfigEmbeddingDefaults) {
  ModelConfig config;
  EXPECT_FALSE(config.is_embedding_model);
  EXPECT_EQ(config.pooling_type, PoolingType::LAST);
  EXPECT_TRUE(config.normalize_embedding);
}

// ============================================================================
// 6. Verify Qwen3-Embedding-8B model directory structure
// ============================================================================

TEST(EmbeddingConfigTest, Qwen3EmbeddingModelStructure) {
  const std::string model_path = "/model/Qwen3-Embedding-8B";
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "Model not available: " << model_path;
  }

  EXPECT_TRUE(std::filesystem::exists(model_path + "/1_Pooling/config.json"));
  EXPECT_TRUE(std::filesystem::exists(model_path + "/modules.json"));
  EXPECT_TRUE(std::filesystem::exists(model_path + "/config.json"));

  // Verify pooling config: last-token pooling.
  nlohmann::json pooling_json;
  {
    std::ifstream f(model_path + "/1_Pooling/config.json");
    f >> pooling_json;
  }
  EXPECT_TRUE(pooling_json.value("pooling_mode_lasttoken", false));
  EXPECT_FALSE(pooling_json.value("pooling_mode_cls_token", false));
  EXPECT_FALSE(pooling_json.value("pooling_mode_mean_tokens", false));

  // Verify modules.json contains Normalize.
  nlohmann::json modules_json;
  {
    std::ifstream f(model_path + "/modules.json");
    f >> modules_json;
  }
  bool found_normalize = false;
  for (const auto& module : modules_json) {
    if (module.value("type", "") == "sentence_transformers.models.Normalize") {
      found_normalize = true;
      break;
    }
  }
  EXPECT_TRUE(found_normalize);
}

// ============================================================================
// 7. Pooling mode detection via EnvModelConfigParser (CLS / MEAN / normalize)
// ============================================================================

// Helper: create a temp model dir with config.json + optional 1_Pooling/ + optional modules.json.
static nlohmann::json MakeMinimalQwen3Config() {
  return {{"architectures", {"Qwen3ForCausalLM"}},
          {"model_type", "qwen3"},
          {"hidden_size", 4096},
          {"num_attention_heads", 32},
          {"num_hidden_layers", 36},
          {"num_key_value_heads", 8},
          {"intermediate_size", 12288},
          {"vocab_size", 151665},
          {"head_dim", 128},
          {"rope_theta", 1000000.0},
          {"rms_norm_eps", 1e-6},
          {"max_position_embeddings", 40960},
          {"torch_dtype", "bfloat16"},
          {"bos_token_id", 151643},
          {"eos_token_id", 151645}};
}

static void WriteJson(const std::string& path, const nlohmann::json& j) {
  std::ofstream f(path);
  f << j.dump(2);
}

TEST(EmbeddingConfigTest, NonEmbeddingModelNotDetected) {
  std::string tmp_dir = "/tmp/test_non_embedding_" + std::to_string(getpid());
  std::filesystem::create_directories(tmp_dir);
  WriteJson(tmp_dir + "/config.json", MakeMinimalQwen3Config());

  EnvModelConfigParser parser("auto", "");
  ModelConfig config{};
  config.path = tmp_dir;
  EXPECT_TRUE(parser.ParseModelConfig(tmp_dir, tmp_dir, "config.json", config).OK());
  EXPECT_FALSE(config.is_embedding_model);

  std::filesystem::remove_all(tmp_dir);
}

TEST(EmbeddingConfigTest, DetectCLSPoolingMode) {
  std::string tmp_dir = "/tmp/test_cls_pooling_" + std::to_string(getpid());
  std::filesystem::create_directories(tmp_dir + "/1_Pooling");
  WriteJson(tmp_dir + "/config.json", MakeMinimalQwen3Config());
  WriteJson(tmp_dir + "/1_Pooling/config.json", {{"pooling_mode_cls_token", true}, {"pooling_mode_lasttoken", false}});

  EnvModelConfigParser parser("auto", "");
  ModelConfig config{};
  config.path = tmp_dir;
  EXPECT_TRUE(parser.ParseModelConfig(tmp_dir, tmp_dir, "config.json", config).OK());
  EXPECT_TRUE(config.is_embedding_model);
  EXPECT_EQ(config.pooling_type, PoolingType::CLS);

  std::filesystem::remove_all(tmp_dir);
}

TEST(EmbeddingConfigTest, DetectMEANPoolingMode) {
  std::string tmp_dir = "/tmp/test_mean_pooling_" + std::to_string(getpid());
  std::filesystem::create_directories(tmp_dir + "/1_Pooling");
  WriteJson(tmp_dir + "/config.json", MakeMinimalQwen3Config());
  WriteJson(tmp_dir + "/1_Pooling/config.json",
            {{"pooling_mode_mean_tokens", true}, {"pooling_mode_lasttoken", false}});

  EnvModelConfigParser parser("auto", "");
  ModelConfig config{};
  config.path = tmp_dir;
  EXPECT_TRUE(parser.ParseModelConfig(tmp_dir, tmp_dir, "config.json", config).OK());
  EXPECT_TRUE(config.is_embedding_model);
  EXPECT_EQ(config.pooling_type, PoolingType::MEAN);

  std::filesystem::remove_all(tmp_dir);
}

TEST(EmbeddingConfigTest, DetectNormalizeFromModulesJson) {
  std::string tmp_dir = "/tmp/test_normalize_" + std::to_string(getpid());
  std::filesystem::create_directories(tmp_dir + "/1_Pooling");
  WriteJson(tmp_dir + "/config.json", MakeMinimalQwen3Config());
  WriteJson(tmp_dir + "/1_Pooling/config.json", {{"pooling_mode_lasttoken", true}});

  // With Normalize module → true
  WriteJson(tmp_dir + "/modules.json",
            nlohmann::json::array({{{"idx", 0}, {"type", "sentence_transformers.models.Transformer"}},
                                   {{"idx", 1}, {"type", "sentence_transformers.models.Normalize"}}}));
  {
    EnvModelConfigParser parser("auto", "");
    ModelConfig config{};
    config.path = tmp_dir;
    parser.ParseModelConfig(tmp_dir, tmp_dir, "config.json", config);
    EXPECT_TRUE(config.normalize_embedding);
  }

  // Without Normalize module → false
  WriteJson(tmp_dir + "/modules.json",
            nlohmann::json::array({{{"idx", 0}, {"type", "sentence_transformers.models.Transformer"}}}));
  {
    EnvModelConfigParser parser("auto", "");
    ModelConfig config{};
    config.path = tmp_dir;
    parser.ParseModelConfig(tmp_dir, tmp_dir, "config.json", config);
    EXPECT_FALSE(config.normalize_embedding);
  }

  // No modules.json → default true
  std::filesystem::remove(tmp_dir + "/modules.json");
  {
    EnvModelConfigParser parser("auto", "");
    ModelConfig config{};
    config.path = tmp_dir;
    parser.ParseModelConfig(tmp_dir, tmp_dir, "config.json", config);
    EXPECT_TRUE(config.normalize_embedding);
  }

  std::filesystem::remove_all(tmp_dir);
}

// ============================================================================
// 8. ModelSampler — embedding model skips sampling
// ============================================================================

#include "executor/model_sampler.h"
#include "runtime/model_output.h"
#include "runtime/sampling_output.h"

// Test that ModelSampler::Sampling() takes the embedding early-return path:
//   - Sets sampling_result_tokens = {0} (dummy token, no real decoding)
//   - Builds minimal SamplingRequest with req_id + response pointer
//   - Returns immediately without invoking the Sampler
TEST_F(QwenEmbeddingTest, ModelSamplerEmbeddingPath) {
#ifdef ENABLE_CUDA
  // ModelSampler reads is_embedding_model from Environment at construction.
  // Our fixture already parsed /model/Qwen3-Embedding-8B, so env_ has is_embedding_model=true.
  auto model_sampler = std::make_shared<ModelSampler>(env_, context);

  // Build a fake ScheduleOutput with two ExecutorInferRequests.
  auto schedule_output = std::make_shared<ScheduleOutput>();

  auto req_0 = std::make_shared<ExecutorInferRequest>();
  req_0->req_id = 100;
  req_0->response["layernorm"] = PythonTensor{{0x01, 0x02}, {1, 4096}, "bfloat16"};

  auto req_1 = std::make_shared<ExecutorInferRequest>();
  req_1->req_id = 200;
  req_1->response["layernorm"] = PythonTensor{{0x03, 0x04}, {1, 4096}, "bfloat16"};

  schedule_output->executor_running_reqs = {req_0, req_1};

  // Build NewModelOutput.
  auto model_output = std::make_shared<NewModelOutput>();
  model_output->schedule_output = schedule_output;
  model_output->run_mode = RunMode::kMain;

  // Build SamplingOutput.
  auto sampling_output = std::make_shared<SamplingOutput>();

  // Call Sampling — should take the embedding early-return path.
  Status status = model_sampler->Sampling(model_output, sampling_output);
  EXPECT_TRUE(status.OK());

  // Verify: sampling_result_tokens set to {0} (dummy, no real decoding).
  EXPECT_EQ(req_0->sampling_result_tokens.size(), 1u);
  EXPECT_EQ(req_0->sampling_result_tokens[0], 0);
  EXPECT_EQ(req_1->sampling_result_tokens.size(), 1u);
  EXPECT_EQ(req_1->sampling_result_tokens[0], 0);

  // Verify: sampling_reqs built with correct req_id and response pointer.
  ASSERT_EQ(sampling_output->sampling_reqs.size(), 2u);
  EXPECT_EQ(sampling_output->sampling_reqs[0].req_id, 100);
  EXPECT_EQ(sampling_output->sampling_reqs[0].response, &(req_0->response));
  EXPECT_EQ(sampling_output->sampling_reqs[1].req_id, 200);
  EXPECT_EQ(sampling_output->sampling_reqs[1].response, &(req_1->response));

  // Verify: logprobs allocated (empty, but not null).
  EXPECT_NE(sampling_output->sampling_reqs[0].logprobs, nullptr);
  EXPECT_NE(sampling_output->sampling_reqs[1].logprobs, nullptr);

  // Verify: original response data preserved (not overwritten by sampling).
  EXPECT_EQ(req_0->response.count("layernorm"), 1u);
  EXPECT_EQ(req_1->response.count("layernorm"), 1u);
#endif
}

// Test that the embedding path returns immediately and does NOT call the underlying Sampler.
// We verify this indirectly: schedule_output should be passed through unchanged,
// and response data should not be corrupted.
TEST_F(QwenEmbeddingTest, ModelSamplerEmbeddingPassthrough) {
#ifdef ENABLE_CUDA
  auto model_sampler = std::make_shared<ModelSampler>(env_, context);

  auto schedule_output = std::make_shared<ScheduleOutput>();
  schedule_output->schedule_id = 42;

  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 999;
  // Embedding response with actual data bytes.
  PythonTensor emb_tensor;
  emb_tensor.shape = {1, 4096};
  emb_tensor.dtype = "bfloat16";
  emb_tensor.data.resize(4096 * 2, 0x42);  // 8192 bytes of dummy data
  req->response["layernorm"] = emb_tensor;
  schedule_output->executor_running_reqs = {req};

  auto model_output = std::make_shared<NewModelOutput>();
  model_output->schedule_output = schedule_output;
  model_output->run_mode = RunMode::kMain;

  auto sampling_output = std::make_shared<SamplingOutput>();
  Status status = model_sampler->Sampling(model_output, sampling_output);
  EXPECT_TRUE(status.OK());

  // schedule_id and schedule_output should be passed through.
  EXPECT_EQ(sampling_output->schedule_id, 42u);
  EXPECT_EQ(sampling_output->schedule_output, schedule_output);

  // Response data should be intact (not corrupted by sampling).
  EXPECT_EQ(req->response["layernorm"].data.size(), 4096u * 2);
  EXPECT_EQ(req->response["layernorm"].data[0], 0x42);
  EXPECT_EQ(req->response["layernorm"].shape[1], 4096u);
#endif
}
