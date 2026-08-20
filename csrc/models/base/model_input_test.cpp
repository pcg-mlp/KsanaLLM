/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

#include <unistd.h>  // getpid
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#  include "device/nvidia/nccl_utils.h"
#endif
#include "configure/global_context.h"
#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"
#include "models/base/attention_meta/flash_attention_meta_builder.h"
#include "models/base/attention_meta/mla_attention_meta_builder.h"
#include "models/base/attention_meta/paged_attention_meta_builder.h"
#include "models/base/model_input.h"
#include "runtime/speculative/mtp/mtp_gather_index_builder.h"
#include "models/deepseek_v3/attention_meta/deepseek_v3_attention_meta.h"
#include "models/deepseek_v3/attention_meta/deepseek_v3_attention_meta_creator.h"
#include "tests/test.h"
#include "utils/singleton.h"

namespace py = pybind11;

namespace ksana_llm {

class ModelInputTest : public testing::Test {
 protected:
  void SetUp() override {
    int rank = 0;

    // Parse the yaml config file.
    const auto& env = Singleton<Environment>::GetInstance();
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();
    env->ParseConfig(config_path);

    // Initialize the context with global and device contexts
    context_ = std::make_shared<Context>();
    context_->global.reset(new GlobalContext(1, 1, 1, env));
    context_->global->InitDistributedConfig();

#ifdef ENABLE_CUDA
    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context_->device = std::make_shared<DeviceContext>(1);
    context_->device->ext->InitNcclParam(nccl_uid);
#else
    context_->device = std::make_shared<DeviceContext>(1);
#endif

    // Initialize the model config
    ModelConfig model_config;
    env->GetModelConfig(model_config);

    // 修改kv_lora_rank为512
    model_config.mla_config.kv_lora_rank = 512;
    // Force mla
    model_config.use_mla = true;
    env->SetModelConfig(model_config);

    InitPipelineConfigForTest(env, model_config);

    // Initialize the block manager.
    auto block_cache_layout = GetTestModelCacheLayout(env);
    env->InitializeBlockManagerConfig(block_cache_layout);
    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.block_host_memory_factor = 0.0;
    block_manager_config.device_allocator_config.blocks_num = 10;  // This test just need a few blocks;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;
    env->SetBlockManagerConfig(block_manager_config);

    RuntimeConfig runtime_config;
    env->GetRuntimeConfig(runtime_config);
    const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = test_info->name();
    if (test_name.find("PrepareFlashRotaryMlaFlexibleCacheTest") != std::string::npos) {
      runtime_config.attn_backend_config.enable_blocked_multi_token_forwarding_kv = true;
      runtime_config.enable_flexible_caching = true;
    } else if (test_name.find("PrepareInputLengthTest") != std::string::npos) {
      model_config.use_dsa = true;
      runtime_config.attn_backend_config.kv_cache_dtype = TYPE_FP8_DS_MLA;
    }
    env->schedule_config_parser_.runtime_config_ = runtime_config;

    // Initialize the model input object.
    model_input = std::make_unique<ModelInput>(model_config, runtime_config, rank, context_, block_cache_layout);

    // When the model uses MLA / DSA, inject the DeepSeekV3-specific creator so that the
    // corresponding shared attention tensors (tile_scheduler_metadatas, num_splits, etc.)
    // are properly allocated for the tests.
    if (model_config.use_mla || model_config.use_dsa) {
      model_input->ReplaceAttentionMetaCreator(
          std::make_unique<DeepSeekV3AttentionMetaCreator>(model_input->GetCreatorConfig()));
    }

    // Initialize the random seed with 0.
    std::srand(0);
  }

  void TearDown() override {}

  AttentionBuilderConfig MakeAttentionBuilderConfig() {
    return {model_input->model_config_,
            model_input->runtime_config_,
            model_input->enable_blocked_multi_token_forwarding_kv_,
            model_input->use_flashinfer_for_decode_,
            model_input->rank_,
            model_input->layer_num_on_node_,
            model_input->block_size_,
            model_input->kv_cache_types_,
            model_input->kv_cache_layer_offsets_,
            model_input->attn_dp_group_id_,
            model_input->attn_dp_rank_id_,
            model_input->context_,
            model_input->kvcache_offset_event,
            model_input->rotary_embedding_event};
  }

 protected:
  std::shared_ptr<Context> context_;
  std::unique_ptr<ModelInput> model_input;
};

TEST_F(ModelInputTest, PrepareInputRefitTest) {
  std::vector<float*> input_refit_emb_ptr;
  std::vector<int64_t> input_refit_pos_list;

  auto VerifyPrepareInputRefit = [&]() {
    const size_t input_refit_size = input_refit_emb_ptr.size();
    EXPECT_EQ(model_input->cpu_input_refit_tensor.emb_tensors.size(), input_refit_size);
    EXPECT_EQ(model_input->cpu_input_refit_tensor.pos_tensor.shape.size(), 1);
    EXPECT_EQ(model_input->cpu_input_refit_tensor.pos_tensor.shape[0], input_refit_size);
    int64_t* cpu_input_refit_pos =
        reinterpret_cast<int64_t*>(model_input->cpu_input_refit_tensor.pos_tensor.GetPtr<void>());
    for (size_t i = 0; i < input_refit_size; i++) {
      EXPECT_EQ(model_input->cpu_input_refit_tensor.emb_tensors[i].GetPtr<float>(), input_refit_emb_ptr[i]);
      EXPECT_EQ(cpu_input_refit_pos[i], input_refit_pos_list[i]);
    }
  };

  // Test for each selected batch size.
  for (const int batch_size : {1, 3, 4}) {
    input_refit_emb_ptr.clear();
    input_refit_pos_list.clear();

    std::vector<ForwardRequest> forward_reqs;

    // Reserve memory to avoid memory address being moved.
    std::vector<std::vector<int>> output_tokens;
    std::vector<EmbeddingSlice> embedding_slices;
    // Store torch tensors to keep them alive during the test
    std::vector<torch::Tensor> torch_tensors;
    forward_reqs.reserve(batch_size);
    output_tokens.reserve(batch_size);
    embedding_slices.reserve(batch_size);

    size_t pos_offset = 0;

    // Construct input refit embedding tensors (new format only).
    for (int i = 0; i < batch_size; i++) {
      ForwardRequest forward_req;
      const size_t output_tokens_size = std::rand() % 4096 + 10;
      output_tokens.emplace_back(output_tokens_size);
      forward_req.forwarding_tokens = std::make_shared<std::vector<int>>(output_tokens.back());
      EmbeddingSlice embedding_slice;
      const int input_refit_size = std::rand() % 3 + 1;
      for (int j = 0; j < input_refit_size; j++) {
        const size_t embedding_size = std::rand() % output_tokens_size + 1;
        const size_t embedding_start_pos = std::rand() % embedding_size;
        embedding_slice.pos.push_back(embedding_start_pos);
        // Create embedding tensor using libtorch
        torch::Tensor embedding_tensor = torch::randn(static_cast<int64_t>(embedding_size), torch::kFloat32);
        torch_tensors.push_back(embedding_tensor);  // Keep tensor alive
        input_refit_emb_ptr.emplace_back(reinterpret_cast<float*>(embedding_tensor.data_ptr()));
        input_refit_pos_list.emplace_back(pos_offset + embedding_start_pos);
        // Convert torch::Tensor to ksana_llm::Tensor
        Tensor ksana_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {embedding_size},
                            embedding_tensor.data_ptr(), "embedding_tensor");
        embedding_slice.embedding_tensors.push_back(ksana_tensor);
      }
      embedding_slices.push_back(std::move(embedding_slice));
      forward_req.input_refit_embedding = &embedding_slices.back();
      forward_reqs.push_back(std::move(forward_req));
      pos_offset += output_tokens_size;
    }

    // Parse and load the input refit embeddings.
    model_input->PrepareInputRefit(forward_reqs);

    // Check the result of PrepareInputRefit.
    VerifyPrepareInputRefit();

    // Construct bad input.
    forward_reqs[0].input_refit_embedding->embedding_tensors.clear();
    EXPECT_THROW(
        try { model_input->PrepareInputRefit(forward_reqs); } catch (const std::runtime_error& e) {
          EXPECT_NE(
              strstr(e.what(), "`input_refit_pos.size()` should be equal to `input_refit_embedding_tensors.size()`."),
              nullptr);
          throw;
        },
        std::runtime_error);
  }
}

TEST_F(ModelInputTest, PrepareCutoffLayerTest) {
  model_input->model_config_.type = "minicpm";

  // Test case 1: When all request_target are nullptr, cutoff_layer should remain unchanged
  int original_cutoff_layer = model_input->cutoff_layer;
  std::vector<ForwardRequest> forward_reqs_null(1);
  forward_reqs_null[0].request_target = nullptr;
  model_input->PrepareCutoffLayer(forward_reqs_null);
  EXPECT_EQ(model_input->cutoff_layer, original_cutoff_layer);

  // Test case 2: When request_target has empty cutoff_layer, use num_layer
  model_input->cutoff_layer = 0;
  ksana_llm::TargetDescribe target_desc_empty;
  target_desc_empty.cutoff_layer.clear();
  std::unordered_map<std::string, ksana_llm::TargetDescribe> targets_empty = {{"lm_head", target_desc_empty}};
  auto req_targets_empty = std::make_shared<decltype(targets_empty)>(targets_empty);
  ForwardRequest req_empty;
  req_empty.request_target = std::make_shared<const decltype(targets_empty)>(*req_targets_empty.get());
  std::vector<ForwardRequest> forward_reqs_empty = {req_empty};
  model_input->model_config_.num_layer = 42;
  model_input->PrepareCutoffLayer(forward_reqs_empty);
  EXPECT_EQ(model_input->cutoff_layer, 42);

  model_input->cutoff_layer = 0;
  ksana_llm::TargetDescribe target_desc;
  target_desc.cutoff_layer = {3, 7, 5};
  std::unordered_map<std::string, ksana_llm::TargetDescribe> targets = {{"lm_head", target_desc}};
  auto req_targets = std::make_shared<decltype(targets)>(targets);
  ForwardRequest req;
  req.request_target = std::make_shared<const decltype(targets)>(*req_targets.get());
  std::vector<ForwardRequest> forward_reqs = {req};
  model_input->PrepareCutoffLayer(forward_reqs);
  EXPECT_EQ(model_input->cutoff_layer, 7);
}

#ifdef ENABLE_CUDA
// Test PrepareFlashRotary with flexible caching
TEST_F(ModelInputTest, PrepareFlashRotaryMlaFlexibleCacheTest) {
  // Dummy input req
  const int block_token_num = 2;
  const int batch_size = 2;
  int dummy_block_memory_id = 0;
  std::vector<int> forwarding_tokens_1 = {0, 1, 2, 4, 5, 6, 7};     // hit {0, 1, 2, 3, 4, 5}
  std::vector<int> forwarding_tokens_2 = {10, 11, 14, 15, 16, 17};  // hit {10, 11, 12, 13, 14, 15}
  std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks_1(3);
  std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks_2(2);
  flexible_cached_copy_tasks_1[0].Update(2, 2, dummy_block_memory_id, dummy_block_memory_id);
  flexible_cached_copy_tasks_1[1].Update(3, 4, dummy_block_memory_id, dummy_block_memory_id);
  flexible_cached_copy_tasks_1[2].Update(4, 5, dummy_block_memory_id, dummy_block_memory_id);
  flexible_cached_copy_tasks_2[0].Update(2, 4, dummy_block_memory_id, dummy_block_memory_id);
  flexible_cached_copy_tasks_2[1].Update(3, 5, dummy_block_memory_id, dummy_block_memory_id);
  // Construct forward requests as test input.
  std::vector<ForwardRequest> forward_reqs(batch_size);
  forward_reqs[0].forwarding_tokens = std::make_shared<std::vector<int>>(forwarding_tokens_1);
  forward_reqs[1].forwarding_tokens = std::make_shared<std::vector<int>>(forwarding_tokens_2);
  forward_reqs[0].kv_cached_token_num = 2;
  forward_reqs[1].kv_cached_token_num = 2;
  // flexible_cache_len 是 flexible_cached_copy_tasks 的大小
  forward_reqs[0].flexible_cache_len = flexible_cached_copy_tasks_1.size();
  forward_reqs[1].flexible_cache_len = flexible_cached_copy_tasks_2.size();
  // prefix_cache_len = kv_cached_token_num + flexible_cache_len
  forward_reqs[0].prefix_cache_len = forward_reqs[0].kv_cached_token_num + forward_reqs[0].flexible_cache_len;
  forward_reqs[1].prefix_cache_len = forward_reqs[1].kv_cached_token_num + forward_reqs[1].flexible_cache_len;
  forward_reqs[0].flexible_cached_copy_tasks = &flexible_cached_copy_tasks_1;
  forward_reqs[1].flexible_cached_copy_tasks = &flexible_cached_copy_tasks_2;

  const size_t total_tokens = forwarding_tokens_1.size() + forwarding_tokens_2.size();
  const size_t total_prefix_len = forward_reqs[0].prefix_cache_len + forward_reqs[1].prefix_cache_len;
  const size_t total_flexible_len = flexible_cached_copy_tasks_1.size() + flexible_cached_copy_tasks_2.size();

  const auto& env = Singleton<Environment>::GetInstance();
  CacheManagerConfig cache_manager_config;
  env->GetCacheManagerConfig(cache_manager_config);
  cache_manager_config.enable_prefix_caching = true;
  cache_manager_config.min_flexible_cache_num = block_token_num;
  env->SetCacheManagerConfig(cache_manager_config);
  env->UpdateModelConfig();

  RuntimeConfig runtime_config;
  env->GetRuntimeConfig(runtime_config);
  EXPECT_TRUE(runtime_config.enable_prefix_caching);
  EXPECT_TRUE(runtime_config.enable_flexible_caching);

  EXPECT_TRUE(model_input->model_config_.use_mla);
  EXPECT_TRUE(model_input->enable_blocked_multi_token_forwarding_kv_);
  model_input->dp_total_prefix_len = total_prefix_len;

  // 确保所有 flexible caching 相关的 tensor 都有正确分配的内存
  // 这些 tensor 在 ModelInput 构造函数中只有当 runtime_config_.enable_flexible_caching 为 true 时才会被创建
  // 但由于构造函数从环境中重新获取配置，可能导致这些 tensor 没有被正确初始化
  auto& test_shared = *model_input->shared_attention_meta->GetStandard();
  if (test_shared.dp_dst_flexible_kv_cache_tensor.GetPtr<void>(false) == nullptr && total_flexible_len > 0) {
    test_shared.dp_dst_flexible_kv_cache_tensor.ReallocateMemory(MemoryLocation::LOCATION_DEVICE, TYPE_POINTER,
                                                                 {total_flexible_len});
  }
  // 设置 shape 以触发 PrepareFlashRotary 中的 flexible cache 处理逻辑
  test_shared.dp_dst_flexible_kv_cache_tensor.shape = {total_flexible_len};

  // 确保 dp_flexible_rotary_embedding_mask 有正确分配的内存
  if (test_shared.dp_flexible_rotary_embedding_mask.GetPtr<void>(false) == nullptr && total_tokens > 0) {
    test_shared.dp_flexible_rotary_embedding_mask.ReallocateMemory(MemoryLocation::LOCATION_DEVICE, TYPE_INT64,
                                                                   {total_tokens});
  }

  // 确保 dp_src_flexible_rotary_embedding_pos 有正确分配的内存
  if (test_shared.dp_src_flexible_rotary_embedding_pos.GetPtr<void>(false) == nullptr && total_tokens > 0) {
    test_shared.dp_src_flexible_rotary_embedding_pos.ReallocateMemory(MemoryLocation::LOCATION_DEVICE, TYPE_INT64,
                                                                      {total_tokens});
  }

  // 确保 dp_dst_flexible_rotary_embedding_pos 有正确分配的内存 (仅当 use_mla 为 true 时需要)
  if (model_input->model_config_.use_mla &&
      test_shared.dp_dst_flexible_rotary_embedding_pos.GetPtr<void>(false) == nullptr && total_tokens > 0) {
    test_shared.dp_dst_flexible_rotary_embedding_pos.ReallocateMemory(MemoryLocation::LOCATION_DEVICE, TYPE_INT64,
                                                                      {total_tokens});
  }

  model_input->runtime_config_ = runtime_config;
  model_input->PrepareInputInfo(forward_reqs);
  EXPECT_EQ(model_input->flash_meta->dp_reqs.size(), forward_reqs.size());
  auto* flash_builder = static_cast<StandardFlashAttentionMetaBuilder*>(model_input->flash_builder_.get());
  auto& shared_meta = *model_input->shared_attention_meta;
  flash_builder->PrepareFlashRotary(*model_input->flash_meta, shared_meta, model_input->dp_total_prefix_len);

  // verify rotary_embedding_pos and rotary_embedding_mask
  std::vector<int64_t> host_rotary_embedding_pos(total_tokens - total_prefix_len);
  std::vector<int64_t> host_rotary_embedding_mask(total_tokens - total_prefix_len);
  std::vector<int64_t> host_src_flexible_rotary_embedding_pos(total_tokens);
  std::vector<int64_t> host_dst_flexible_rotary_embedding_pos(total_tokens);
  std::vector<int64_t> host_flexible_rotary_embedding_mask(total_tokens);

  Memcpy(host_rotary_embedding_pos.data(), model_input->flash_meta->rotary_embedding_pos.GetPtr<void>(),
         (total_tokens - total_prefix_len) * sizeof(int64_t), MEMCPY_DEVICE_TO_HOST);
  Memcpy(host_rotary_embedding_mask.data(), model_input->flash_meta->rotary_embedding_mask.GetPtr<void>(),
         (total_tokens - total_prefix_len) * sizeof(int64_t), MEMCPY_DEVICE_TO_HOST);
  Memcpy(host_src_flexible_rotary_embedding_pos.data(), test_shared.dp_src_flexible_rotary_embedding_pos.GetPtr<void>(),
         total_tokens * sizeof(int64_t), MEMCPY_DEVICE_TO_HOST);
  Memcpy(host_dst_flexible_rotary_embedding_pos.data(), test_shared.dp_dst_flexible_rotary_embedding_pos.GetPtr<void>(),
         total_tokens * sizeof(int64_t), MEMCPY_DEVICE_TO_HOST);
  Memcpy(host_flexible_rotary_embedding_mask.data(), test_shared.dp_flexible_rotary_embedding_mask.GetPtr<void>(),
         total_tokens * sizeof(int64_t), MEMCPY_DEVICE_TO_HOST);

  int rope_idx = 0;
  int flexible_rope_idx = 0;
  for (int i = 0; i < batch_size; ++i) {
    int context_forwarding_token_num = forward_reqs[i].forwarding_tokens->size() - forward_reqs[i].prefix_cache_len;
    for (int idx = 0; idx < context_forwarding_token_num; ++idx) {
      EXPECT_EQ(host_rotary_embedding_pos[rope_idx], idx + forward_reqs[i].prefix_cache_len);
      EXPECT_EQ(host_rotary_embedding_mask[rope_idx], 1);
      rope_idx++;
    }

    for (size_t idx = 0; idx < forward_reqs[i].flexible_cached_copy_tasks->size(); ++idx) {
      auto& task = forward_reqs[i].flexible_cached_copy_tasks->at(idx);
      EXPECT_EQ(host_src_flexible_rotary_embedding_pos[flexible_rope_idx + task.dst_token_idx_], task.src_token_idx_);
      EXPECT_EQ(host_dst_flexible_rotary_embedding_pos[flexible_rope_idx + task.dst_token_idx_], task.dst_token_idx_);
    }

    for (size_t idx = 0; idx < forward_reqs[i].forwarding_tokens->size(); ++idx) {
      if (static_cast<int>(idx) >= forward_reqs[i].kv_cached_token_num &&
          static_cast<int>(idx) < forward_reqs[i].prefix_cache_len) {
        EXPECT_EQ(host_flexible_rotary_embedding_mask[flexible_rope_idx], 1);
      } else {
        EXPECT_EQ(host_flexible_rotary_embedding_mask[flexible_rope_idx], 0);
      }
      flexible_rope_idx++;
    }
  }
}
#endif

#ifdef ENABLE_CUDA
TEST_F(ModelInputTest, PrepareFlashMlaTest) {
  model_input->page_metas.clear();
  model_input->page_metas.emplace_back(model_input->attention_meta_creator_->CreatePagedAttentionMeta());
  // Creator is DeepSeekV3AttentionMetaCreator (injected in SetUp when use_mla=true),
  // so page_meta is actually a DeepSeekV3PagedAttentionMeta.
  auto& page_meta = static_cast<DeepSeekV3PagedAttentionMeta&>(*model_input->page_metas.back());
  page_meta.q_seq_len = 1;

  auto& shared_meta = *model_input->shared_attention_meta;
  auto& standard_shared = *shared_meta.GetStandard();
  auto& mla_shared = *shared_meta.GetMLA();
  auto& dsa_shared = *shared_meta.GetDSA();
  page_meta.input_length = standard_shared.input_length;
  mla_shared.num_splits.shape = {0};

  model_input->model_config_.use_mla = true;
  auto builder_config = MakeAttentionBuilderConfig();

  // 测试用例1: 当 dp_reqs 为空时，PrepareFlashMla 应该直接返回（空指针保护）
  ASSERT_TRUE(page_meta.dp_reqs.empty());
  PrepareFlashMla(page_meta.mla, page_meta.dsa, mla_shared, dsa_shared, static_cast<StandardAttentionMeta&>(page_meta),
                  builder_config);
  // 确保不崩溃，num_splits 未被修改
  EXPECT_EQ(mla_shared.num_splits.shape[0], 0u);

  // 测试用例2: 当所有条件满足时，PrepareFlashMla 应该填充 mla_meta
  model_input->model_config_.head_num = 16;
  model_input->runtime_config_.parallel_basic_config.tensor_parallel_size = 1;
  page_meta.dp_reqs.resize(4);

  std::vector<int> input_lengths = {0, 20, 30, 40, 50};
  MemcpyAsync(page_meta.input_length.GetPtr<void>(), input_lengths.data(), input_lengths.size() * sizeof(int),
              MEMCPY_HOST_TO_DEVICE, model_input->context_->device->GetH2DStream());

  PrepareFlashMla(page_meta.mla, page_meta.dsa, mla_shared, dsa_shared, static_cast<StandardAttentionMeta&>(page_meta),
                  builder_config);

  llm_kernels::nvidia::FlashMlaWorkspaceMap flash_mla_workspace_map;
  llm_kernels::nvidia::GetNumSmParts(flash_mla_workspace_map, 16, 1, 0);
  if (flash_mla_workspace_map.num_sm_parts > 1) {
    int num_splits_cpu;
    MemcpyAsync(&num_splits_cpu, page_meta.mla.num_splits.GetPtr<void>(), sizeof(int), MEMCPY_DEVICE_TO_HOST,
                model_input->context_->device->GetH2DStream());
    StreamSynchronize(model_input->context_->device->GetH2DStream());
    EXPECT_EQ(num_splits_cpu, 0);
  }
}
#endif

#ifdef ENABLE_CUDA
TEST_F(ModelInputTest, PrepareNextNGatherIdxTest) {
  auto RandomNum = [](const size_t min, const size_t max) {
    static std::default_random_engine random_engine;
    return std::uniform_int_distribution<size_t>(min, max)(random_engine);
  };

  constexpr size_t kReqNum = 10;
  constexpr size_t kMaxReqLen = 1024;

  std::vector<ForwardRequest> forward_reqs(kReqNum);
  std::vector<std::vector<int>> req_tokens(kReqNum);
  for (size_t i = 0; i < kReqNum; ++i) {
    req_tokens[i].resize(RandomNum(0, kMaxReqLen));
    forward_reqs[i].forwarding_tokens = std::make_shared<std::vector<int>>(req_tokens[i]);
    forward_reqs[i].kv_cached_token_num = RandomNum(0, req_tokens[i].size());
    forward_reqs[i].req_id = i;
  }

  model_input->runtime_config_.mtp_step_num = 1;
  model_input->model_config_.dspark_config.dspark_block_size = 0;

  // Use MtpGatherIndexBuilder to exercise the current gather index pipeline.
  // The builder is the replacement for the old PrepareNextNGatherIdx logic:
  // - UpdateMainForward records each request's start position
  // - BuildGatherIndices generates the gather tensor from those positions
  const size_t max_token_num = kReqNum * kMaxReqLen;
  MtpGatherIndexBuilder builder(max_token_num, context_, /*rank=*/0);
  model_input->SetExternalGatherIdxBuilder(&builder);

  model_input->builder_config_->run_mode = RunMode::kMain;
  builder.UpdateMainForward(forward_reqs);

  size_t total_len = 0;
  for (size_t i = 0; i < forward_reqs.size(); ++i) {
    total_len += forward_reqs[i].forwarding_tokens->size() - forward_reqs[i].kv_cached_token_num;
  }

  model_input->builder_config_->run_mode = RunMode::kNextN;
  builder.BuildGatherIndices(forward_reqs);
  EXPECT_EQ(builder.GetGatherTensor()->shape.size(), 1);
  EXPECT_EQ(builder.GetGatherTensor()->shape[0], total_len);

  std::vector<size_t> host_idx_result(total_len);
  Memcpy(host_idx_result.data(), builder.GetGatherTensor()->GetPtr<void>(), total_len * sizeof(size_t),
         MEMCPY_DEVICE_TO_HOST);
  size_t result_i = 0, counter_i = 0;
  for (size_t i = 0; i < forward_reqs.size(); ++i) {
    const auto& req = forward_reqs[i];
    for (size_t token_i = 0; token_i < req.forwarding_tokens->size(); ++token_i) {
      if (token_i < static_cast<size_t>(req.kv_cached_token_num)) {
        continue;
      }
      EXPECT_EQ(host_idx_result[result_i++], counter_i++);
    }
  }
  EXPECT_EQ(host_idx_result.size(), result_i);
}
#endif

TEST_F(ModelInputTest, PrepareMRopePosTest) {
  // Set the model type to qwen2_vl to enable MRoPE tensor creation
  model_input->model_config_.type = "qwen2_vl";
  model_input->model_config_.rope_scaling_factor_config.mrope_section = std::vector<int>{16, 24, 24};
  model_input->CreateVLTensors();

  auto VerifyPrepareMRopePos = [&](const std::vector<ForwardRequest>& forward_reqs,
                                   const std::vector<int64_t>& expected_mrotary_embedding_pos,
                                   const std::vector<int64_t>& expected_offsets) {
    EXPECT_EQ(model_input->dp_mrotary_embedding_pos.shape.size(), 2);
    EXPECT_GE(model_input->dp_mrotary_embedding_pos.shape[1], expected_mrotary_embedding_pos.size());

    // Verify the offsets.
    EXPECT_EQ(expected_offsets.size(), forward_reqs.size());
    for (size_t i = 0; i < forward_reqs.size(); i++) {
      EXPECT_EQ(*forward_reqs[i].mrotary_embedding_pos_offset, expected_offsets[i]);
    }

    // Verify the mrotary_embedding_pos tensor.
    const size_t total_pos_count = expected_mrotary_embedding_pos.size();
    std::vector<int64_t> actual_mrotary_embedding_pos(total_pos_count);
    Memcpy(actual_mrotary_embedding_pos.data(), model_input->dp_mrotary_embedding_pos.GetPtr<void>(),
           total_pos_count * sizeof(int64_t), MEMCPY_DEVICE_TO_HOST);
    for (size_t i = 0; i < total_pos_count; i++) {
      EXPECT_EQ(actual_mrotary_embedding_pos[i], expected_mrotary_embedding_pos[i]);
    }
  };

  // Test for each selected batch size.
  for (const int batch_size : {1, 3, 4}) {
    std::vector<ForwardRequest> forward_reqs;
    std::vector<std::vector<int>> output_tokens;
    std::vector<EmbeddingSlice> embedding_slices;
    std::vector<int64_t> mrotary_offsets;
    // Store torch tensors to keep them alive during the test
    std::vector<torch::Tensor> pos_tensors_storage;
    std::vector<torch::Tensor> offset_tensors_storage;

    // Reserve memory to avoid memory address being moved.
    forward_reqs.reserve(batch_size);
    output_tokens.reserve(batch_size);
    embedding_slices.reserve(batch_size);
    mrotary_offsets.reserve(batch_size);
    pos_tensors_storage.reserve(batch_size);
    offset_tensors_storage.reserve(batch_size);

    std::vector<int64_t> expected_mrotary_embedding_pos;
    std::vector<int64_t> expected_offsets;

    // Create a mix of plain text and visual inputs
    for (int i = 0; i < batch_size; i++) {
      ForwardRequest forward_req;
      const size_t token_size = 10 + i * 5;
      output_tokens.emplace_back(token_size);
      forward_req.forwarding_tokens = std::make_shared<std::vector<int>>(output_tokens.back());
      EmbeddingSlice embedding_slice;
      embedding_slices.push_back(std::move(embedding_slice));
      forward_req.input_refit_embedding = &embedding_slices.back();
      mrotary_offsets.push_back(0);
      forward_req.mrotary_embedding_pos_offset = &mrotary_offsets.back();

      // Alternate between plain text and visual input
      if (i % 2 == 0) {
        // Plain text input (empty additional_tensors)
        // For plain text, the function creates positions where each triplet is [i, i, i]
        int64_t num_tokens = forward_req.forwarding_tokens->size();
        for (int64_t j = 0; j < num_tokens; ++j) {
          expected_mrotary_embedding_pos.push_back(j);
          expected_mrotary_embedding_pos.push_back(j);
          expected_mrotary_embedding_pos.push_back(j);
        }
        expected_offsets.push_back(0);
      } else {
        // Visual input (non-empty additional_tensors)
        // Create position tensor with deterministic values for testing
        torch::Tensor pos_tensor = torch::randint(0, 100, {static_cast<int64_t>(token_size * 3)}, torch::kInt64);
        int64_t offset_value = i * 10;
        torch::Tensor offset_tensor = torch::tensor(offset_value, torch::kInt64);
        // Store tensors to keep them alive
        pos_tensors_storage.push_back(pos_tensor);
        offset_tensors_storage.push_back(offset_tensor);
        auto pos_accessor = pos_tensor.data_ptr<int64_t>();
        for (int64_t j = 0; j < pos_tensor.numel(); j++) {
          expected_mrotary_embedding_pos.push_back(pos_accessor[j]);
        }
        expected_offsets.push_back(offset_value);
        // Add tensors to additional_tensors (convert torch::Tensor to ksana_llm::Tensor)
        Tensor ksana_pos_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT64,
                                {static_cast<size_t>(token_size * 3)}, pos_tensors_storage.back().data_ptr(),
                                "pos_tensor");
        Tensor ksana_offset_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT64, {1},
                                   offset_tensors_storage.back().data_ptr(), "offset_tensor");
        forward_req.input_refit_embedding->additional_tensors.push_back(ksana_pos_tensor);
        forward_req.input_refit_embedding->additional_tensors.push_back(ksana_offset_tensor);
      }
      forward_reqs.push_back(std::move(forward_req));
    }

    // Parse MRopePos.
    model_input->PrepareMRopePos(forward_reqs);

    // Check the result of VerifyPrepareMRopePos.
    VerifyPrepareMRopePos(forward_reqs, expected_mrotary_embedding_pos, expected_offsets);
  }

  // Construct bad input.
  {
    std::vector<ForwardRequest> forward_reqs(1);
    std::vector<std::vector<int>> output_tokens(1, std::vector<int>(10));
    std::vector<EmbeddingSlice> embedding_slices(1);
    std::vector<int64_t> mrotary_offsets(1, 0);

    forward_reqs[0].forwarding_tokens = std::make_shared<std::vector<int>>(output_tokens[0]);
    forward_reqs[0].input_refit_embedding = &embedding_slices[0];
    forward_reqs[0].mrotary_embedding_pos_offset = &mrotary_offsets[0];

    {
      torch::Tensor pos_tensor = torch::randint(0, 100, {30}, torch::kInt64);
      forward_reqs[0].input_refit_embedding->additional_tensors.clear();
      // Convert torch::Tensor to ksana_llm::Tensor
      Tensor ksana_pos_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT64, {30}, pos_tensor.data_ptr(),
                              "pos_tensor");
      forward_reqs[0].input_refit_embedding->additional_tensors.push_back(ksana_pos_tensor);

      EXPECT_THROW(
          try { model_input->PrepareMRopePos(forward_reqs); } catch (const std::runtime_error& e) {
            EXPECT_NE(strstr(e.what(), "additional_tensors must contain at least 2 tensors"), nullptr);
            throw;
          },
          std::runtime_error);
    }
  }
}

TEST_F(ModelInputTest, PrepareXDRopePosTest) {
  // Set the model type to arc_hunyuan_video to enable XDRoPE tensor creation
  model_input->model_config_.type = "arc_hunyuan_video";
  model_input->CreateVLTensors();

  auto VerifyPrepareXDRopePos = [&](const std::vector<ForwardRequest>& forward_reqs,
                                    const std::vector<int64_t>& expected_xdrotary_embedding_pos,
                                    const std::vector<int64_t>& expected_offsets) {
    EXPECT_EQ(model_input->dp_xdrotary_embedding_pos.shape.size(), 2);
    EXPECT_GE(model_input->dp_xdrotary_embedding_pos.shape[1], expected_xdrotary_embedding_pos.size());

    // Verify the offsets.
    EXPECT_EQ(expected_offsets.size(), forward_reqs.size());
    for (size_t i = 0; i < forward_reqs.size(); i++) {
      EXPECT_EQ(*forward_reqs[i].xdrotary_embedding_pos_offset, expected_offsets[i]);
    }

    // Verify the xdrotary_embedding_pos tensor.
    const size_t total_pos_count = expected_xdrotary_embedding_pos.size();
    std::vector<int64_t> actual_xdrotary_embedding_pos(total_pos_count);
    Memcpy(actual_xdrotary_embedding_pos.data(), model_input->dp_xdrotary_embedding_pos.GetPtr<void>(),
           total_pos_count * sizeof(int64_t), MEMCPY_DEVICE_TO_HOST);
    for (size_t i = 0; i < total_pos_count; i++) {
      EXPECT_EQ(actual_xdrotary_embedding_pos[i], expected_xdrotary_embedding_pos[i]);
    }
  };

  // Test for each selected batch size.
  for (const int batch_size : {1, 3, 4}) {
    std::vector<ForwardRequest> forward_reqs;
    std::vector<std::vector<int>> output_tokens;
    std::vector<EmbeddingSlice> embedding_slices;
    std::vector<int64_t> xdrotary_offsets;
    // Store torch tensors to keep them alive during the test
    std::vector<torch::Tensor> pos_tensors_storage;
    std::vector<torch::Tensor> offset_tensors_storage;

    // Reserve memory to avoid memory address being moved.
    forward_reqs.reserve(batch_size);
    output_tokens.reserve(batch_size);
    embedding_slices.reserve(batch_size);
    xdrotary_offsets.reserve(batch_size);
    pos_tensors_storage.reserve(batch_size);
    offset_tensors_storage.reserve(batch_size);

    std::vector<int64_t> expected_xdrotary_embedding_pos;
    std::vector<int64_t> expected_offsets;

    // XDRoPE uses factor of 4 (compared to MRoPE's factor of 3)
    constexpr int kXDRotaryEmbeddingPosFactor = 4;

    // Create a mix of plain text and visual inputs
    for (int i = 0; i < batch_size; i++) {
      ForwardRequest forward_req;
      const size_t token_size = 10 + i * 5;
      output_tokens.emplace_back(token_size);
      forward_req.forwarding_tokens = std::make_shared<std::vector<int>>(output_tokens.back());
      EmbeddingSlice embedding_slice;
      embedding_slices.push_back(std::move(embedding_slice));
      forward_req.input_refit_embedding = &embedding_slices.back();
      xdrotary_offsets.push_back(0);
      forward_req.xdrotary_embedding_pos_offset = &xdrotary_offsets.back();

      // Alternate between plain text and visual input
      if (i % 2 == 0) {
        // Plain text input (empty additional_tensors)
        // For plain text, the function creates positions where each quadruplet is [i, i, i, i]
        int64_t list_size = forward_req.forwarding_tokens->size() * kXDRotaryEmbeddingPosFactor;
        for (int64_t j = 0; j < list_size; j += kXDRotaryEmbeddingPosFactor) {
          for (int t = 0; t < kXDRotaryEmbeddingPosFactor; ++t) {
            expected_xdrotary_embedding_pos.push_back(j);
          }
        }
        expected_offsets.push_back(0);
      } else {
        // Visual input (non-empty additional_tensors)
        // Create position tensor with deterministic values for testing
        torch::Tensor pos_tensor =
            torch::randint(0, 100, {static_cast<int64_t>(token_size * kXDRotaryEmbeddingPosFactor)}, torch::kInt64);
        int64_t offset_value = i * 10;
        torch::Tensor offset_tensor = torch::tensor(offset_value, torch::kInt64);
        // Store tensors to keep them alive
        pos_tensors_storage.push_back(pos_tensor);
        offset_tensors_storage.push_back(offset_tensor);
        auto pos_accessor = pos_tensor.data_ptr<int64_t>();
        for (int64_t j = 0; j < pos_tensor.numel(); j++) {
          expected_xdrotary_embedding_pos.push_back(pos_accessor[j]);
        }
        expected_offsets.push_back(offset_value);
        // Add tensors to additional_tensors (convert torch::Tensor to ksana_llm::Tensor)
        Tensor ksana_pos_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT64,
                                {static_cast<size_t>(token_size * kXDRotaryEmbeddingPosFactor)},
                                pos_tensors_storage.back().data_ptr(), "pos_tensor");
        Tensor ksana_offset_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT64, {1},
                                   offset_tensors_storage.back().data_ptr(), "offset_tensor");
        forward_req.input_refit_embedding->additional_tensors.push_back(ksana_pos_tensor);
        forward_req.input_refit_embedding->additional_tensors.push_back(ksana_offset_tensor);
      }
      forward_reqs.push_back(std::move(forward_req));
    }

    // Parse XDRopePos.
    model_input->PrepareXDRopePos(forward_reqs);

    // Check the result of VerifyPrepareXDRopePos.
    VerifyPrepareXDRopePos(forward_reqs, expected_xdrotary_embedding_pos, expected_offsets);
  }

  // Construct bad input.
  {
    std::vector<ForwardRequest> forward_reqs(1);
    std::vector<std::vector<int>> output_tokens(1, std::vector<int>(10));
    std::vector<EmbeddingSlice> embedding_slices(1);
    std::vector<int64_t> xdrotary_offsets(1, 0);

    forward_reqs[0].forwarding_tokens = std::make_shared<std::vector<int>>(output_tokens[0]);
    forward_reqs[0].input_refit_embedding = &embedding_slices[0];
    forward_reqs[0].xdrotary_embedding_pos_offset = &xdrotary_offsets[0];

    {
      torch::Tensor pos_tensor = torch::randint(0, 100, {40}, torch::kInt64);
      forward_reqs[0].input_refit_embedding->additional_tensors.clear();
      // Convert torch::Tensor to ksana_llm::Tensor
      Tensor ksana_pos_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT64, {40}, pos_tensor.data_ptr(),
                              "pos_tensor");
      forward_reqs[0].input_refit_embedding->additional_tensors.push_back(ksana_pos_tensor);

      EXPECT_THROW(
          try { model_input->PrepareXDRopePos(forward_reqs); } catch (const std::runtime_error& e) {
            EXPECT_NE(strstr(e.what(), "additional_tensors must contain 2 tensors"), nullptr);
            throw;
          },
          std::runtime_error);
    }
  }
}

TEST_F(ModelInputTest, PrepareDeepstackTest) {
  // Configure model as qwen3_vl with DeepStack.
  constexpr int kNumDeepstackLayers = 2;
  constexpr int kHiddenSize = 16;
  model_input->model_config_.type = "qwen3_vl";
  model_input->model_config_.is_visual = true;
  model_input->model_config_.num_deepstack_layers = kNumDeepstackLayers;
  model_input->model_config_.hidden_units = kHiddenSize;
  model_input->model_config_.weight_data_type = TYPE_FP32;
  model_input->CreateVLTensors();

  // Verify pre-allocation in CreateVLTensors.
  EXPECT_EQ(model_input->deepstack_data.num_layers, kNumDeepstackLayers);
  EXPECT_EQ(model_input->deepstack_data.layer_embeds.size(), static_cast<size_t>(kNumDeepstackLayers));
  for (int i = 0; i < kNumDeepstackLayers; ++i) {
    EXPECT_EQ(model_input->deepstack_data.layer_embeds[i].shape.size(), 2u);
    EXPECT_EQ(model_input->deepstack_data.layer_embeds[i].shape[1], static_cast<size_t>(kHiddenSize));
  }

  // ---- Test case 1: No visual requests → deepstack_data should be inactive ----
  {
    std::vector<ForwardRequest> forward_reqs(1);
    std::vector<int> tokens(20);
    EmbeddingSlice embedding_slice;
    forward_reqs[0].forwarding_tokens = std::make_shared<std::vector<int>>(tokens);
    forward_reqs[0].input_refit_embedding = &embedding_slice;
    // kv_cached_token_num = 0, so GetType() returns kFlash but additional_tensors is empty (size < 5).

    model_input->PrepareDeepstack(forward_reqs);
    EXPECT_FALSE(model_input->deepstack_data.active);
    EXPECT_TRUE(model_input->deepstack_data.segments.empty());
    EXPECT_TRUE(model_input->deepstack_data.layer_to_deepstack_idx.empty());
  }

  // ---- Test case 2: Decode requests (kPage) → deepstack_data should be inactive ----
  {
    std::vector<ForwardRequest> forward_reqs(1);
    std::vector<int> tokens(5);
    EmbeddingSlice embedding_slice;
    forward_reqs[0].forwarding_tokens = std::make_shared<std::vector<int>>(tokens);
    forward_reqs[0].input_refit_embedding = &embedding_slice;
    forward_reqs[0].kv_cached_token_num = 4;  // GetInputIdsLength() = 1, kv_cached > 0 → kPage

    model_input->PrepareDeepstack(forward_reqs);
    EXPECT_FALSE(model_input->deepstack_data.active);
  }

  // ---- Test case 3: Single visual request → verify segments and layer mapping ----
  {
    constexpr int kTotalVisualTokens = 4;
    constexpr int kNumSegments = 2;
    const int32_t layer_ids[kNumDeepstackLayers] = {0, 3};
    const int32_t seg_starts[kNumSegments] = {2, 8};
    const int32_t seg_lengths[kNumSegments] = {2, 2};

    std::vector<ForwardRequest> forward_reqs(1);
    std::vector<int> tokens(20);
    EmbeddingSlice embedding_slice;
    forward_reqs[0].forwarding_tokens = std::make_shared<std::vector<int>>(tokens);
    forward_reqs[0].input_refit_embedding = &embedding_slice;

    // Build additional_tensors: [0]=mrope_pos, [1]=mrope_delta, [2]=seg_starts, [3]=seg_lengths,
    //                           [4]=layer_ids, [5]=embed_layer_0, [6]=embed_layer_1
    // Slots [0] and [1] are MRoPE-related, kept as placeholders.
    torch::Tensor dummy_pos = torch::zeros({static_cast<int64_t>(tokens.size() * 3)}, torch::kInt64);
    torch::Tensor dummy_delta = torch::zeros({1}, torch::kInt64);
    torch::Tensor seg_starts_t =
        torch::from_blob(const_cast<int32_t*>(seg_starts), {kNumSegments}, torch::kInt32).clone();
    torch::Tensor seg_lengths_t =
        torch::from_blob(const_cast<int32_t*>(seg_lengths), {kNumSegments}, torch::kInt32).clone();
    torch::Tensor layer_ids_t =
        torch::from_blob(const_cast<int32_t*>(layer_ids), {kNumDeepstackLayers}, torch::kInt32).clone();

    // Create per-layer embed tensors [total_visual_tokens, hidden_size] with known values.
    torch::Tensor embed_layer0 = torch::ones({kTotalVisualTokens, kHiddenSize}, torch::kFloat32) * 1.0f;
    torch::Tensor embed_layer1 = torch::ones({kTotalVisualTokens, kHiddenSize}, torch::kFloat32) * 2.0f;

    auto make_tensor = [](torch::Tensor& t, DataType dtype) {
      std::vector<size_t> shape;
      for (int64_t i = 0; i < t.dim(); ++i) shape.push_back(static_cast<size_t>(t.size(i)));
      return Tensor(MemoryLocation::LOCATION_HOST, dtype, shape, t.data_ptr(), "test");
    };

    embedding_slice.additional_tensors.push_back(make_tensor(dummy_pos, TYPE_INT64));
    embedding_slice.additional_tensors.push_back(make_tensor(dummy_delta, TYPE_INT64));
    embedding_slice.additional_tensors.push_back(make_tensor(seg_starts_t, TYPE_INT32));
    embedding_slice.additional_tensors.push_back(make_tensor(seg_lengths_t, TYPE_INT32));
    embedding_slice.additional_tensors.push_back(make_tensor(layer_ids_t, TYPE_INT32));
    embedding_slice.additional_tensors.push_back(make_tensor(embed_layer0, TYPE_FP32));
    embedding_slice.additional_tensors.push_back(make_tensor(embed_layer1, TYPE_FP32));

    model_input->PrepareDeepstack(forward_reqs);

    // Verify active state.
    EXPECT_TRUE(model_input->deepstack_data.active);

    // Verify layer mapping.
    EXPECT_EQ(model_input->deepstack_data.layer_to_deepstack_idx.size(), static_cast<size_t>(kNumDeepstackLayers));
    EXPECT_EQ(model_input->deepstack_data.layer_to_deepstack_idx[0], 0);
    EXPECT_EQ(model_input->deepstack_data.layer_to_deepstack_idx[3], 1);

    // Verify segments.
    EXPECT_EQ(model_input->deepstack_data.segments.size(), static_cast<size_t>(kNumSegments));
    // Segment 0: batch_pos = 0 + 2, embed_offset = 0 + 0, length = 2
    EXPECT_EQ(model_input->deepstack_data.segments[0].batch_pos, 2);
    EXPECT_EQ(model_input->deepstack_data.segments[0].embed_offset, 0);
    EXPECT_EQ(model_input->deepstack_data.segments[0].length, 2);
    // Segment 1: batch_pos = 0 + 8, embed_offset = 0 + 2, length = 2
    EXPECT_EQ(model_input->deepstack_data.segments[1].batch_pos, 8);
    EXPECT_EQ(model_input->deepstack_data.segments[1].embed_offset, 2);
    EXPECT_EQ(model_input->deepstack_data.segments[1].length, 2);
  }

  // ---- Test case 4: Multiple visual requests → verify batch_offset accumulation ----
  {
    constexpr int kTokens0 = 10;
    constexpr int kTokens1 = 15;
    constexpr int kVisual0 = 3;
    constexpr int kVisual1 = 2;

    const int32_t layer_ids[kNumDeepstackLayers] = {1, 5};
    const int32_t seg_starts0[1] = {4};
    const int32_t seg_lengths0[1] = {kVisual0};
    const int32_t seg_starts1[1] = {6};
    const int32_t seg_lengths1[1] = {kVisual1};

    std::vector<ForwardRequest> forward_reqs(2);
    std::vector<int> tokens0(kTokens0);
    std::vector<int> tokens1(kTokens1);
    EmbeddingSlice slice0, slice1;
    forward_reqs[0].forwarding_tokens = std::make_shared<std::vector<int>>(tokens0);
    forward_reqs[0].input_refit_embedding = &slice0;
    forward_reqs[1].forwarding_tokens = std::make_shared<std::vector<int>>(tokens1);
    forward_reqs[1].input_refit_embedding = &slice1;

    auto make_tensor = [](torch::Tensor& t, DataType dtype) {
      std::vector<size_t> shape;
      for (int64_t i = 0; i < t.dim(); ++i) shape.push_back(static_cast<size_t>(t.size(i)));
      return Tensor(MemoryLocation::LOCATION_HOST, dtype, shape, t.data_ptr(), "test");
    };

    // Keep tensors alive.
    std::vector<torch::Tensor> storage;

    auto build_additional_tensors = [&](EmbeddingSlice& slice, int token_count, const int32_t* starts,
                                        const int32_t* lengths, int num_segs, int num_visual) {
      storage.push_back(torch::zeros({static_cast<int64_t>(token_count * 3)}, torch::kInt64));
      storage.push_back(torch::zeros({1}, torch::kInt64));
      storage.push_back(torch::from_blob(const_cast<int32_t*>(starts), {num_segs}, torch::kInt32).clone());
      storage.push_back(torch::from_blob(const_cast<int32_t*>(lengths), {num_segs}, torch::kInt32).clone());
      storage.push_back(
          torch::from_blob(const_cast<int32_t*>(layer_ids), {kNumDeepstackLayers}, torch::kInt32).clone());
      for (int l = 0; l < kNumDeepstackLayers; ++l) {
        storage.push_back(torch::ones({num_visual, kHiddenSize}, torch::kFloat32) * static_cast<float>(l + 1));
      }
      size_t base = storage.size() - (5 + kNumDeepstackLayers);
      for (size_t i = base; i < storage.size(); ++i) {
        slice.additional_tensors.push_back(make_tensor(storage[i], i - base < 2 ? TYPE_INT64 : TYPE_INT32));
      }
      // Fix dtypes for embed tensors (last kNumDeepstackLayers).
      for (int l = 0; l < kNumDeepstackLayers; ++l) {
        slice.additional_tensors[5 + l].dtype = TYPE_FP32;
      }
    };

    build_additional_tensors(slice0, kTokens0, seg_starts0, seg_lengths0, 1, kVisual0);
    build_additional_tensors(slice1, kTokens1, seg_starts1, seg_lengths1, 1, kVisual1);

    model_input->PrepareDeepstack(forward_reqs);

    EXPECT_TRUE(model_input->deepstack_data.active);
    EXPECT_EQ(model_input->deepstack_data.segments.size(), 2u);

    // Request 0: batch_offset=0, seg batch_pos = 0 + 4 = 4
    EXPECT_EQ(model_input->deepstack_data.segments[0].batch_pos, 4);
    EXPECT_EQ(model_input->deepstack_data.segments[0].embed_offset, 0);
    EXPECT_EQ(model_input->deepstack_data.segments[0].length, kVisual0);

    // Request 1: batch_offset=10, seg batch_pos = 10 + 6 = 16
    EXPECT_EQ(model_input->deepstack_data.segments[1].batch_pos, kTokens0 + 6);
    EXPECT_EQ(model_input->deepstack_data.segments[1].embed_offset, kVisual0);
    EXPECT_EQ(model_input->deepstack_data.segments[1].length, kVisual1);

    // Verify layer mapping.
    EXPECT_EQ(model_input->deepstack_data.layer_to_deepstack_idx.at(1), 0);
    EXPECT_EQ(model_input->deepstack_data.layer_to_deepstack_idx.at(5), 1);
  }

  // ---- Test case 5: Bad input — insufficient additional_tensors → should throw ----
  {
    const int32_t layer_ids[kNumDeepstackLayers] = {0, 1};
    const int32_t seg_starts[1] = {0};
    const int32_t seg_lengths[1] = {2};

    std::vector<ForwardRequest> forward_reqs(1);
    std::vector<int> tokens(10);
    EmbeddingSlice embedding_slice;
    forward_reqs[0].forwarding_tokens = std::make_shared<std::vector<int>>(tokens);
    forward_reqs[0].input_refit_embedding = &embedding_slice;

    auto make_tensor = [](torch::Tensor& t, DataType dtype) {
      std::vector<size_t> shape;
      for (int64_t i = 0; i < t.dim(); ++i) shape.push_back(static_cast<size_t>(t.size(i)));
      return Tensor(MemoryLocation::LOCATION_HOST, dtype, shape, t.data_ptr(), "test");
    };

    torch::Tensor dummy_pos = torch::zeros({30}, torch::kInt64);
    torch::Tensor dummy_delta = torch::zeros({1}, torch::kInt64);
    torch::Tensor seg_starts_t = torch::from_blob(const_cast<int32_t*>(seg_starts), {1}, torch::kInt32).clone();
    torch::Tensor seg_lengths_t = torch::from_blob(const_cast<int32_t*>(seg_lengths), {1}, torch::kInt32).clone();
    torch::Tensor layer_ids_t =
        torch::from_blob(const_cast<int32_t*>(layer_ids), {kNumDeepstackLayers}, torch::kInt32).clone();
    // Only provide 5 tensors (missing per-layer embeds for 2 deepstack layers).
    embedding_slice.additional_tensors.push_back(make_tensor(dummy_pos, TYPE_INT64));
    embedding_slice.additional_tensors.push_back(make_tensor(dummy_delta, TYPE_INT64));
    embedding_slice.additional_tensors.push_back(make_tensor(seg_starts_t, TYPE_INT32));
    embedding_slice.additional_tensors.push_back(make_tensor(seg_lengths_t, TYPE_INT32));
    embedding_slice.additional_tensors.push_back(make_tensor(layer_ids_t, TYPE_INT32));
    // Missing embed tensors for 2 layers → should throw.

    EXPECT_THROW(
        try { model_input->PrepareDeepstack(forward_reqs); } catch (const std::runtime_error& e) {
          EXPECT_NE(strstr(e.what(), "additional_tensors must have at least 5 + num_deepstack_layers elements"),
                    nullptr);
          throw;
        },
        std::runtime_error);
  }
}

#if defined(ENABLE_ACL) || defined(ENABLE_KUNLUN)

class MockBlockAllocatorForATB : public BlockAllocatorInterface {
 public:
  explicit MockBlockAllocatorForATB(void* base_ptr) : base_ptr_(base_ptr) {}
  void PreAllocateBlocks() override {}
  void Clear() override {}
  Status AllocateBlocks(size_t block_num, std::vector<int>& blocks) override { return Status(); }
  Status FreeBlocks(const std::vector<int>& blocks) override { return Status(); }
  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) override { return Status(); }
  void* GetBlocksBasePtr() override { return base_ptr_; }
  int GetBlocksBaseId() override { return 0; }
  size_t GetFreeBlockNumber() override { return 0; }
  size_t GetUsedBlockNumber() override { return 0; }

 private:
  void* base_ptr_;
};

class MockBlockAllocatorGroupForATB : public BlockAllocatorGroupInterface {
 public:
  MockBlockAllocatorGroupForATB(size_t device_block_num, void* base_ptr)
      : config_({/*.devices=*/{0}, /*.device_block_num=*/device_block_num, /*.host_block_num=*/0, /*.block_size=*/0}),
        allocator_(std::make_shared<MockBlockAllocatorForATB>(base_ptr)) {}

  std::vector<int> GetBlockAllocatorDevices() const override { return config_.devices; }
  std::shared_ptr<BlockAllocatorInterface> GetHostBlockAllocator() const override { return nullptr; }
  std::shared_ptr<BlockAllocatorInterface> GetDeviceBlockAllocator(int index = 0) const override { return allocator_; }
  const BlockAllocatorGroupConfig& GetBlockAllocatorGroupConfig() const override { return config_; }

 private:
  BlockAllocatorGroupConfig config_;
  std::shared_ptr<BlockAllocatorInterface> allocator_;
};

class MockBlockAllocatorManagerForATB : public BlockAllocatorManagerInterface {
 public:
  MockBlockAllocatorManagerForATB(size_t device_block_num, void* base_ptr)
      : group_(std::make_shared<MockBlockAllocatorGroupForATB>(device_block_num, base_ptr)) {}

  void Initialize(const BlockAllocatorManagerConfig& config) override {}
  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup(int group_id) override { return group_; }

 private:
  std::shared_ptr<BlockAllocatorGroupInterface> group_;
};

// Test fixture for PrepareATBKVCache.
//
// Builds a throwaway yaml + config.json under the temp dir at SetUp (and removes them at TearDown),
// modeled on examples/ksana_llm_kimi_k26_kunlun.yaml but trimmed to a single-device DeepSeek-V3 MLA
// setup. This keeps the important knobs (tensor_para_size, block_token_num, DeepSeek-V3 model_type,
// num_attention_heads, kv_lora_rank, ...) visible in the test itself instead of depending on a real
// absolute model dir (e.g. /data1/klx/Kimi-K2.6-klx-w4a8-gptq) or an 8-device environment.
// See environment_test.cpp's ConnectorConfigInitialization for the same temp-yaml pattern.
class ATBKVCacheTest : public testing::Test {
 protected:
  void SetUp() override {
    int rank = 0;

    // 1) 在临时目录下生成一个最小化的 DeepSeek-V3 config.json 和一个单卡 yaml。
    // 路径中嵌入 PID 做隔离，避免同机并发 CI 进程互相覆盖/删除临时文件。
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    const std::string pid_suffix = std::to_string(getpid());
    temp_yaml_path_ = (temp_dir / ("ksana_atb_kvcache_test_" + pid_suffix + ".yaml")).string();
    temp_model_dir_ = (temp_dir / ("ksana_atb_kvcache_test_model_" + pid_suffix)).string();

    if (std::filesystem::exists(temp_model_dir_)) {
      std::filesystem::remove_all(temp_model_dir_);
    }
    std::filesystem::create_directory(temp_model_dir_);

    // 最小化 config.json：只保留解析必需字段（architectures / model_type / torch_dtype）
    // 以及单测断言用到的字段（num_attention_heads=64、kv_lora_rank=512）。use_mla 由
    // deepseek_v3 分支恒置为 true。
    {
      std::ofstream config_json(std::filesystem::path(temp_model_dir_) / "config.json");
      config_json << "{\n";
      config_json << "  \"architectures\": [\"DeepseekV3ForCausalLM\"],\n";
      config_json << "  \"model_type\": \"deepseek_v3\",\n";
      config_json << "  \"torch_dtype\": \"bfloat16\",\n";
      config_json << "  \"hidden_size\": 7168,\n";
      config_json << "  \"num_attention_heads\": 64,\n";  // -> model_config.head_num == 64
      config_json << "  \"num_hidden_layers\": 4,\n";     // 少量层即可，单测按 layer_num_on_node_ 动态取值
      config_json << "  \"kv_lora_rank\": 512,\n";        // -> mla_config.kv_lora_rank == 512
      config_json << "  \"max_position_embeddings\": 4096\n";
      config_json << "}\n";
    }

    // 最小化 yaml：源自 ksana_llm_kimi_k26_kunlun.yaml，但改为单卡、关键参数直接可见。
    {
      std::ofstream yaml(temp_yaml_path_);
      yaml << "setting:\n";
      yaml << "  global:\n";
      yaml << "    tensor_para_size: 1\n";
      yaml << "    pipeline_para_size: 1\n";
      yaml << "    attn_data_para_size: 1\n";
      yaml << "    expert_para_size: 1\n";
      yaml << "    enable_cuda_graph: true\n";
      yaml << "  batch_scheduler:\n";
      yaml << "    max_step_tokens: 4096\n";
      yaml << "    max_batch_size: 32\n";
      yaml << "    max_token_len: 4096\n";
      yaml << "  block_manager:\n";
      yaml << "    block_token_num: 64\n";
      yaml << "    reserved_device_memory_ratio: 0.05\n";
      yaml << "    block_host_memory_factor: 0.0\n";
      yaml << "  attn_backend:\n";
      yaml << "    enable_blocked_multi_token_forwarding_kv: false\n";
      yaml << "model_spec:\n";
      yaml << "  base_model:\n";
      yaml << "    model_dir: \"" << temp_model_dir_ << "\"\n";
    }

    // 2) 解析临时 yaml 并按单卡初始化。
    const auto& env = Singleton<Environment>::GetInstance();
    env->ParseConfig(temp_yaml_path_);

    auto context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(1, 1, 1, env));
    context->global->InitDistributedConfig();
    context->device = std::make_shared<DeviceContext>(1);

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

    // Override tensor_para_size to 1 for single-device test
    runtime_config.parallel_basic_config.attn_tensor_parallel_size = 1;
    runtime_config.parallel_basic_config.tensor_parallel_size = 1;
    env->schedule_config_parser_.runtime_config_ = runtime_config;

    model_input = std::make_unique<ModelInput>(model_config, runtime_config, rank, context, block_cache_layout);
  }

  void TearDown() override {
    try {
      if (!temp_yaml_path_.empty() && std::filesystem::exists(temp_yaml_path_)) {
        std::filesystem::remove(temp_yaml_path_);
      }
      if (!temp_model_dir_.empty() && std::filesystem::exists(temp_model_dir_)) {
        std::filesystem::remove_all(temp_model_dir_);
      }
    } catch (const std::exception&) {
      // best-effort cleanup
    }
  }

 protected:
  std::unique_ptr<ModelInput> model_input;
  std::string temp_yaml_path_;
  std::string temp_model_dir_;
};

// Test PrepareATBKVCache for MHA model (use_mla = false) in single-token (decode) mode
// Uses Kimi-K2.6 config but overrides use_mla to false to simulate MHA
TEST_F(ATBKVCacheTest, PrepareATBKVCacheMHADecodeTest) {
  model_input->model_config_.use_mla = false;

  const int layer_num = model_input->layer_num_on_node_;
  const int block_token_num = model_input->runtime_config_.attn_backend_config.block_token_num;
  const size_t device_block_num = 10;

  char fake_base[1024];
  void* base_ptr = static_cast<void*>(fake_base);
  auto mock_manager = std::make_shared<MockBlockAllocatorManagerForATB>(device_block_num, base_ptr);

  const int batch_size = 3;
  std::vector<ForwardRequest> forward_reqs(batch_size);
  std::vector<std::shared_ptr<std::vector<int>>> tokens_storage(batch_size);

  // seq_lens: request 0 has 5 tokens, request 1 has 10, request 2 has 3
  std::vector<int> seq_lens = {5, 10, 3};
  // For MHA: kv_multiplier=2, block IDs are multiples of layer_num*2
  std::vector<std::vector<int32_t>> block_ids = {
      {0, static_cast<int32_t>(layer_num * 2)},
      {0, static_cast<int32_t>(layer_num * 2), static_cast<int32_t>(layer_num * 2 * 2)},
      {static_cast<int32_t>(layer_num * 2)}};

  for (int i = 0; i < batch_size; ++i) {
    tokens_storage[i] = std::make_shared<std::vector<int>>(seq_lens[i], 0);
    forward_reqs[i].forwarding_tokens = tokens_storage[i];
    forward_reqs[i].kv_cached_token_num = seq_lens[i] - 1;
    forward_reqs[i].prefix_cache_len = 0;
    forward_reqs[i].attn_dp_group_id = 0;
    forward_reqs[i].block_allocator_manager =
        std::reinterpret_pointer_cast<BlockAllocatorManager>(std::shared_ptr<BlockAllocatorManagerInterface>(
            mock_manager, static_cast<BlockAllocatorManagerInterface*>(mock_manager.get())));
    forward_reqs[i].atb_kv_cache_base_blk_ids = {block_ids[i]};
    forward_reqs[i].kv_cache_ptrs.resize(block_ids[i].size(), nullptr);
  }

  model_input->PrepareATBKVCache(forward_reqs, /*is_multi_token_forward=*/false);

  // Verify seq_len_host
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(model_input->seq_len_host.GetPtr<int32_t>()[i], seq_lens[i]);
  }

  // Verify atb_attention_attr
  size_t slot_mapping_dim_1 = model_input->atb_attention_attr.GetPtr<uint64_t>()[0];
  size_t max_num_blocks_per_query = model_input->atb_attention_attr.GetPtr<uint64_t>()[1];
  EXPECT_EQ(slot_mapping_dim_1, static_cast<size_t>(batch_size));
  // In XPU graph mode (enable_cuda_graph=true) decode pins max_num_blocks_per_query to
  // graph_fixed_max_blocks_per_query_ (= ceil(max_seq_len / block_token_num)) so graph replay reads a fixed
  // stride; otherwise it is the block count of the longest request in the batch (max of {2, 3, 1} = 3).
  size_t expected_max_blocks =
      model_input->xpu_graph_mode_ ? model_input->graph_fixed_max_blocks_per_query_ : static_cast<size_t>(3);
  EXPECT_EQ(max_num_blocks_per_query, expected_max_blocks);

  // Verify layers_slot_mapping_host for single-token (decode) mode
  // For MHA: adjusted_blk = base_blk (no division)
  for (int layer_idx = 0; layer_idx < layer_num; ++layer_idx) {
    for (int req_idx = 0; req_idx < batch_size; ++req_idx) {
      int seq_len = seq_lens[req_idx];
      int last_token_block_idx = (seq_len - 1) / block_token_num;
      int32_t base_blk = block_ids[req_idx][last_token_block_idx];
      int32_t adjusted_blk = base_blk;  // MHA: no adjustment
      int32_t expected_slot = (adjusted_blk + layer_idx) * block_token_num + ((seq_len - 1) % block_token_num);
      EXPECT_EQ(model_input->layers_slot_mapping_host[layer_idx * slot_mapping_dim_1 + req_idx], expected_slot)
          << "Mismatch at layer=" << layer_idx << " req=" << req_idx;
    }
  }

  // Verify layers_block_table_host for single-token (decode) mode
  for (int layer_idx = 0; layer_idx < layer_num; ++layer_idx) {
    for (int req_idx = 0; req_idx < batch_size; ++req_idx) {
      for (size_t blk_idx = 0; blk_idx < block_ids[req_idx].size(); ++blk_idx) {
        int32_t adjusted_blk = block_ids[req_idx][blk_idx];  // MHA: no adjustment
        int32_t expected = adjusted_blk + layer_idx;
        size_t idx = layer_idx * batch_size * max_num_blocks_per_query + req_idx * max_num_blocks_per_query + blk_idx;
        EXPECT_EQ(model_input->layers_block_table_host[idx], expected)
            << "Mismatch at layer=" << layer_idx << " req=" << req_idx << " blk=" << blk_idx;
      }
      // Padding blocks should be -1
      for (size_t blk_idx = block_ids[req_idx].size(); blk_idx < max_num_blocks_per_query; ++blk_idx) {
        size_t idx = layer_idx * batch_size * max_num_blocks_per_query + req_idx * max_num_blocks_per_query + blk_idx;
        EXPECT_EQ(model_input->layers_block_table_host[idx], -1)
            << "Padding not -1 at layer=" << layer_idx << " req=" << req_idx << " blk=" << blk_idx;
      }
    }
  }
}

// Test PrepareATBKVCache for MLA model (use_mla = true, native Kimi-K2.6 config) in single-token (decode) mode
TEST_F(ATBKVCacheTest, PrepareATBKVCacheMLADecodeTest) {
  // Kimi-K2.6 is DeepSeek-V3 architecture with MLA, verify config loaded correctly
  ASSERT_TRUE(model_input->model_config_.use_mla);
  EXPECT_EQ(model_input->model_config_.head_num, 64);  // num_attention_heads from config.json
  EXPECT_EQ(model_input->model_config_.mla_config.kv_lora_rank, 512);

  const int layer_num = model_input->layer_num_on_node_;
  const int block_token_num = model_input->runtime_config_.attn_backend_config.block_token_num;
  const size_t device_block_num = 10;

  char fake_base[1024];
  void* base_ptr = static_cast<void*>(fake_base);
  auto mock_manager = std::make_shared<MockBlockAllocatorManagerForATB>(device_block_num, base_ptr);

  const int batch_size = 2;
  std::vector<ForwardRequest> forward_reqs(batch_size);
  std::vector<std::shared_ptr<std::vector<int>>> tokens_storage(batch_size);

  std::vector<int> seq_lens = {7, 4};
  // For MLA: BuildFlatKVCacheBlkIds bakes the ×1 K/V factor into the block id (original_id * layer_num),
  // so the consumer uses the id directly (adjusted_blk = base_blk, no /2).
  std::vector<std::vector<int32_t>> block_ids = {
      {static_cast<int32_t>(layer_num * 0), static_cast<int32_t>(layer_num * 1)},
      {static_cast<int32_t>(layer_num * 2)}};

  for (int i = 0; i < batch_size; ++i) {
    tokens_storage[i] = std::make_shared<std::vector<int>>(seq_lens[i], 0);
    forward_reqs[i].forwarding_tokens = tokens_storage[i];
    forward_reqs[i].kv_cached_token_num = seq_lens[i] - 1;
    forward_reqs[i].prefix_cache_len = 0;
    forward_reqs[i].attn_dp_group_id = 0;
    forward_reqs[i].block_allocator_manager =
        std::reinterpret_pointer_cast<BlockAllocatorManager>(std::shared_ptr<BlockAllocatorManagerInterface>(
            mock_manager, static_cast<BlockAllocatorManagerInterface*>(mock_manager.get())));
    forward_reqs[i].atb_kv_cache_base_blk_ids = {block_ids[i]};
    forward_reqs[i].kv_cache_ptrs.resize(block_ids[i].size(), nullptr);
  }

  model_input->PrepareATBKVCache(forward_reqs, /*is_multi_token_forward=*/false);

  // Verify atb_attention_attr
  size_t slot_mapping_dim_1 = model_input->atb_attention_attr.GetPtr<uint64_t>()[0];
  size_t max_num_blocks_per_query = model_input->atb_attention_attr.GetPtr<uint64_t>()[1];
  EXPECT_EQ(slot_mapping_dim_1, static_cast<size_t>(batch_size));
  // In XPU graph mode (enable_cuda_graph=true) decode pins max_num_blocks_per_query to
  // graph_fixed_max_blocks_per_query_ (= ceil(max_seq_len / block_token_num)) so graph replay reads a fixed
  // stride; otherwise it is the block count of the longest request in the batch (max of {2, 1} = 2).
  size_t expected_max_blocks =
      model_input->xpu_graph_mode_ ? model_input->graph_fixed_max_blocks_per_query_ : static_cast<size_t>(2);
  EXPECT_EQ(max_num_blocks_per_query, expected_max_blocks);

  // Verify layers_slot_mapping_host for MLA decode
  // For MLA: K/V factor already baked into the block id by the producer, so adjusted_blk = base_blk
  for (int layer_idx = 0; layer_idx < layer_num; ++layer_idx) {
    for (int req_idx = 0; req_idx < batch_size; ++req_idx) {
      int seq_len = seq_lens[req_idx];
      int last_token_block_idx = (seq_len - 1) / block_token_num;
      int32_t base_blk = block_ids[req_idx][last_token_block_idx];
      int32_t adjusted_blk = base_blk;  // MLA: factor baked in by producer, no adjustment
      int32_t expected_slot = (adjusted_blk + layer_idx) * block_token_num + ((seq_len - 1) % block_token_num);
      EXPECT_EQ(model_input->layers_slot_mapping_host[layer_idx * slot_mapping_dim_1 + req_idx], expected_slot)
          << "Mismatch at layer=" << layer_idx << " req=" << req_idx;
    }
  }

  // Verify layers_block_table_host for MLA decode
  for (int layer_idx = 0; layer_idx < layer_num; ++layer_idx) {
    for (int req_idx = 0; req_idx < batch_size; ++req_idx) {
      for (size_t blk_idx = 0; blk_idx < block_ids[req_idx].size(); ++blk_idx) {
        int32_t adjusted_blk = block_ids[req_idx][blk_idx];  // MLA: factor baked in by producer
        int32_t expected = adjusted_blk + layer_idx;
        size_t idx = layer_idx * batch_size * max_num_blocks_per_query + req_idx * max_num_blocks_per_query + blk_idx;
        EXPECT_EQ(model_input->layers_block_table_host[idx], expected)
            << "Mismatch at layer=" << layer_idx << " req=" << req_idx << " blk=" << blk_idx;
      }
    }
  }
}

// Test PrepareATBKVCache for MHA model in multi-token (prefill) mode
TEST_F(ATBKVCacheTest, PrepareATBKVCacheMHAPrefillTest) {
  model_input->model_config_.use_mla = false;

  const int layer_num = model_input->layer_num_on_node_;
  const int block_token_num = model_input->runtime_config_.attn_backend_config.block_token_num;
  const size_t device_block_num = 10;

  char fake_base[1024];
  void* base_ptr = static_cast<void*>(fake_base);
  auto mock_manager = std::make_shared<MockBlockAllocatorManagerForATB>(device_block_num, base_ptr);

  const int batch_size = 2;
  std::vector<ForwardRequest> forward_reqs(batch_size);
  std::vector<std::shared_ptr<std::vector<int>>> tokens_storage(batch_size);

  // Request 0: 6 total tokens, 2 already cached → 4 new tokens
  // Request 1: 4 total tokens, 0 cached → 4 new tokens
  std::vector<int> total_tokens = {6, 4};
  std::vector<int> kv_cached = {2, 0};
  std::vector<int> prefix_cache = {2, 0};

  std::vector<std::vector<int32_t>> block_ids = {
      {0, static_cast<int32_t>(layer_num * 2), static_cast<int32_t>(layer_num * 2 * 2)},
      {0, static_cast<int32_t>(layer_num * 2)}};

  for (int i = 0; i < batch_size; ++i) {
    tokens_storage[i] = std::make_shared<std::vector<int>>(total_tokens[i], 0);
    forward_reqs[i].forwarding_tokens = tokens_storage[i];
    forward_reqs[i].kv_cached_token_num = kv_cached[i];
    forward_reqs[i].prefix_cache_len = prefix_cache[i];
    forward_reqs[i].attn_dp_group_id = 0;
    forward_reqs[i].block_allocator_manager =
        std::reinterpret_pointer_cast<BlockAllocatorManager>(std::shared_ptr<BlockAllocatorManagerInterface>(
            mock_manager, static_cast<BlockAllocatorManagerInterface*>(mock_manager.get())));
    forward_reqs[i].atb_kv_cache_base_blk_ids = {block_ids[i]};
    forward_reqs[i].kv_cache_ptrs.resize(block_ids[i].size(), nullptr);
  }

  model_input->PrepareATBKVCache(forward_reqs, /*is_multi_token_forward=*/true);

  // In multi-token mode, slot_mapping_dim_1 = sum of (total_tokens - skip_tokens) for all reqs
  size_t slot_mapping_dim_1 = model_input->atb_attention_attr.GetPtr<uint64_t>()[0];
  size_t expected_dim_1 = 0;
  for (int i = 0; i < batch_size; ++i) {
    size_t skip = std::max(kv_cached[i], prefix_cache[i]);
    expected_dim_1 += total_tokens[i] - skip;
  }
  EXPECT_EQ(slot_mapping_dim_1, expected_dim_1);

  // Verify layers_slot_mapping_host for multi-token (prefill) mode
  size_t offset = 0;
  for (int req_idx = 0; req_idx < batch_size; ++req_idx) {
    size_t skip = std::max(kv_cached[req_idx], prefix_cache[req_idx]);
    size_t new_tokens = total_tokens[req_idx] - skip;
    for (int layer_idx = 0; layer_idx < layer_num; ++layer_idx) {
      for (size_t t = 0; t < new_tokens; ++t) {
        size_t token_idx = skip + t;
        int32_t inner_block_offset = token_idx % block_token_num;
        int32_t base_blk = block_ids[req_idx][token_idx / block_token_num];
        int32_t adjusted_blk = base_blk;  // MHA: no adjustment
        int32_t expected_slot = (adjusted_blk + layer_idx) * block_token_num + inner_block_offset;
        EXPECT_EQ(model_input->layers_slot_mapping_host[layer_idx * slot_mapping_dim_1 + offset + t], expected_slot)
            << "Mismatch at layer=" << layer_idx << " req=" << req_idx << " token=" << t;
      }
    }
    offset += new_tokens;
  }

  // Verify layers_block_table_host is empty for multi-token mode
  EXPECT_TRUE(model_input->layers_block_table_host.empty());
}

// Test PrepareATBKVCache for MLA model (native Kimi-K2.6) in multi-token (prefill) mode
TEST_F(ATBKVCacheTest, PrepareATBKVCacheMLAPrefillTest) {
  ASSERT_TRUE(model_input->model_config_.use_mla);

  const int layer_num = model_input->layer_num_on_node_;
  const int block_token_num = model_input->runtime_config_.attn_backend_config.block_token_num;
  const size_t device_block_num = 10;

  char fake_base[1024];
  void* base_ptr = static_cast<void*>(fake_base);
  auto mock_manager = std::make_shared<MockBlockAllocatorManagerForATB>(device_block_num, base_ptr);

  const int batch_size = 2;
  std::vector<ForwardRequest> forward_reqs(batch_size);
  std::vector<std::shared_ptr<std::vector<int>>> tokens_storage(batch_size);

  std::vector<int> total_tokens = {5, 3};
  std::vector<int> kv_cached = {1, 0};
  std::vector<int> prefix_cache = {1, 0};

  // For MLA: BuildFlatKVCacheBlkIds bakes the ×1 K/V factor into the block id (original_id * layer_num),
  // so the consumer uses the id directly (adjusted_blk = base_blk, no /2).
  std::vector<std::vector<int32_t>> block_ids = {
      {static_cast<int32_t>(layer_num * 0), static_cast<int32_t>(layer_num * 1), static_cast<int32_t>(layer_num * 2)},
      {static_cast<int32_t>(layer_num * 0), static_cast<int32_t>(layer_num * 1)}};

  for (int i = 0; i < batch_size; ++i) {
    tokens_storage[i] = std::make_shared<std::vector<int>>(total_tokens[i], 0);
    forward_reqs[i].forwarding_tokens = tokens_storage[i];
    forward_reqs[i].kv_cached_token_num = kv_cached[i];
    forward_reqs[i].prefix_cache_len = prefix_cache[i];
    forward_reqs[i].attn_dp_group_id = 0;
    forward_reqs[i].block_allocator_manager =
        std::reinterpret_pointer_cast<BlockAllocatorManager>(std::shared_ptr<BlockAllocatorManagerInterface>(
            mock_manager, static_cast<BlockAllocatorManagerInterface*>(mock_manager.get())));
    forward_reqs[i].atb_kv_cache_base_blk_ids = {block_ids[i]};
    forward_reqs[i].kv_cache_ptrs.resize(block_ids[i].size(), nullptr);
  }

  model_input->PrepareATBKVCache(forward_reqs, /*is_multi_token_forward=*/true);

  size_t slot_mapping_dim_1 = model_input->atb_attention_attr.GetPtr<uint64_t>()[0];
  size_t expected_dim_1 = 0;
  for (int i = 0; i < batch_size; ++i) {
    size_t skip = std::max(kv_cached[i], prefix_cache[i]);
    expected_dim_1 += total_tokens[i] - skip;
  }
  EXPECT_EQ(slot_mapping_dim_1, expected_dim_1);

  // Verify layers_slot_mapping_host for MLA prefill
  size_t offset = 0;
  for (int req_idx = 0; req_idx < batch_size; ++req_idx) {
    size_t skip = std::max(kv_cached[req_idx], prefix_cache[req_idx]);
    size_t new_tokens = total_tokens[req_idx] - skip;
    for (int layer_idx = 0; layer_idx < layer_num; ++layer_idx) {
      for (size_t t = 0; t < new_tokens; ++t) {
        size_t token_idx = skip + t;
        int32_t inner_block_offset = token_idx % block_token_num;
        int32_t base_blk = block_ids[req_idx][token_idx / block_token_num];
        int32_t adjusted_blk = base_blk;  // MLA: factor baked in by producer, no adjustment
        int32_t expected_slot = (adjusted_blk + layer_idx) * block_token_num + inner_block_offset;
        EXPECT_EQ(model_input->layers_slot_mapping_host[layer_idx * slot_mapping_dim_1 + offset + t], expected_slot)
            << "Mismatch at layer=" << layer_idx << " req=" << req_idx << " token=" << t;
      }
    }
    offset += new_tokens;
  }
}

// Test PrepareATBKVCache k_cache_blocks_base shape update for both MHA and MLA
TEST_F(ATBKVCacheTest, PrepareATBKVCacheBlocksBaseShapeTest) {
  const int layer_num = model_input->layer_num_on_node_;
  const int block_token_num = model_input->runtime_config_.attn_backend_config.block_token_num;
  const size_t device_block_num = 8;

  char fake_base[1024];
  void* base_ptr = static_cast<void*>(fake_base);
  auto mock_manager = std::make_shared<MockBlockAllocatorManagerForATB>(device_block_num, base_ptr);

  ForwardRequest forward_req;
  std::vector<int32_t> blk_ids = {0};
  auto tokens = std::make_shared<std::vector<int>>(std::vector<int>{1, 2, 3});
  forward_req.forwarding_tokens = tokens;
  forward_req.kv_cached_token_num = 2;
  forward_req.prefix_cache_len = 0;
  forward_req.attn_dp_group_id = 0;
  forward_req.block_allocator_manager =
      std::reinterpret_pointer_cast<BlockAllocatorManager>(std::shared_ptr<BlockAllocatorManagerInterface>(
          mock_manager, static_cast<BlockAllocatorManagerInterface*>(mock_manager.get())));
  forward_req.atb_kv_cache_base_blk_ids = {blk_ids};
  forward_req.kv_cache_ptrs.resize(1, nullptr);

  std::vector<ForwardRequest> forward_reqs = {forward_req};

  // Test MHA: kv_multiplier = 2
  model_input->model_config_.use_mla = false;
  model_input->k_cache_blocks_base.shape[0] = 0;  // force re-computation
  model_input->PrepareATBKVCache(forward_reqs, /*is_multi_token_forward=*/false);

  size_t expected_total_blocks_mha = device_block_num * 2 * layer_num;
  EXPECT_EQ(model_input->k_cache_blocks_base.shape[0], expected_total_blocks_mha);
  EXPECT_EQ(model_input->k_cache_blocks_base.shape[1], static_cast<size_t>(block_token_num));
  EXPECT_EQ(model_input->k_cache_blocks_base.shape[2], static_cast<size_t>(model_input->model_config_.head_num));
  EXPECT_EQ(model_input->k_cache_blocks_base.shape[3], static_cast<size_t>(model_input->model_config_.size_per_head));
  EXPECT_EQ(model_input->v_cache_blocks_base.shape[0], expected_total_blocks_mha);

  // Test MLA: kv_multiplier = 1
  model_input->model_config_.use_mla = true;
  model_input->k_cache_blocks_base.shape[0] = 0;  // force re-computation
  model_input->PrepareATBKVCache(forward_reqs, /*is_multi_token_forward=*/false);

  size_t expected_total_blocks_mla = device_block_num * 1 * layer_num;
  EXPECT_EQ(model_input->k_cache_blocks_base.shape[0], expected_total_blocks_mla);
  EXPECT_EQ(model_input->v_cache_blocks_base.shape[0], expected_total_blocks_mla);
}

#endif  // defined(ENABLE_ACL) || defined(ENABLE_KUNLUN)

}  // namespace ksana_llm
