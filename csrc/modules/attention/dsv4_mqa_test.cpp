/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cmath>
#include <cstdlib>
#include <filesystem>

#include "cache_manager/prefix_cache_manager.h"
#include "executor/inflight_resource.h"
#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"
#include "models/common/common_model.h"
#include "models/common/common_weight.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta_creator.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_compressor_slot_resource.h"
#include "modules/attention/dsv4_mqa.h"
#include "tests/test.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/memory_allocator.h"
#include "utils/singleton.h"

namespace ksana_llm {

class Dsv4MqaTestModel : public CommonModel {
 public:
  using CommonModel::cast_layer_;
  using CommonModel::context_;
  using CommonModel::forwarding_context_;
  using CommonModel::model_config_;

  std::unique_ptr<Dsv4Mqa> mqa_;
  Dsv4AttnBuffer dsv4_attn_buffer_;

  Dsv4MqaTestModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                   std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
                   std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
      : CommonModel(model_config, runtime_config, rank, context) {
    ModelRunConfig model_run_config;
    model_run_config.position_encoding = PositionEncoding::ROPE;
    CommonModel::InitRunConfig(model_run_config, base_weight, block_cache_layout);
  }

  ~Dsv4MqaTestModel() = default;

  std::unique_ptr<AttentionMetaCreator> CreateCustomAttentionMetaCreator(
      const AttentionMetaCreator::Config& config) override {
    return std::make_unique<DeepSeekV4AttentionMetaCreator>(config);
  }

  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override {
    Dsv4Mqa::CreateBuffers(CommonModel::GetBufferManager(), model_creation_config.attn_config,
                           creation_context.runtime_config, dsv4_attn_buffer_);
    mqa_ = std::make_unique<Dsv4Mqa>(/*layer_idx*/ 0, /*is_neox=*/false, /*add_qkv_bias=*/false,
                                     /*use_qk_norm=*/false, creation_context, model_creation_config, dsv4_attn_buffer_);
    return Status();
  }

  Status LayerForward(ForwardingContext& forwarding_context_, const RunMode run_mode = RunMode::kMain) override {
    return Status();
  }

  Status CommonAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                         const bool is_multi_token_forward, const std::vector<ForwardRequest>& forward_reqs) {
    forwarding_context_->GetModelInput()->ParseFromRequests(forward_reqs);

    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context_->buffers_->hidden_buffer_0);

    // Prepare input
    const size_t hidden_units = model_config_.hidden_units;
    hidden_buffer_tensors_0[0].shape = {forwarding_context_->GetModelInput()->input_ids.shape[0], hidden_units};
    hidden_buffer_tensors_0[0].dtype = model_config_.weight_data_type;

    std::vector<bfloat16> input_bf16;
    input_bf16.reserve(hidden_buffer_tensors_0[0].GetElementNumber());
    size_t count = 0;
    for (const auto& req : forward_reqs) {
      count += req.prefix_cache_len * hidden_units;
      for (size_t i = 0; i < req.GetInputIdsLength(); i++) {
        for (size_t j = 0; j < hidden_units; j++) {
          input_bf16.push_back(
              static_cast<bfloat16>(1.0f / (count % 97 * 0.1f + 1.0f) * ((count % 7) % 2 ? -1.0f : 1.0f)));
          ++count;
        }
      }
    }
    Memcpy(hidden_buffer_tensors_0[0].GetPtr<void>(), input_bf16.data(), hidden_buffer_tensors_0[0].GetTotalBytes(),
           MEMCPY_HOST_TO_DEVICE);

    // Forward
    CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context_->buffers_->shared_buffer);
    Status status = mqa_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, *forwarding_context_);
    StreamSynchronize(context_->device->GetPrimaryComputeStream());

    // Output shape sanity: [total_tokens, hidden_units]
    EXPECT_EQ(hidden_buffer_tensors_0[0].shape.size(), 2u);
    EXPECT_EQ(hidden_buffer_tensors_0[0].shape[0], forwarding_context_->GetModelInput()->input_ids.shape[0]);
    EXPECT_EQ(hidden_buffer_tensors_0[0].shape[1], hidden_units);

    // Verify output against reference values
    constexpr size_t kVerifyNum = 6;
    size_t num = std::min(kVerifyNum, hidden_buffer_tensors_0[0].GetElementNumber());
    std::vector<bfloat16> output_bf16(num);
    Memcpy(output_bf16.data(), hidden_buffer_tensors_0[0].GetPtr<void>(), num * sizeof(bfloat16),
           MEMCPY_DEVICE_TO_HOST);

    std::vector<float> expected;
    if (is_multi_token_forward) {
      if (forwarding_context_->GetModelInput()->dp_total_prefix_len == 0) {
        if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_E4M3) {
          expected = {1144, -676, -1176, -1448, 438, -25.5};
        } else {  // TYPE_FP8_DS_MLA
          expected = {1144, -676, -1168, -1440, 442, -27.375};
        }
      } else {
        if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_E4M3) {
          expected = {-47.75, -1160, -1368, -2544, -660, -1552};
        } else {  // TYPE_FP8_DS_MLA
          expected = {-38.5, -1144, -1384, -2544, -664, -1560};
        }
      }
    } else {
      if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_E4M3) {
        expected = {704, -1224, -976, -3024, 286, -1256};
      } else {  // TYPE_FP8_DS_MLA
        expected = {708, -1232, -964, -3024, 292, -1264};
      }
    }

    for (size_t i = 0; i < expected.size(); i++) {
      const float actual = static_cast<float>(output_bf16[i]);
      const float diff = std::fabs((actual - expected[i]) / expected[i]);
      EXPECT_LT(diff, 0.05);
    }

    return status;
  }
};

class Dsv4MqaWeight : public CommonWeight<bfloat16> {
  using CommonWeight<bfloat16>::weights_map_;
  using CommonWeight<bfloat16>::context_;

 public:
  Dsv4MqaWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                std::shared_ptr<Context> context)
      : CommonWeight<bfloat16>(model_config, runtime_config, rank, context) {}
  ~Dsv4MqaWeight() override = default;

  void AddDsv4MqaWeight(int device_id = 0) {
    const auto& dsv4_attn_config = model_config_.dsv4_attn_config;
    const size_t hidden_units = model_config_.hidden_units;
    const size_t q_lora_rank = dsv4_attn_config.q_lora_rank;
    const size_t o_lora_rank = dsv4_attn_config.o_lora_rank;
    const size_t head_dim = model_config_.size_per_head;
    const size_t o_groups = dsv4_attn_config.o_groups;
    const size_t n_local_heads = model_config_.head_num;
    const size_t o_local_groups = o_groups;
    const size_t d_per_group = n_local_heads * head_dim / o_local_groups;

    std::unordered_map<std::string, std::vector<size_t>> add_tensor_map;
    add_tensor_map["model.layers.0.self_attn.wqkv_a.weight"] = {hidden_units, q_lora_rank + head_dim};
    add_tensor_map["model.layers.0.self_attn.q_norm.weight"] = {q_lora_rank};
    add_tensor_map["model.layers.0.self_attn.kv_norm.weight"] = {head_dim};
    add_tensor_map["model.layers.0.self_attn.wq_b.weight"] = {q_lora_rank, n_local_heads * head_dim};
    add_tensor_map["model.layers.0.self_attn.wo_a.weight"] = {o_local_groups, o_lora_rank, d_per_group};
    add_tensor_map["model.layers.0.self_attn.wo_b.weight"] = {o_local_groups * o_lora_rank, hidden_units};
    add_tensor_map["model.layers.0.self_attn.attn_sink"] = {static_cast<size_t>(n_local_heads), head_dim};
    for (auto& [tensor_name, shape] : add_tensor_map) {
      Tensor tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_BF16, shape);
      weights_map_.emplace(tensor_name, tensor);
      std::vector<bfloat16> input_data(tensor.GetElementNumber());
      for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<bfloat16>(1.0f / (i % 97 * 0.1f + 1.0f) * ((i % 7) % 2 ? -1.0f : 1.0f));
      }
      Memcpy(tensor.GetPtr<void>(), input_data.data(), tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE);
    }
  }
};

class Dsv4MqaTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    test_name = test_info->name();

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/ksana_llm_deepseekv2.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    AttentionBackendManager::GetInstance()->Initialize();
    const auto& env = Singleton<Environment>::GetInstance();
    context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(1, 1, 1, env));
    context->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context->device = std::make_shared<DeviceContext>(1);
    context->device->ext->InitNcclParam(nccl_uid);

    env->ParseConfig(config_path, std::filesystem::absolute(config_path_relate.parent_path() / "deepseekv2").string());
    env->GetModelConfig(model_config);
    InitPipelineConfigForTest(env, model_config);

    // Override config for dsv4 MQA
    model_config.use_dsv4_attn = true;
    model_config.use_dsa = true;
    model_config.dsv4_attn_config.o_groups = 8;
    model_config.dsv4_attn_config.o_lora_rank = 1024;
    model_config.dsv4_attn_config.q_lora_rank = 1024;
    model_config.dsv4_attn_config.qk_rope_head_dim = 64;
    model_config.dsv4_attn_config.sliding_window = 128;
    model_config.dsv4_attn_config.compress_ratios.assign(model_config.num_layer, 0);
    model_config.size_per_head = 512;
    model_config.num_key_value_heads = 1;
    model_config.rope_theta = 10000.0f;
    model_config.rotary_embedding = model_config.dsv4_attn_config.qk_rope_head_dim;

    // Set kv_cache_dtype based on test name
    if (test_name.find("FP8DsMla") != std::string::npos) {
      env->schedule_config_parser_.runtime_config_.attn_backend_config.kv_cache_dtype = TYPE_FP8_DS_MLA;
    } else if (test_name.find("FP8E4M3") != std::string::npos) {
      env->schedule_config_parser_.runtime_config_.attn_backend_config.kv_cache_dtype = TYPE_FP8_E4M3;
    }

    // block_token_num must satisfy Dsv4MqaMeta (multiple of 128) and FlashDsv4 (page_bits in {1,6,8})
    env->schedule_config_parser_.runtime_config_.attn_backend_config.block_token_num = 256;

    env->SetModelConfig(model_config);
    block_cache_layout_ = GetTestModelCacheLayout(env);
    env->InitializeBlockManagerConfig(block_cache_layout_);
    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.block_host_memory_factor = 0;
    block_manager_config.reserved_device_memory_ratio = 0.98;
    block_manager_config.host_allocator_config.blocks_num = 0;
    block_manager_config.device_allocator_config.blocks_num = 10;
    env->SetBlockManagerConfig(block_manager_config);

    env->GetRuntimeConfig(runtime_config);
    runtime_config.max_batch_size = 5;
    runtime_config.max_seq_len = 20;
    runtime_config.max_step_token_num = 40;
    runtime_config.enable_prefix_caching = true;
    env->schedule_config_parser_.runtime_config_ = runtime_config;

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

    // Allocate dedicated SWA pool memory
    swa_block_size_ = block_cache_layout_->GetBlockCacheLayout("SWA-cache")->GetBlockCacheSize();
    swa_pool_block_num_ = 2;
    CUDA_CHECK(cudaMalloc(&swa_pool_base_, swa_block_size_ * swa_pool_block_num_));

    CacheManagerConfig cache_manager_config;
    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = 1;
    cache_manager_config.enable_prefix_caching = true;
    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, engine_block_group);
  }

  void TearDown() override {
    if (swa_pool_base_) {
      CUDA_CHECK(cudaFree(swa_pool_base_));
      swa_pool_base_ = nullptr;
    }
  }

 protected:
  std::string test_name;
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  PipelineConfig pipeline_config;

  std::shared_ptr<BlockAllocatorManager> engine_block_manager;
  std::shared_ptr<BlockAllocatorManager> executor_block_manager;
  std::shared_ptr<BlockAllocatorGroupInterface> engine_block_group;
  std::shared_ptr<BlockAllocatorGroupInterface> executor_block_group;
  std::shared_ptr<Context> context;
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_;
  std::shared_ptr<PrefixCacheManager> cache_manager;

  // SWA pool
  void* swa_pool_base_ = nullptr;
  size_t swa_block_size_ = 0;
  int64_t swa_pool_block_num_ = 0;

  void SetupBf16Config() {
    model_config.is_quant = false;
    model_config.weight_data_type = TYPE_BF16;
    model_config.quant_config.method = QUANT_NONE;
    runtime_config.inter_data_type = model_config.weight_data_type;
    const auto& env = Singleton<Environment>::GetInstance();
    env->schedule_config_parser_.runtime_config_.inter_data_type = model_config.weight_data_type;
  }

  // Build a forward request with a single block allocated and SWA pool wired up.
  ForwardRequest MakeRequest(std::vector<int>& input_ids_storage, std::vector<FlexibleCachedCopyTask>& flex_tasks,
                             EmbeddingSlice& embedding_slice) {
    ForwardRequest forward;
    forward.block_allocator_manager = executor_block_manager;
    forward.forwarding_tokens = std::make_shared<std::vector<int>>(input_ids_storage);
    forward.flexible_cached_copy_tasks = &flex_tasks;
    embedding_slice.pos = {};
    embedding_slice.embeddings = {};
    forward.input_refit_embedding = &embedding_slice;

    std::vector<int> block_ids;
    engine_block_group->GetDeviceBlockAllocator(0)->AllocateBlocks(1, block_ids);
    executor_block_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, forward.kv_cache_ptrs);
    forward.atb_kv_cache_base_blk_ids.assign(1, {});
    BuildFlatKVCacheBlockIds(GetTotalLayerNum(model_config, pipeline_config), {block_ids},
                             forward.atb_kv_cache_base_blk_ids, executor_block_group->GetDeviceBlockAllocator(0));

    forward.swa_block_base_addr = swa_pool_base_;
    forward.swa_pool_block_num = swa_pool_block_num_;
    forward.swa_kv_cache_blocks = {0};
    forward.swa_kv_cache_block_idx_offsets = {0};
    return forward;
  }

  void TestMqaForward(int device_id = 0) {
    SetDevice(device_id);
    auto base_weight = std::make_shared<Dsv4MqaWeight>(model_config, runtime_config, 0, context);
    base_weight->AddDsv4MqaWeight(device_id);

    std::shared_ptr<ksana_llm::BaseWeight> bs1 = base_weight;
    auto test_model =
        std::make_shared<Dsv4MqaTestModel>(model_config, runtime_config, 0, context, bs1, block_cache_layout_);

    // Wire InflightResourceManager so PrepareCompressor can look up compressor slots.
    auto inflight_mgr = std::make_shared<InflightResourceManager>();
    auto slot_res = std::make_shared<CompressorSlotResource>();
    slot_res->Init(runtime_config);
    inflight_mgr->RegisterResource("compressor_slot", slot_res);
    test_model->SetInflightResourceManager(inflight_mgr.get());

    std::vector<int> input_ids = {233, 1681};
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
    EmbeddingSlice embedding_slice;
    ForwardRequest forward = MakeRequest(input_ids, flexible_cached_copy_tasks, embedding_slice);

    // Allocate a compressor slot for this request before any ParseFromRequests call.
    slot_res->Alloc(forward.req_id);

    // Prefill (context phase): two new tokens, no prefix cache.
    test_model->CommonAttention(0, bs1, true, {forward});

    // Prefill with prefix caching: append a token, mark first token as already cached.
    forward.forwarding_tokens->push_back(321);
    forward.prefix_cache_len = 1;
    forward.kv_cached_token_num = 1;
    test_model->CommonAttention(0, bs1, true, {forward});
    forward.forwarding_tokens->pop_back();

    // Decode phase: append a token, mark prior two as cached so q_seq_len==1.
    forward.forwarding_tokens->push_back(321);
    forward.prefix_cache_len = 0;
    forward.kv_cached_token_num = 2;
    test_model->CommonAttention(0, bs1, false, {forward});
  }
};

TEST_F(Dsv4MqaTest, ForwardWithDsv4MqaKvFP8DsMlaTest) {
  SetupBf16Config();
  TestMqaForward();
}

TEST_F(Dsv4MqaTest, ForwardWithDsv4MqaKvFP8E4M3Test) {
  SetupBf16Config();
  TestMqaForward();
}

}  // namespace ksana_llm
