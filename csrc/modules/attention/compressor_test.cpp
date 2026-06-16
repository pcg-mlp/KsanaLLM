/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cmath>
#include <cstdlib>
#include <filesystem>

#include "cache_manager/prefix_cache_manager.h"
#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"
#include "models/common/common_model.h"
#include "models/common/common_weight.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta_creator.h"
#include "modules/attention/compressor.h"
#include "tests/test.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/memory_allocator.h"
#include "utils/singleton.h"

namespace ksana_llm {

class CompressorTestModel : public CommonModel {
 public:
  using CommonModel::context_;
  using CommonModel::forwarding_context_;
  using CommonModel::model_config_;

  std::unique_ptr<Compressor> compressor_;
  // Workspace for compressor intermediates (kv_score + optional FP32 cast buffer).
  Tensor workspace_tensor_;
  // Output buffer [max_token_num, head_dim].
  Tensor kv_compressed_output_tensor_;

  CompressorTestModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                      std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
                      std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
      : CommonModel(model_config, runtime_config, rank, context) {
    ModelRunConfig model_run_config;
    model_run_config.position_encoding = PositionEncoding::ROPE;
    CommonModel::InitRunConfig(model_run_config, base_weight, block_cache_layout);
  }

  ~CompressorTestModel() = default;

  std::unique_ptr<AttentionMetaCreator> CreateCustomAttentionMetaCreator(
      const AttentionMetaCreator::Config& config) override {
    return std::make_unique<DeepSeekV4AttentionMetaCreator>(config);
  }

  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override {
    model_creation_config.attn_config.idx = 0;
    compressor_ =
        std::make_unique<Compressor>(/*layer_idx=*/0, /*is_in_indexer=*/false, creation_context, model_creation_config);

    // Allocate workspace & output buffers up-front, sized for max_step_token_num.
    const auto& attn_config = model_creation_config.attn_config;
    const size_t head_dim = attn_config.size_per_head;
    const uint32_t compress_ratio = model_config_.dsv4_attn_config.compress_ratios[0];
    const uint32_t coff = 1 + (compress_ratio == 4);
    const size_t max_tokens = creation_context.runtime_config.max_step_token_num;

    // workspace holds kv_score [T, 2*coff*head_dim] BF16, plus FP32 cast buffer when state_dtype==FP32.
    const bool use_fp32 = creation_context.runtime_config.compressor_config.state_dtype == TYPE_FP32;
    const size_t bf16_per_token = 2 * coff * head_dim;
    const size_t workspace_elems = max_tokens * (bf16_per_token + (use_fp32 ? 2 * bf16_per_token : 0));
    workspace_tensor_ = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {workspace_elems});

    kv_compressed_output_tensor_ = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {max_tokens * head_dim});
    return Status();
  }

  Status LayerForward(ForwardingContext& forwarding_context_, const RunMode run_mode = RunMode::kMain) override {
    return Status();
  }

  // Drive Compressor.Forward end-to-end on a batch of forward requests.
  Status RunCompressor(const std::vector<ForwardRequest>& forward_reqs) {
    forwarding_context_->GetModelInput()->ParseFromRequests(forward_reqs);

    // Build hidden states input [T, hidden_units] in BF16 with deterministic pseudo values.
    const size_t hidden_units = model_config_.hidden_units;
    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context_->buffers_->hidden_buffer_0);
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
    Status status = compressor_->Forward(hidden_buffer_tensors_0[0], workspace_tensor_, kv_compressed_output_tensor_,
                                         *forwarding_context_);
    StreamSynchronize(context_->device->GetPrimaryComputeStream());

    // Output shape sanity: [total_tokens, head_dim]
    EXPECT_EQ(kv_compressed_output_tensor_.shape.size(), 2u);
    EXPECT_EQ(kv_compressed_output_tensor_.shape[0], forwarding_context_->GetModelInput()->input_ids.shape[0]);
    EXPECT_EQ(kv_compressed_output_tensor_.shape[1], model_config_.size_per_head);

    // Verify a few output values are finite (not NaN / Inf).
    constexpr size_t kVerifyNum = 6;
    const size_t num = std::min(kVerifyNum, kv_compressed_output_tensor_.GetElementNumber());
    std::vector<bfloat16> output_bf16(num);
    Memcpy(output_bf16.data(), kv_compressed_output_tensor_.GetPtr<void>(), num * sizeof(bfloat16),
           MEMCPY_DEVICE_TO_HOST);
    for (size_t i = 0; i < num; i++) {
      const float v = static_cast<float>(output_bf16[i]);
      EXPECT_NEAR(v, 0, 1e-3);  // No tokens require compression
    }

    Memcpy(output_bf16.data(), workspace_tensor_.GetPtr<void>(), num * sizeof(bfloat16), MEMCPY_DEVICE_TO_HOST);
    std::vector<float> expected;
    if (model_config_.dsv4_attn_config.compress_ratios[0] == 4) {
      expected = {-20.375, -20.5, -20, -19.875, 57.5, -19.25};
    } else {
      expected = {-19.375, -21.375, 60, -19, -18.75, -18.75};
    }
    for (size_t i = 0; i < expected.size(); i++) {
      const float actual = static_cast<float>(output_bf16[i]);
      const float diff = std::fabs((actual - expected[i]) / expected[i]);
      EXPECT_LT(diff, 0.05);
    }
    return status;
  }
};

class CompressorWeight : public CommonWeight<bfloat16> {
  using CommonWeight<bfloat16>::weights_map_;
  using CommonWeight<bfloat16>::context_;

 public:
  CompressorWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                   std::shared_ptr<Context> context)
      : CommonWeight<bfloat16>(model_config, runtime_config, rank, context) {}
  ~CompressorWeight() override = default;

  void AddCompressorWeights() {
    const size_t hidden_units = model_config_.hidden_units;
    const size_t head_dim = model_config_.size_per_head;
    const uint32_t ratio = model_config_.dsv4_attn_config.compress_ratios[0];
    const uint32_t coff = 1 + (ratio == 4);

    // Compressor weights
    AddBf16Tensor("model.layers.0.self_attn.compressor.wkv_gate.weight", {hidden_units, 2 * coff * head_dim});
    AddBf16Tensor("model.layers.0.self_attn.compressor.norm.weight", {head_dim});
    AddFp32Tensor("model.layers.0.self_attn.compressor.ape", {ratio, coff * head_dim});
  }

 private:
  void AddBf16Tensor(const std::string& name, const std::vector<size_t>& shape) {
    Tensor tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, shape);
    std::vector<bfloat16> data(tensor.GetElementNumber());
    for (size_t i = 0; i < data.size(); i++) {
      data[i] = static_cast<bfloat16>(1.0f / (i % 97 * 0.1f + 1.0f) * ((i % 7) % 2 ? -1.0f : 1.0f));
    }
    Memcpy(tensor.GetPtr<void>(), data.data(), tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE);
    weights_map_.emplace(name, tensor);
  }

  void AddFp32Tensor(const std::string& name, const std::vector<size_t>& shape) {
    Tensor tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, shape);
    std::vector<float> data(tensor.GetElementNumber());
    for (size_t i = 0; i < data.size(); i++) {
      data[i] = 1.0f / (i % 97 * 0.1f + 1.0f) * ((i % 7) % 2 ? -1.0f : 1.0f);
    }
    Memcpy(tensor.GetPtr<void>(), data.data(), tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE);
    weights_map_.emplace(name, tensor);
  }
};

class CompressorTest : public testing::Test {
 protected:
  // Initialise environment / block manager / cache for a specific compress_ratio (4 or 128).
  // Replaces the usual SetUp() since cache layout depends on compress_ratios.
  void InitWithRatio(uint32_t compress_ratio) {
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

    // Configure DSV4 compressor with the requested ratio (head_dim can be 512 for both C4 and C128);
    model_config.use_dsv4_attn = true;
    model_config.use_dsa = true;
    model_config.dsv4_attn_config.sliding_window = 128;
    model_config.dsv4_attn_config.compress_ratios.assign(model_config.num_layer, 0);
    model_config.dsv4_attn_config.compress_ratios[0] = compress_ratio;
    model_config.size_per_head = 512;
    model_config.num_key_value_heads = 1;
    model_config.dsv4_attn_config.compress_rope_theta = 160000.0f;
    model_config.dsv4_attn_config.qk_rope_head_dim = 64;
    model_config.rotary_embedding = model_config.dsv4_attn_config.qk_rope_head_dim;

    env->schedule_config_parser_.runtime_config_.attn_backend_config.kv_cache_dtype = TYPE_FP8_DS_MLA;
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

  void TestCompressorForward(int device_id = 0) {
    SetDevice(device_id);
    auto base_weight = std::make_shared<CompressorWeight>(model_config, runtime_config, 0, context);
    base_weight->AddCompressorWeights();

    std::shared_ptr<BaseWeight> bs1 = base_weight;
    auto test_model =
        std::make_shared<CompressorTestModel>(model_config, runtime_config, 0, context, bs1, block_cache_layout_);

    std::vector<int> input_ids = {233, 1681};
    std::vector<FlexibleCachedCopyTask> flex_tasks;
    EmbeddingSlice slice;
    ForwardRequest forward = MakeRequest(input_ids, flex_tasks, slice);

    // Prefill (context phase): two new tokens, no prefix.
    test_model->RunCompressor({forward});

    // Decode: append a token, mark prior two as cached.
    forward.forwarding_tokens->push_back(321);
    forward.prefix_cache_len = 0;
    forward.kv_cached_token_num = 2;
    test_model->RunCompressor({forward});
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
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_;
  std::shared_ptr<PrefixCacheManager> cache_manager;

  void* swa_pool_base_ = nullptr;
  size_t swa_block_size_ = 0;
  int64_t swa_pool_block_num_ = 0;
};

TEST_F(CompressorTest, ForwardC128) {
  InitWithRatio(128);
  SetupBf16Config();
  TestCompressorForward();
  Singleton<Environment>::GetInstance()->schedule_config_parser_.runtime_config_.compressor_config.state_dtype =
      TYPE_FP32;
  TestCompressorForward();
}

TEST_F(CompressorTest, ForwardC4) {
  InitWithRatio(4);
  SetupBf16Config();
  TestCompressorForward();
}

}  // namespace ksana_llm
