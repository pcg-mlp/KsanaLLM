/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>

#include "cache_manager/prefix_cache_manager.h"
#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"
#include "kernels/nvidia/kernel_wrapper.h"
#include "layers/cast_layer.h"
#include "models/base/model_weight.h"
#include "models/deepseek_v3/attention_meta/deepseek_v3_attention_meta_creator.h"
#include "modules/attention/multihead_latent_attention.h"
#include "samplers/sampler.h"
#include "tests/test.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/calc_intvec_hash.h"
#include "utils/get_custom_weight_name.h"
#include "utils/memory_allocator.h"
#include "utils/search_path.h"
#include "utils/singleton.h"

using namespace ksana_llm;

template <typename T>
DataType GetKsanaDataType();
#define GET_KSANA_DATA_TYPE(T, KSANA_TYPE) \
  template <>                              \
  DataType GetKsanaDataType<T>() {         \
    return KSANA_TYPE;                     \
  }
GET_KSANA_DATA_TYPE(int32_t, TYPE_INT32);
GET_KSANA_DATA_TYPE(float, TYPE_FP32);
GET_KSANA_DATA_TYPE(half, TYPE_FP16);
GET_KSANA_DATA_TYPE(__nv_bfloat16, TYPE_BF16);
#undef GET_KSANA_DATA_TYPE

size_t schedule_id = 0;

Status CastDeviceTensorType(Tensor& input_tensor, DataType new_dtype, int dev_rank, std::shared_ptr<Context>& context) {
#ifdef ENABLE_ACL
  KLLM_THROW(fmt::format("Unsupported tensor cast for Ascend device"));
#elif ENABLE_TOPS
  KLLM_THROW(fmt::format("Unsupported tensor cast for Tops device"));
#endif
  if (input_tensor.dtype == new_dtype) {
    return Status();
  }
#ifdef ENABLE_CUDA
  if (input_tensor.dtype == DataType::TYPE_FP32 && new_dtype == DataType::TYPE_FP16) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, nullptr,
                               &(context->device->GetMemoryManageStream()));
    CUDA_CHECK_LAST_ERROR(
        llm_kernels::nvidia::FloatToHalf(input_tensor.GetPtr<float>(), input_tensor.GetElementNumber(),
                                         new_tensor.GetPtr<float16>(), context->device->GetMemoryManageStream().Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_FP32 && new_dtype == DataType::TYPE_BF16) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, nullptr,
                               &(context->device->GetMemoryManageStream()));
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::FloatToBFloat16(
        input_tensor.GetPtr<float>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<bfloat16>(),
        context->device->GetMemoryManageStream().Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_FP16 && new_dtype == DataType::TYPE_FP32) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, nullptr,
                               &(context->device->GetMemoryManageStream()));
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::HalfToFloat(input_tensor.GetPtr<float16>(),
                                                           input_tensor.GetElementNumber(), new_tensor.GetPtr<float>(),
                                                           context->device->GetMemoryManageStream().Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_BF16 && new_dtype == DataType::TYPE_FP32) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, nullptr,
                               &(context->device->GetMemoryManageStream()));
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToFloat(
        input_tensor.GetPtr<bfloat16>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<float>(),
        context->device->GetMemoryManageStream().Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_BF16 && new_dtype == DataType::TYPE_FP16) {
    input_tensor.dtype = new_dtype;
    // Inplace cast
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToHalf(
        input_tensor.GetPtr<void>(), input_tensor.GetElementNumber(), context->device->GetMemoryManageStream().Get()));
  } else if (input_tensor.dtype == DataType::TYPE_FP16 && new_dtype == DataType::TYPE_BF16) {
    input_tensor.dtype = new_dtype;
    // Inplace cast
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::HalfToBFloat16(
        input_tensor.GetPtr<void>(), input_tensor.GetElementNumber(), context->device->GetMemoryManageStream().Get()));
  } else {
    KLLM_THROW(fmt::format("Unsupported tensor cast from type {} to {}", input_tensor.dtype, new_dtype));
  }
#endif
  return Status();
}

void AssignFromVector(Tensor& tensor, const std::vector<float>& f_vector, std::shared_ptr<Context>& context) {
  int device_rank;
  GetDevice(&device_rank);

  if (f_vector.size() != tensor.GetElementNumber()) {
    KLLM_THROW("Vector size does not match tensor element count");
  }

  Tensor fp32_cast = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, tensor.shape, nullptr,
                            &(context->device->GetMemoryManageStream()));
  MemcpyAsync(fp32_cast.GetPtr<void>(), f_vector.data(), f_vector.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context->device->GetMemoryManageStream());
  CastDeviceTensorType(fp32_cast, tensor.dtype, device_rank, context);
  MemcpyAsync(tensor.GetPtr<void>(), fp32_cast.GetPtr<void>(), tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
              context->device->GetMemoryManageStream());
  DeviceSynchronize();
}

template <typename T>
class MultiHeadLatentAttentionTestModel : public CommonModel {
 public:
  using CommonModel::context_;
  using CommonModel::forwarding_context_;
  using CommonModel::model_config_;

  using CommonModel::cast_layer_;

  std::shared_ptr<MultiHeadLatentAttention> mla_;
  MlaBuffers mla_buffers_;

  MultiHeadLatentAttentionTestModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                    const int rank, std::shared_ptr<Context> context,
                                    std::shared_ptr<BaseWeight> base_weight,
                                    std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
      : CommonModel(model_config, runtime_config, rank, context) {
    ModelRunConfig model_run_config;
    model_run_config.position_encoding = PositionEncoding::ROPE;
    CommonModel::InitRunConfig(model_run_config, base_weight, block_cache_layout);
  }

  ~MultiHeadLatentAttentionTestModel() {}

  std::unique_ptr<AttentionMetaCreator> CreateCustomAttentionMetaCreator(
      const AttentionMetaCreator::Config& config) override {
    if (model_config_.use_mla) {
      return std::make_unique<DeepSeekV3AttentionMetaCreator>(config);
    }
    return nullptr;
  }

  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override {
    MultiHeadLatentAttention::CreateBuffers(CommonModel::GetBufferManager(), model_creation_config.attn_config,
                                            creation_context.runtime_config, mla_buffers_);
    bool is_neox = true;
    int layer_idx = 0;
    mla_ = std::make_shared<MultiHeadLatentAttention>(layer_idx, is_neox, creation_context, model_creation_config,
                                                      mla_buffers_);
    return Status();
  }

  Status LayerForward(ForwardingContext& forwarding_context_, const RunMode run_mode = RunMode::kMain) override {
    return Status();
  }

  Status CommonAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                         const bool is_multi_token_forward, const std::vector<ForwardRequest>& forward_reqs) {
    forwarding_context_->model_input_->ParseFromRequests(forward_reqs);

    // create forward shape tensor
    forwarding_context_->GetAttentionForwardContext().forward_shape.shape = {
        forwarding_context_->model_input_->multi_token_request_num,
        forwarding_context_->model_input_->multi_token_request_max_tokens,
        forwarding_context_->model_input_->context_kv_cache_block_num,
        forwarding_context_->model_input_->single_token_request_num,
        forwarding_context_->model_input_->single_token_request_max_tokens,
        forwarding_context_->model_input_->decode_kv_cache_block_num,
        forwarding_context_->model_input_->dp_max_forwarding_tokens,
        forwarding_context_->model_input_->dp_multi_token_request_num,
        forwarding_context_->model_input_->dp_multi_token_request_max_tokens,
        forwarding_context_->model_input_->dp_single_token_request_num,
        forwarding_context_->model_input_->dp_single_token_request_max_tokens,
        forwarding_context_->model_input_->dp_total_prefix_len};

    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context_->buffers_->hidden_buffer_0);
    CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context_->buffers_->shared_buffer);

    forwarding_context_->GetAttentionForwardContext().flag_tensor.template GetPtr<bool>()[0] =
        forwarding_context_->model_input_->use_cache;
    hidden_buffer_tensors_0[0].shape = {forwarding_context_->model_input_->input_ids.shape[0],
                                        model_config_.hidden_units};
    hidden_buffer_tensors_0[0].dtype = model_config_.weight_data_type;
    std::vector<float> input_data;
    input_data.reserve(hidden_buffer_tensors_0[0].GetElementNumber());
    size_t count = 0;
    for (const auto& req : forward_reqs) {
      // Skip cached prefix
      count += req.prefix_cache_len * model_config_.hidden_units;
      for (size_t i = 0; i < req.GetInputIdsLength(); i++) {
        for (size_t j = 0; j < model_config_.hidden_units; j++) {
          input_data.push_back(1.0f / (count % 97 * 0.1f + 1.0f) * pow(-1, (count % 7)));
          ++count;
        }
      }
    }
    AssignFromVector(hidden_buffer_tensors_0[0], input_data, context_);
    Status status = mla_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, nullptr, is_multi_token_forward,
                                  *forwarding_context_);
    hidden_buffer_tensors_0[0].dtype = model_config_.weight_data_type;
    forwarding_context_->GetAttentionForwardContext().forward_shape.shape = {0, 1, 1};
    {
      CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context_->buffers_->hidden_buffer_1);
      STATUS_CHECK_RETURN(static_cast<CastLayer*>(cast_layer_.get())
                              ->Forward(&hidden_buffer_tensors_0[0], 0, 1, 1, &hidden_buffer_tensors_1[0]));
      StreamSynchronize(context_->device->GetPrimaryComputeStream());
      Memcpy(input_data.data(), hidden_buffer_tensors_1[0].template GetPtr<void>(),
             sizeof(float) * hidden_buffer_tensors_0[0].GetElementNumber(), MEMCPY_DEVICE_TO_HOST);
      // Reset dtype
      hidden_buffer_tensors_1[0].dtype = model_config_.weight_data_type;
    }
    std::vector<float> output;
    if (is_multi_token_forward) {
      if (model_config_.use_dsa && std::getenv("CONTEXT_DSA_THRESHOLD") != nullptr) {
        // Prefill phase (no prefix cache for DSA test)
        // The numerical difference here is due to sparse mla applies weight absorption for prefill tokens,
        // while in this test, weights w_uk_t/w_uv and kv_b_nope_proj/v_head_proj are not equivalent
        if (forwarding_context_->model_input_->dp_total_prefix_len == 0) {
          if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_DS_MLA) {
            output = {1416, 65.5, -1248, -2672, 856, -692};
          } else {
            output = {1408, 61.75, -1256, -2672, 856, -688};
          }
        } else {
          output = {604, 880, 374, -474, 134, -708};
        }
      } else if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_E4M3) {
        if (forwarding_context_->model_input_->dp_total_prefix_len == 0) {
          output = {-1984, 2096, -3392, 784, 117.5, -564};
        } else {
          output = {346, 588, -1008, 764, -648, 936};
        }
      } else if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_DS_MLA) {
        if (forwarding_context_->model_input_->dp_total_prefix_len == 0) {
          output = {-1976, 2096, -3392, 776, 126, -576};
        } else {
          output = {318, 632, -1032, 756, -672, 924};
        }
      } else {
        if (forwarding_context_->model_input_->dp_total_prefix_len == 0) {
          output = {-1977, 2088, -3386, 772.5, 125.75, -575.5};
        } else {
          output = {349, 585, -1004, 759, -643.5, 928};
        }
      }
    } else {
      if (model_config_.use_dsa) {
        if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_DS_MLA) {
          output = {744, 748, 118, -684, 201, -716};
        } else {
          output = {752, 716, -57, -920, 276, -728};
        }
      } else if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_E4M3) {
        output = {548, 940, 362, -45.75, -75.5, -564};
      } else {
        output = {559.5, 935.5, 366.25, -55.0312, -76.1875, -564};
      }
    }

    for (size_t i = 0; i < output.size(); i++) {
      const float diff = std::fabs((input_data[i] - output[i]) / output[i]);
      EXPECT_LT(diff, 0.05);
    }

    return status;
  }
};

template <typename T>
class TestWeight : public BaseWeight {
 public:
  using ksana_llm::BaseWeight::weights_map_;
  TestWeight() {}
  explicit TestWeight(const ksana_llm::ModelConfig& model_config, const ksana_llm::RuntimeConfig& runtime_config,
                      int rank, std::shared_ptr<ksana_llm::Context> context)
      : ksana_llm::BaseWeight(model_config, runtime_config, rank, context) {}
  ~TestWeight() override {}

  Tensor GetModelWeights(const std::string& weight_name) override {
    auto it = weights_map_.find(weight_name);
    if (it == weights_map_.end()) {
      KLLM_LOG_WARNING << fmt::format("weight_name: {} not in weights map", weight_name);
      return Tensor();
    }
    return it->second;
  }

  virtual Status LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                     const std::vector<std::string>& weight_name_list,
                                     const std::vector<std::string>& custom_name_list) override {
    return Status();
  }

  virtual void ProcessWeights() override { return; }
  virtual void SetEmbeddingsConfig() override { return; }

  void AddMlaWeight(int device_id = 0) {
    std::unordered_map<std::string, std::vector<size_t>> add_tensor_map;
    const int layer_num = 4;
    for (int i = 0; i < layer_num; i++) {
      add_tensor_map[fmt::format("model.layers.{}.self_attn.kv_a_layernorm.weight", i)] = {512};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.o_proj.weight", i)] = {2048, 2048};
      add_tensor_map[fmt::format("model.layers.{}.post_attention_layernorm.weight", i)] = {2048};
      add_tensor_map[fmt::format("model.layers.{}.input_layernorm.weight", i)] = {2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.kv_a_rope_proj.weight", i)] = {2048, 64};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.v_head_proj.weight", i)] = {512, 2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.kv_a_lora_proj.weight", i)] = {2048, 512};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.kv_b_nope_proj.weight", i)] = {512, 2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.w_uk_t.weight", i)] = {16, 128, 512};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.w_uv.weight", i)] = {16, 512, 128};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.q_b_nope_rope_proj.weight", i)] = {2048, 3072};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.q_a_proj.weight", i)] = {2048, 1536};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.q_a_layernorm.weight", i)] = {1536};
      // Add indexer weights
      add_tensor_map[fmt::format("model.layers.{}.self_attn.indexer.wq_b.weight", i)] = {4096, 1536};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.indexer.wk.weight", i)] = {128, 2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.indexer.weights_proj.weight", i)] = {32, 2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.indexer.k_norm.weight", i)] = {128};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.indexer.k_norm.bias", i)] = {128};
    }
    DataType weight_type = GetKsanaDataType<T>();
    for (auto& [tensor_name, shape] : add_tensor_map) {
      weights_map_[tensor_name] = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, shape, nullptr,
                                         &(context_->device->GetMemoryManageStream()));
      Tensor& tensor = weights_map_.at(tensor_name);
      std::vector<float> input_data(tensor.GetElementNumber());
      for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = 1.0f / (i % 97 * 0.1f + 1.0f) * pow(-1, (i % 7));
      }
      MemcpyAsync(tensor.template GetPtr<void>(), reinterpret_cast<void*>(input_data.data()), tensor.GetTotalBytes(),
                  MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
      CastDeviceTensorType(tensor, weight_type, device_id, context_);
    }
  }
};

// 定义一个 MlaTest 类,继承自 testing::Test
class MlaTest : public testing::Test {
 protected:
  void SetUp() override {
    origin_stderr_verbosity = loguru::g_stderr_verbosity;
    loguru::g_stderr_verbosity = loguru::Verbosity_MAX;

    const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    test_name = test_info->name();

    // 解析 config.json,初始化 ModelConfig 以及 BlockManager
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

    // kv_cache_dtype must be set before calling env->InitializeBlockManagerConfig()
    if (test_name.find("FlashMlaKvFP8") != std::string::npos) {
      env->schedule_config_parser_.runtime_config_.attn_backend_config.kv_cache_dtype = TYPE_FP8_E4M3;
    } else if (test_name.find("FlashDsa") != std::string::npos) {
      // For DSA test, we need to set use_dsa BEFORE InitializeBlockManagerConfig
      // so that indexer kv_cache_config can be properly initialized
      model_config.use_dsa = true;
      model_config.dsa_config.index_head_dim = 128;
      model_config.dsa_config.index_n_heads = 32;
      model_config.dsa_config.index_topk = 2048;
      model_config.mla_config.q_lora_rank = 1536;
      // Update the environment's model config
      env->SetModelConfig(model_config);
      if (test_name.find("FP8") != std::string::npos) {
        env->schedule_config_parser_.runtime_config_.attn_backend_config.kv_cache_dtype = TYPE_FP8_E4M3;
      } else {
        env->schedule_config_parser_.runtime_config_.attn_backend_config.kv_cache_dtype = TYPE_FP8_DS_MLA;
      }
    }
    block_cache_layout_ = GetTestModelCacheLayout(env);
    env->InitializeBlockManagerConfig(block_cache_layout_);
    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.block_host_memory_factor = 0;
    block_manager_config.reserved_device_memory_ratio = 0.98;
    block_manager_config.host_allocator_config.blocks_num = 0;
    block_manager_config.device_allocator_config.blocks_num = 10;
    env->SetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);

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

    CacheManagerConfig cache_manager_config;
    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = 1;
    cache_manager_config.swap_threadpool_size = 2;
    cache_manager_config.enable_prefix_caching = true;
    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, engine_block_group);
  }

  void TearDown() override { loguru::g_stderr_verbosity = origin_stderr_verbosity; }

 protected:
  int origin_stderr_verbosity = loguru::Verbosity_MAX;
  std::string test_name;
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  PipelineConfig pipeline_config;
  std::shared_ptr<BlockAllocatorManager> engine_block_manager;
  std::shared_ptr<BlockAllocatorManager> executor_block_manager;
  std::shared_ptr<BlockAllocatorGroupInterface> engine_block_group;
  std::shared_ptr<BlockAllocatorGroupInterface> executor_block_group;
  std::shared_ptr<Context> context;
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;

  std::shared_ptr<PrefixCacheManager> cache_manager;

  template <typename weight_data_type>
  void TestMlaForward(int device_id = 0) {
    SetDevice(device_id);
    std::shared_ptr<TestWeight<weight_data_type>> base_weight =
        std::make_shared<TestWeight<weight_data_type>>(model_config, runtime_config, 0, context);
    base_weight->AddMlaWeight(device_id);

    std::shared_ptr<ksana_llm::BaseWeight> bs1 = base_weight;
    std::shared_ptr<MultiHeadLatentAttentionTestModel<weight_data_type>> test_mla_model =
        std::make_shared<MultiHeadLatentAttentionTestModel<weight_data_type>>(model_config, runtime_config, 0, context,
                                                                              bs1, block_cache_layout_);

    // ContextDecode
    ForwardRequest forward;
    forward.block_allocator_manager = executor_block_manager;
    std::vector<int> input_ids = {233, 1681};
    forward.forwarding_tokens = std::make_shared<std::vector<int>>(input_ids);
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
    forward.flexible_cached_copy_tasks = &flexible_cached_copy_tasks;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    EmbeddingSlice embedding_slice;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;
    forward.input_refit_embedding = &embedding_slice;
    std::vector<int> block_ids;
    engine_block_group->GetDeviceBlockAllocator(0)->AllocateBlocks(1, block_ids);
    executor_block_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, forward.kv_cache_ptrs);
    forward.atb_kv_cache_base_blk_ids.assign(1, {});
    BuildFlatKVCacheBlockIds(GetTotalLayerNum(model_config, pipeline_config), {block_ids},
                             forward.atb_kv_cache_base_blk_ids, executor_block_group->GetDeviceBlockAllocator(0));

    test_mla_model->CommonAttention(0, bs1, true, {forward});

    // Test Prefix caching
    forward.forwarding_tokens->push_back(321);
    forward.prefix_cache_len = 1;
    forward.kv_cached_token_num = 1;
    test_mla_model->CommonAttention(0, bs1, true, {forward});
    forward.forwarding_tokens->pop_back();

    // Decode
    forward.forwarding_tokens->push_back(321);
    forward.prefix_cache_len = 0;
    forward.kv_cached_token_num = 2;
    test_mla_model->CommonAttention(0, bs1, false, {forward});
  }
};

// 运行时检测是否是 Hopper 架构的辅助函数
bool IsHopperSupported() {
#ifdef ENABLE_CUDA
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return false;
  }

  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    return false;
  }

  int compute_capability = major * 10 + minor;
  // Hopper 架构是 SM 9.0 (compute capability 90)
  return compute_capability >= 90;
#else
  return false;
#endif
}

TEST_F(MlaTest, ForwardWithFlashMlaTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
  // 运行时检测 GPU 架构，只有在 Hopper 架构时才运行测试
  if (!IsHopperSupported()) {
    GTEST_SKIP_("Test requires FA3 support or Hopper architecture (SM >= 9.0)");
    return;
  }
  // fp16 forward
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  model_config.quant_config.method = QUANT_NONE;
  runtime_config.inter_data_type = model_config.weight_data_type;
  const auto& env = Singleton<Environment>::GetInstance();
  env->schedule_config_parser_.runtime_config_.inter_data_type = model_config.weight_data_type;
  std::cout << "Test TYPE_FP16 weight_data_type forward." << std::endl;
  TestMlaForward<float16>();
  return;
}

TEST_F(MlaTest, ForwardWithFlashMlaKvFP8Test) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
  // 运行时检测 GPU 架构，只有在支持 FA3 或 Hopper 架构时才运行测试
  if (!IsHopperSupported()) {
    GTEST_SKIP_("Test requires FA3 support or Hopper architecture (SM >= 9.0)");
    return;
  }
  // bf16 forward, FA3 only supports BF16 output for FP8 input
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_BF16;
  model_config.quant_config.method = QUANT_NONE;
  runtime_config.inter_data_type = model_config.weight_data_type;
  const auto& env = Singleton<Environment>::GetInstance();
  env->schedule_config_parser_.runtime_config_.inter_data_type = model_config.weight_data_type;
  std::cout << "Test TYPE_BF16 weight_data_type forward." << std::endl;
  TestMlaForward<bfloat16>();

  // test kv scales, FA3 requires special scale input
  model_config.k_scales[0] = 0.5f;
  model_config.v_scales[0] = 0.5f;
  std::cout << "Test TYPE_BF16 weight_data_type forward with kv scales." << std::endl;
  TestMlaForward<bfloat16>();
  return;
}

TEST_F(MlaTest, ForwardWithFlashDsaTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif

  // 运行时检测 GPU 架构，只有在 Hopper 架构时才运行测试
  if (!IsHopperSupported()) {
    GTEST_SKIP_("Test requires Hopper architecture (SM >= 9.0)");
    return;
  }
  // BF16 forward, DeepSeek Sparse MLA only supports BF16
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_BF16;
  model_config.quant_config.method = QUANT_NONE;
  runtime_config.inter_data_type = model_config.weight_data_type;
  const auto& env = Singleton<Environment>::GetInstance();
  env->schedule_config_parser_.runtime_config_.inter_data_type = model_config.weight_data_type;
  std::cout << "Test TYPE_BF16 weight_data_type forward." << std::endl;
  // Adaptive fall back
  TestMlaForward<bfloat16>();
  // Force DSA
  setenv("CONTEXT_DSA_THRESHOLD", "0", 1);
  TestMlaForward<bfloat16>();
  // Unset the env
  unsetenv("CONTEXT_DSA_THRESHOLD");
  return;
}

TEST_F(MlaTest, ForwardWithFlashDsaFP8Test) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif

  // 运行时检测 GPU 架构，只有在 Hopper 架构时才运行测试
  if (!IsHopperSupported()) {
    GTEST_SKIP_("Test requires Hopper architecture (SM >= 9.0)");
    return;
  }
  // BF16 forward, DeepSeek Sparse MLA only supports BF16
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_BF16;
  model_config.quant_config.method = QUANT_NONE;
  runtime_config.inter_data_type = model_config.weight_data_type;
  const auto& env = Singleton<Environment>::GetInstance();
  env->schedule_config_parser_.runtime_config_.inter_data_type = model_config.weight_data_type;
  std::cout << "Test TYPE_BF16 weight_data_type forward with kv fp8." << std::endl;
  // Force DSA
  setenv("CONTEXT_DSA_THRESHOLD", "0", 1);
  TestMlaForward<bfloat16>();
  // Unset the env
  unsetenv("CONTEXT_DSA_THRESHOLD");
  return;
}
