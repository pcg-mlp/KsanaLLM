/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/

// build-ahead 路径下 ModelInput::PrepareInputIds 的 cur_batch_pairs 填充单测。
// 不依赖外部模型权重：SetUp 在临时目录写入最小 config.json，仅满足 ParseConfig 所需字段。

#ifdef ENABLE_CUDA

#  include <unistd.h>

#  include <filesystem>
#  include <fstream>
#  include <memory>
#  include <string>
#  include <vector>

#  include "configure/global_context.h"
#  include "device/nvidia/nccl_utils.h"
#  include "helpers/block_cache_layout_helper.h"
#  include "helpers/model_test_helper.h"
#  include "models/base/model_input.h"
#  include "runtime/fast_path_controller.h"
#  include "tests/test.h"
#  include "utils/singleton.h"

namespace ksana_llm {

namespace {

// 在临时目录写入最小 llama config.json，供 ParseConfig 解析；无需真实权重文件。
void WriteMinimalLlamaConfigJson(const std::filesystem::path& config_path) {
  std::ofstream config_json(config_path);
  ASSERT_TRUE(config_json.is_open()) << "failed to write " << config_path;
  config_json << R"({
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "torch_dtype": "float16",
  "vocab_size": 32000,
  "hidden_size": 128,
  "intermediate_size": 256,
  "num_attention_heads": 4,
  "num_key_value_heads": 4,
  "num_hidden_layers": 2,
  "max_position_embeddings": 1024,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "rms_norm_eps": 1e-6
})";
}

std::filesystem::path MakeTempModelDir() {
  const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string suffix = test_info != nullptr ? test_info->name() : "unknown";
  std::filesystem::path model_dir =
      std::filesystem::temp_directory_path() / ("ksana_cur_pairs_model_" + std::to_string(::getpid()) + "_" + suffix);
  if (std::filesystem::exists(model_dir)) {
    std::filesystem::remove_all(model_dir);
  }
  std::filesystem::create_directories(model_dir);
  WriteMinimalLlamaConfigJson(model_dir / "config.json");
  return model_dir;
}

class ModelInputCurPairsTest : public testing::Test {
 protected:
  void SetUp() override {
    temp_model_dir_ = MakeTempModelDir();

    const auto& env = Singleton<Environment>::GetInstance();
    std::filesystem::path config_path = std::filesystem::absolute(std::filesystem::path(__FILE__).parent_path() /
                                                                  "../../../examples/llama7b/ksana_llm.yaml");
    ASSERT_TRUE(env->ParseConfig(config_path.string(), temp_model_dir_.string()).OK());

    context_ = std::make_shared<Context>();
    context_->global.reset(new GlobalContext(1, 1, 1, env));
    context_->global->InitDistributedConfig();
    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context_->device = std::make_shared<DeviceContext>(1);
    context_->device->ext->InitNcclParam(nccl_uid);

    env->GetModelConfig(model_config_);
    InitPipelineConfigForTest(env, model_config_);

    auto block_cache_layout = GetTestModelCacheLayout(env);
    env->InitializeBlockManagerConfig(block_cache_layout);
    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.block_host_memory_factor = 0.0;
    block_manager_config.device_allocator_config.blocks_num = 10;
    block_manager_config.host_allocator_config.blocks_num = 10;
    env->SetBlockManagerConfig(block_manager_config);

    env->GetRuntimeConfig(runtime_config_);
    model_input_ =
        std::make_unique<ModelInput>(model_config_, runtime_config_, /*rank=*/0, context_, block_cache_layout);

    FastPathController::GetInstance().ResetForTesting();
  }

  void TearDown() override {
    if (!temp_model_dir_.empty() && std::filesystem::exists(temp_model_dir_)) {
      std::filesystem::remove_all(temp_model_dir_);
    }
  }

  void ReinitFastPathGate(const BatchSchedulerConfig& bsc) {
    FastPathController::GetInstance().ResetForTesting();
    FastPathController::GetInstance().InitializeAtStartup(bsc, runtime_config_, model_config_);
  }

  std::filesystem::path temp_model_dir_;
  std::shared_ptr<Context> context_;
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  std::unique_ptr<ModelInput> model_input_;
};

ForwardRequest MakeDecodeForwardRequest(int64_t req_id, const std::vector<int>& tokens, size_t kv_cached) {
  ForwardRequest req;
  req.req_id = req_id;
  req.forwarding_tokens = std::make_shared<std::vector<int>>(tokens);
  req.kv_cached_token_num = kv_cached;
  req.prefix_cache_len = 0;
  req.sampling_token_num = 1;
  req.attn_dp_group_id = 0;
  return req;
}

// PrepareInputIds 由 ParseFromRequests 调用时会先累加 total_sampling_token_num_；
// 单测直调时需手动设置，否则 logits_idx_list 长度为 0 会越界。
void InvokePrepareInputIds(ModelInput* model_input, const std::vector<ForwardRequest>& forward_reqs) {
  size_t total_sampling = 0;
  for (const auto& req : forward_reqs) {
    total_sampling += req.sampling_token_num;
  }
  // 测试目标带 -fno-access-control，可直写 ParseFromRequests 才会设置的字段。
  model_input->total_sampling_token_num_ = total_sampling;
  model_input->PrepareInputIds(forward_reqs);
}

}  // namespace

TEST_F(ModelInputCurPairsTest, PrepareInputIdsFillsCurBatchPairsWhenBuildAheadEnabled) {
  const auto& env = Singleton<Environment>::GetInstance();
  BatchSchedulerConfig bsc;
  env->GetBatchSchedulerConfig(bsc);
  const bool orig_flag = bsc.enable_executor_build_ahead;
  bsc.enable_executor_build_ahead = true;
  bsc.enable_async = true;
  bsc.scheduler_type = SchedulerType::DEFAULT;
  env->SetBatchSchedulerConfig(bsc);
  runtime_config_.enable_prefix_caching = false;
  runtime_config_.enable_flexible_caching = false;
  runtime_config_.separate_prefill_decode = false;
  runtime_config_.max_pp_batch_num = 1;
  ReinitFastPathGate(bsc);
  ASSERT_TRUE(FastPathController::GetInstance().IsEnabledAtStartup());

  auto req1 = MakeDecodeForwardRequest(101, {1, 2, 3, 4}, /*kv_cached=*/3);
  auto req2 = MakeDecodeForwardRequest(202, {5, 6, 7, 8}, /*kv_cached=*/3);
  InvokePrepareInputIds(model_input_.get(), {req1, req2});

  ASSERT_EQ(model_input_->input_ids_cpu.size(), 2u);
  EXPECT_EQ(model_input_->input_ids_cpu[0], 4);
  EXPECT_EQ(model_input_->input_ids_cpu[1], 8);
  ASSERT_EQ(model_input_->cur_batch_pairs_tensor.shape.size(), 1u);
  EXPECT_EQ(model_input_->cur_batch_pairs_tensor.shape[0], 4u);

  std::vector<int64_t> pairs_host(4);
  MemcpyAsync(pairs_host.data(), model_input_->cur_batch_pairs_tensor.GetPtr<void>(),
              pairs_host.size() * sizeof(int64_t), MEMCPY_DEVICE_TO_HOST, context_->device->GetH2DStream());
  StreamSynchronize(context_->device->GetH2DStream());
  EXPECT_EQ(pairs_host[0], 101);
  EXPECT_EQ(pairs_host[1], 0);
  EXPECT_EQ(pairs_host[2], 202);
  EXPECT_EQ(pairs_host[3], 1);

  bsc.enable_executor_build_ahead = orig_flag;
  env->SetBatchSchedulerConfig(bsc);
  ReinitFastPathGate(bsc);
}

TEST_F(ModelInputCurPairsTest, PrepareInputIdsSkipsCurBatchPairsWhenBuildAheadDisabled) {
  const auto& env = Singleton<Environment>::GetInstance();
  BatchSchedulerConfig bsc;
  env->GetBatchSchedulerConfig(bsc);
  const bool orig_flag = bsc.enable_executor_build_ahead;
  bsc.enable_executor_build_ahead = false;
  env->SetBatchSchedulerConfig(bsc);
  ReinitFastPathGate(bsc);

  auto req = MakeDecodeForwardRequest(303, {1, 2, 3, 9}, /*kv_cached=*/3);
  InvokePrepareInputIds(model_input_.get(), {req});
  // 关闭 build-ahead 时仅将 shape 置为 {0}，底层 buffer 仍在构造期预分配。
  EXPECT_TRUE(model_input_->cur_batch_pairs_cpu_.empty());
  ASSERT_EQ(model_input_->cur_batch_pairs_tensor.shape.size(), 1u);
  EXPECT_EQ(model_input_->cur_batch_pairs_tensor.shape[0], 0u);

  bsc.enable_executor_build_ahead = orig_flag;
  env->SetBatchSchedulerConfig(bsc);
  ReinitFastPathGate(bsc);
}

TEST_F(ModelInputCurPairsTest, PrepareInputIdsSkipsZeroLenDecodeInCurBatchPairs) {
  const auto& env = Singleton<Environment>::GetInstance();
  BatchSchedulerConfig bsc;
  env->GetBatchSchedulerConfig(bsc);
  const bool orig_flag = bsc.enable_executor_build_ahead;
  bsc.enable_executor_build_ahead = true;
  bsc.enable_async = true;
  bsc.scheduler_type = SchedulerType::DEFAULT;
  env->SetBatchSchedulerConfig(bsc);
  runtime_config_.enable_prefix_caching = false;
  runtime_config_.enable_flexible_caching = false;
  runtime_config_.separate_prefill_decode = false;
  runtime_config_.max_pp_batch_num = 1;
  ReinitFastPathGate(bsc);
  ASSERT_TRUE(FastPathController::GetInstance().IsEnabledAtStartup());

  // kv_cached == token 数 → input_ids_len=0，不应写入 cur_batch_pairs。
  auto zero_len = MakeDecodeForwardRequest(404, {9, 10, 11}, /*kv_cached=*/3);
  auto normal = MakeDecodeForwardRequest(505, {1, 2, 3, 7}, /*kv_cached=*/3);
  InvokePrepareInputIds(model_input_.get(), {zero_len, normal});

  ASSERT_EQ(model_input_->input_ids_cpu.size(), 1u);
  EXPECT_EQ(model_input_->input_ids_cpu[0], 7);
  ASSERT_EQ(model_input_->cur_batch_pairs_tensor.shape.size(), 1u);
  EXPECT_EQ(model_input_->cur_batch_pairs_tensor.shape[0], 2u);

  std::vector<int64_t> pairs_host(2);
  MemcpyAsync(pairs_host.data(), model_input_->cur_batch_pairs_tensor.GetPtr<void>(),
              pairs_host.size() * sizeof(int64_t), MEMCPY_DEVICE_TO_HOST, context_->device->GetH2DStream());
  StreamSynchronize(context_->device->GetH2DStream());
  EXPECT_EQ(pairs_host[0], 505);
  EXPECT_EQ(pairs_host[1], 0);

  bsc.enable_executor_build_ahead = orig_flag;
  env->SetBatchSchedulerConfig(bsc);
  ReinitFastPathGate(bsc);
}

}  // namespace ksana_llm

#endif  // ENABLE_CUDA
