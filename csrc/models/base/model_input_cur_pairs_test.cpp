/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/

// ModelInput::PrepareInputIds 单测：cur_batch_pairs + 统一 H2D pack / shared view。
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
  const bool orig_xgrammar = bsc.enable_xgrammar;
  bsc.enable_executor_build_ahead = true;
  bsc.enable_async = true;
  bsc.scheduler_type = SchedulerType::DEFAULT;
  // FastPath 与 xgrammar 互斥；服务默认开启 xgrammar 时，本测需显式关闭。
  bsc.enable_xgrammar = false;
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
  bsc.enable_xgrammar = orig_xgrammar;
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
  const bool orig_xgrammar = bsc.enable_xgrammar;
  bsc.enable_executor_build_ahead = true;
  bsc.enable_async = true;
  bsc.scheduler_type = SchedulerType::DEFAULT;
  // FastPath 与 xgrammar 互斥；服务默认开启 xgrammar 时，本测需显式关闭。
  bsc.enable_xgrammar = false;
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
  bsc.enable_xgrammar = orig_xgrammar;
  env->SetBatchSchedulerConfig(bsc);
  ReinitFastPathGate(bsc);
}

// 构造期：固定 meta + 末尾 input_ids 布局，公开 Tensor 必须是 pack 上的 shared view。
TEST_F(ModelInputCurPairsTest, InputPackLayoutAndSharedViews) {
  const auto& layout = model_input_->input_pack_layout_;
  const size_t max_batch_size = model_input_->runtime_config_.max_batch_size;
  const size_t max_token_num = model_input_->runtime_config_.max_step_token_num;

  EXPECT_EQ(layout.off_cur_batch_pairs, 0u);
  EXPECT_EQ(layout.off_logits_idx, layout.off_cur_batch_pairs + layout.cur_batch_pairs_bytes);
  EXPECT_EQ(layout.off_input_offset, layout.off_logits_idx + layout.logits_idx_bytes);
  EXPECT_EQ(layout.off_input_prefix, layout.off_input_offset + layout.batch_offset_bytes);
  EXPECT_EQ(layout.off_dp_input_offset, layout.off_input_prefix + layout.batch_offset_bytes);
  EXPECT_EQ(layout.off_dp_input_prefix, layout.off_dp_input_offset + layout.batch_offset_bytes);
  EXPECT_EQ(layout.off_dp_prefill_q_offset, layout.off_dp_input_prefix + layout.batch_offset_bytes);
  EXPECT_EQ(layout.meta_pack_bytes, layout.off_dp_prefill_q_offset + layout.batch_offset_bytes);
  EXPECT_EQ(layout.off_input_ids, layout.meta_pack_bytes);
  EXPECT_EQ(layout.total_pack_bytes, layout.meta_pack_bytes + max_token_num * sizeof(int32_t));
  EXPECT_EQ(layout.cur_batch_pairs_bytes, 2 * max_batch_size * sizeof(int64_t));
  EXPECT_EQ(layout.batch_offset_bytes, (max_batch_size + 1) * sizeof(uint64_t));

  ASSERT_EQ(model_input_->input_pack_device_.GetTotalBytes(), layout.total_pack_bytes);
  ASSERT_EQ(model_input_->input_pack_host_.GetTotalBytes(), layout.total_pack_bytes);

  uint8_t* pack = model_input_->input_pack_device_.GetPtr<uint8_t>();
  EXPECT_EQ(reinterpret_cast<uint8_t*>(model_input_->cur_batch_pairs_tensor.GetPtr<void>()),
            pack + layout.off_cur_batch_pairs);
  EXPECT_EQ(reinterpret_cast<uint8_t*>(model_input_->logits_idx_uint64_tensor.GetPtr<void>()),
            pack + layout.off_logits_idx);
  EXPECT_EQ(reinterpret_cast<uint8_t*>(model_input_->input_offset_uint64_tensor.GetPtr<void>()),
            pack + layout.off_input_offset);
  EXPECT_EQ(reinterpret_cast<uint8_t*>(model_input_->input_prefix_uint64_tensor.GetPtr<void>()),
            pack + layout.off_input_prefix);
  EXPECT_EQ(reinterpret_cast<uint8_t*>(model_input_->dp_input_offset_uint64_tensor.GetPtr<void>()),
            pack + layout.off_dp_input_offset);
  EXPECT_EQ(reinterpret_cast<uint8_t*>(model_input_->dp_input_prefix_uint64_tensor.GetPtr<void>()),
            pack + layout.off_dp_input_prefix);
  EXPECT_EQ(reinterpret_cast<uint8_t*>(model_input_->dp_prefill_q_offset_uint64_tensor.GetPtr<void>()),
            pack + layout.off_dp_prefill_q_offset);
  EXPECT_EQ(reinterpret_cast<uint8_t*>(model_input_->input_ids.GetPtr<void>()), pack + layout.off_input_ids);
}

// PrepareInputIds 一次 H2D 后：经 shared view 读回的 device 数据须与 CPU 源一致。
TEST_F(ModelInputCurPairsTest, PrepareInputIdsUnifiedPackH2DMatchesCpuSources) {
  // prefill（kv_cached=0 → kFlash）+ decode，覆盖 offset/prefix/logits/input_ids 多段。
  auto prefill = MakeDecodeForwardRequest(601, {10, 20, 30, 40, 50}, /*kv_cached=*/0);
  auto decode = MakeDecodeForwardRequest(602, {11, 22, 33, 44}, /*kv_cached=*/3);
  ASSERT_EQ(prefill.GetType(), ForwardRequestType::kFlash);
  ASSERT_EQ(decode.GetType(), ForwardRequestType::kPage);

  InvokePrepareInputIds(model_input_.get(), {prefill, decode});
  StreamSynchronize(context_->device->GetH2DStream());

  // host 侧期望
  const std::vector<int> expect_ids = {10, 20, 30, 40, 50, 44};
  ASSERT_EQ(model_input_->input_ids_cpu, expect_ids);
  ASSERT_EQ(model_input_->input_ids.shape[0], expect_ids.size());

  std::vector<int32_t> ids_dev(expect_ids.size());
  MemcpyAsync(ids_dev.data(), model_input_->input_ids.GetPtr<void>(), ids_dev.size() * sizeof(int32_t),
              MEMCPY_DEVICE_TO_HOST, context_->device->GetH2DStream());
  StreamSynchronize(context_->device->GetH2DStream());
  EXPECT_EQ(ids_dev, expect_ids);

  // logits：每 req sampling_token_num=1 → 下标指向各 req 末 token 在 flat input_ids 中的位置
  ASSERT_EQ(model_input_->logits_idx_uint64_tensor.shape[0], 2u);
  std::vector<uint64_t> logits_dev(2);
  MemcpyAsync(logits_dev.data(), model_input_->logits_idx_uint64_tensor.GetPtr<void>(),
              logits_dev.size() * sizeof(uint64_t), MEMCPY_DEVICE_TO_HOST, context_->device->GetH2DStream());
  StreamSynchronize(context_->device->GetH2DStream());
  EXPECT_EQ(logits_dev[0], 4u);  // prefill 末 token
  EXPECT_EQ(logits_dev[1], 5u);  // decode 末 token

  // input_offset / input_prefix：累积 forwarding 长度与 skip
  ASSERT_EQ(model_input_->input_offset_list_uint64.size(), 3u);
  ASSERT_EQ(model_input_->input_prefix_list_uint64.size(), 3u);
  std::vector<uint64_t> offset_dev(3);
  std::vector<uint64_t> prefix_dev(3);
  MemcpyAsync(offset_dev.data(), model_input_->input_offset_uint64_tensor.GetPtr<void>(),
              offset_dev.size() * sizeof(uint64_t), MEMCPY_DEVICE_TO_HOST, context_->device->GetH2DStream());
  MemcpyAsync(prefix_dev.data(), model_input_->input_prefix_uint64_tensor.GetPtr<void>(),
              prefix_dev.size() * sizeof(uint64_t), MEMCPY_DEVICE_TO_HOST, context_->device->GetH2DStream());
  StreamSynchronize(context_->device->GetH2DStream());
  EXPECT_EQ(offset_dev[0], 0u);
  EXPECT_EQ(offset_dev[1], 5u);
  EXPECT_EQ(offset_dev[2], 9u);
  EXPECT_EQ(prefix_dev[0], 0u);
  EXPECT_EQ(prefix_dev[1], 0u);
  EXPECT_EQ(prefix_dev[2], 3u);

  // 经 pack 基址 + 布局偏移读 input_ids，须与 public view 同一块内存（已在上一条用例验证指针）；
  // 此处再确认 D2H 内容与 host_pack 中拼好的片段一致。
  const auto& layout = model_input_->input_pack_layout_;
  std::vector<int32_t> ids_via_pack(expect_ids.size());
  MemcpyAsync(ids_via_pack.data(), model_input_->input_pack_device_.GetPtr<uint8_t>() + layout.off_input_ids,
              ids_via_pack.size() * sizeof(int32_t), MEMCPY_DEVICE_TO_HOST, context_->device->GetH2DStream());
  StreamSynchronize(context_->device->GetH2DStream());
  EXPECT_EQ(ids_via_pack, expect_ids);
}

}  // namespace ksana_llm

#endif  // ENABLE_CUDA
