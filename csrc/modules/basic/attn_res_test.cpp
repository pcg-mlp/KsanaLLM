/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/attn_res.h"

#include "configure/context.h"
#include "configure/environment.h"
#include "models/base/fake_weight_for_test.h"
#include "tests/test.h"
#include "utils/tensor.h"

using namespace ksana_llm;

class AttnResTest : public testing::Test {
 protected:
  static constexpr size_t kHiddenSize = 7168;
  static constexpr size_t kMaxBlocks = 8;
  static constexpr float kEps = 1e-6f;
  static constexpr const char* kWeightPrefix = "model.layers.0.attn_res";
  static constexpr const char* kOutputNormName = "model.layers.0.output_norm.weight";

  void SetUp() override {
    model_config.hidden_units = kHiddenSize;
    model_config.layernorm_eps = kEps;
    model_config.weight_data_type = TYPE_BF16;
    runtime_config.inter_data_type = TYPE_BF16;
    runtime_config.max_step_token_num = 16;

    auto env = Singleton<Environment>::GetInstance();
    env->schedule_config_parser_.runtime_config_ = runtime_config;

    context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(1, 1, 1, env));
    context->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context->device = std::make_shared<DeviceContext>(1);
    context->device->ext->InitNcclParam(nccl_uid);
  }

  void TearDown() override {}

  LayerCreationContext CreateCtx(std::shared_ptr<FakeWeight> fake_weight) {
    SetDevice(0);
    buffer_mgr.SetRank(0);
    LayerCreationContext ctx;
    ctx.Init(fake_weight, context, 0, pipeline_config, model_config, runtime_config, &buffer_mgr);
    return ctx;
  }

  void FillDeterministic(Tensor& tensor, int seed_offset) {
    const size_t n = tensor.GetElementNumber();
    std::vector<bfloat16> host(n);
    for (size_t i = 0; i < n; i++) {
      const size_t k = i + static_cast<size_t>(seed_offset);
      host[i] = static_cast<bfloat16>(1.0f / (k % 97 * 0.1f + 1.0f) * ((k % 7) % 2 ? -1.0f : 1.0f));
    }
    Memcpy(tensor.GetPtr<void>(), host.data(), tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE);
    DeviceSynchronize();
  }

  std::vector<bfloat16> ReadBf16(const Tensor& tensor) {
    std::vector<bfloat16> host(tensor.GetElementNumber());
    Memcpy(host.data(), tensor.GetPtr<void>(), tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
    DeviceSynchronize();
    return host;
  }

  std::shared_ptr<FakeWeight> MakeWeights(bool with_output_norm) {
    auto fw = std::make_shared<FakeWeight>(0);
    fw->CreateWeight(std::string(kWeightPrefix) + "_norm.weight", {kHiddenSize}, TYPE_BF16,
                     MemoryLocation::LOCATION_DEVICE);
    fw->CreateWeight(std::string(kWeightPrefix) + "_proj.weight", {kHiddenSize}, TYPE_BF16,
                     MemoryLocation::LOCATION_DEVICE);
    FillDeterministic(fw->weights_map_[std::string(kWeightPrefix) + "_norm.weight"], /*seed_offset=*/101);
    FillDeterministic(fw->weights_map_[std::string(kWeightPrefix) + "_proj.weight"], /*seed_offset=*/103);
    if (with_output_norm) {
      fw->CreateWeight(kOutputNormName, {kHiddenSize}, TYPE_BF16, MemoryLocation::LOCATION_DEVICE);
      FillDeterministic(fw->weights_map_[kOutputNormName], /*seed_offset=*/107);
    }
    return fw;
  }

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  PipelineConfig pipeline_config;
  BufferManager buffer_mgr;
  std::shared_ptr<Context> context;
};

// num_blocks = 0, no delta, no write_block, no output norm. Softmax degenerates
// to identity on prefix so output == prefix (BF16 round-trip).
TEST_F(AttnResTest, ForwardPrefixOnly) {
  auto ctx = CreateCtx(MakeWeights(/*with_output_norm=*/false));
  AttnRes attn_res(kWeightPrefix, ctx);

  const size_t num_tokens = 4;
  Tensor prefix(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, kHiddenSize});
  Tensor bank(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, kMaxBlocks, kHiddenSize});
  Tensor blocks = bank.GetView({num_tokens, 0, kHiddenSize});
  Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, kHiddenSize});
  FillDeterministic(prefix, /*seed_offset=*/0);
  auto prefix_host = ReadBf16(prefix);

  std::vector<Tensor> in{prefix, blocks};
  std::vector<Tensor> out{output};
  EXPECT_TRUE(attn_res.Forward(in, out, /*write_block=*/false).OK());
  EXPECT_EQ(out[0].shape[0], num_tokens);
  EXPECT_EQ(out[0].shape[1], kHiddenSize);

  auto output_host = ReadBf16(out[0]);
  for (size_t i = 0; i < output_host.size(); i++) {
    EXPECT_NEAR(static_cast<float>(output_host[i]), static_cast<float>(prefix_host[i]), 1e-3f)
        << "mismatch at index " << i;
  }
}

// Delta + write_block: prefix updated in place to prefix + delta and
// snapshotted into blocks[:, num_valid_blocks, :].
TEST_F(AttnResTest, ForwardWithDeltaAndWriteBlock) {
  auto ctx = CreateCtx(MakeWeights(/*with_output_norm=*/false));
  AttnRes attn_res(kWeightPrefix, ctx);

  const size_t num_tokens = 2;
  const size_t num_valid_blocks = 2;
  Tensor bank(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, kMaxBlocks, kHiddenSize});
  FillDeterministic(bank, /*seed_offset=*/7);
  Tensor blocks_view = bank.GetView({num_tokens, num_valid_blocks, kHiddenSize});

  Tensor prefix(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, kHiddenSize});
  Tensor delta(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, kHiddenSize});
  Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, kHiddenSize});
  FillDeterministic(prefix, /*seed_offset=*/11);
  FillDeterministic(delta, /*seed_offset=*/23);
  auto prefix_before = ReadBf16(prefix);
  auto delta_before = ReadBf16(delta);

  std::vector<Tensor> in{prefix, blocks_view, delta};
  std::vector<Tensor> out{output};
  EXPECT_TRUE(attn_res.Forward(in, out, /*write_block=*/true).OK());
  EXPECT_EQ(out[0].shape[0], num_tokens);
  EXPECT_EQ(out[0].shape[1], kHiddenSize);

  auto output_host = ReadBf16(out[0]);
  std::vector<float> output_ref{0.58984375, -0.5546875, 0.52734375, -0.5, 0.4765625};
  for (size_t i = 0; i < output_ref.size(); i++) {
    EXPECT_NEAR(output_ref[i], static_cast<float>(output_host[i]), 1e-3f);
  }

  // Prefix updated in place to (prefix + delta) with bf16 rounding.
  auto prefix_after = ReadBf16(prefix);
  for (size_t i = 0; i < prefix_after.size(); i++) {
    float expected = static_cast<float>(
        static_cast<bfloat16>(static_cast<float>(prefix_before[i]) + static_cast<float>(delta_before[i])));
    EXPECT_NEAR(static_cast<float>(prefix_after[i]), expected, 1e-3f);
  }

  // Bank row [num_valid_blocks] equals updated prefix.
  auto bank_host = ReadBf16(bank);
  for (size_t t = 0; t < num_tokens; t++) {
    for (size_t d = 0; d < kHiddenSize; d++) {
      const size_t bank_idx = (t * kMaxBlocks + num_valid_blocks) * kHiddenSize + d;
      const size_t prefix_idx = t * kHiddenSize + d;
      EXPECT_EQ(static_cast<float>(bank_host[bank_idx]), static_cast<float>(prefix_after[prefix_idx]))
          << "mismatch at token=" << t << " d=" << d;
    }
  }
}

// Softmax mix over valid blocks + output RMSNorm.
TEST_F(AttnResTest, ForwardWithOutputNorm) {
  auto ctx = CreateCtx(MakeWeights(/*with_output_norm=*/true));
  AttnRes attn_res(kWeightPrefix, ctx, kOutputNormName);

  const size_t num_tokens = 3;
  const size_t num_valid_blocks = 1;
  Tensor bank(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, kMaxBlocks, kHiddenSize});
  FillDeterministic(bank, /*seed_offset=*/31);
  Tensor blocks_view = bank.GetView({num_tokens, num_valid_blocks, kHiddenSize});

  Tensor prefix(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, kHiddenSize});
  Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, kHiddenSize});
  FillDeterministic(prefix, /*seed_offset=*/47);

  std::vector<Tensor> in{prefix, blocks_view};
  std::vector<Tensor> out{output};
  EXPECT_TRUE(attn_res.Forward(in, out, /*write_block=*/false).OK());
  EXPECT_EQ(out[0].shape[0], num_tokens);
  EXPECT_EQ(out[0].shape[1], kHiddenSize);

  auto output_host = ReadBf16(out[0]);
  std::vector<float> output_ref{-0.388671875, -0.361328125, -0.3359375, -0.314453125};
  for (size_t i = 0; i < output_ref.size(); i++) {
    EXPECT_NEAR(output_ref[i], static_cast<float>(output_host[i]), 1e-3f);
  }
}
