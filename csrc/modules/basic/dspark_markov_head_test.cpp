/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/dspark_markov_head.h"

#include "configure/context.h"
#include "configure/environment.h"
#include "models/base/fake_weight_for_test.h"
#include "tests/test.h"
#include "utils/random_generator.h"
#include "utils/tensor.h"

namespace ksana_llm {

class DSparkMarkovHeadTest : public testing::Test {
 protected:
  void SetUp() override {
    model_config.hidden_units = 128;
    model_config.vocab_size = 32;
    model_config.layernorm_eps = 1e-6;
    model_config.weight_data_type = TYPE_BF16;
    runtime_config.inter_data_type = TYPE_BF16;
    runtime_config.max_step_token_num = 16;
    runtime_config.parallel_basic_config.tensor_parallel_size = 1;

    Singleton<Environment>::GetInstance()->schedule_config_parser_.runtime_config_ = runtime_config;

    context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(1, 1, 1, Singleton<Environment>::GetInstance()));
    context->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context->device = std::make_shared<DeviceContext>(1);
    context->device->ext->InitNcclParam(nccl_uid);
  }

  void TearDown() override {}

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  PipelineConfig pipeline_config;
  BufferManager buffer_mgr;
  std::shared_ptr<Context> context;

  // Fill a device tensor with values from a host vector<float> using dtype conversion.
  template <typename T>
  void FillDevice(Tensor& tensor, const std::vector<float>& src) {
    Tensor host(MemoryLocation::LOCATION_HOST, tensor.dtype, tensor.shape);
    std::vector<float> src_copy = src;
    CopyVectorToHostTensor(host, tensor.dtype, src_copy);
    Memcpy(tensor.GetPtr<void>(), host.GetPtr<void>(), tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE);
  }
};

// Full-path Forward with tp=1 (model_comm_ is null so AllGathers are skipped).
// Verifies embed lookup + w2 matmul + logits += bias against a CPU reference.
TEST_F(DSparkMarkovHeadTest, ForwardBf16Tp1) {
  SetDevice(0);
  buffer_mgr.SetRank(0);

  const size_t V = model_config.vocab_size;
  const size_t R = 16;
  const size_t B = 4;
  const std::string w1_name = "markov_head.markov_w1.weight";
  const std::string w2_name = "markov_head.markov_w2.weight";

  // Weights: markov_w1 [V, R] (embed lookup table), markov_w2 [R, V] (Linear stores as [in, out],
  // transposed from the file layout [V, R] by the weight loader).
  auto fake_weight = std::make_shared<FakeWeight>(0);
  fake_weight->CreateWeight(w1_name, {V, R}, TYPE_BF16, MemoryLocation::LOCATION_DEVICE);
  fake_weight->CreateWeight(w2_name, {R, V}, TYPE_BF16, MemoryLocation::LOCATION_DEVICE);

  RandomGenerator<float> gen(42);
  std::vector<float> w1_host(V * R), w2_host(V * R);
  for (auto& v : w1_host) v = gen(-0.1f, 0.1f);
  for (auto& v : w2_host) v = gen(-0.1f, 0.1f);
  // Transpose w2_host from logical [V, R] to device [R, V] before filling.
  std::vector<float> w2_host_transposed(V * R);
  for (size_t v = 0; v < V; v++) {
    for (size_t r = 0; r < R; r++) {
      w2_host_transposed[r * V + v] = w2_host[v * R + r];
    }
  }
  Tensor w1 = fake_weight->GetModelWeights(w1_name);
  Tensor w2 = fake_weight->GetModelWeights(w2_name);
  FillDevice<bfloat16>(w1, w1_host);
  FillDevice<bfloat16>(w2, w2_host_transposed);

  LayerCreationContext creation_ctx;
  creation_ctx.Init(fake_weight, context, 0, pipeline_config, model_config, runtime_config, &buffer_mgr);

  DSparkMarkovHead head(w1_name, w2_name, creation_ctx, DEFAULT_LINEAR_BACKEND, /*model_comm=*/nullptr);

  // Inputs.
  Tensor token_ids(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {B});
  std::vector<int32_t> token_ids_host = {0, 3, 7, static_cast<int32_t>(V - 1)};
  Memcpy(token_ids.GetPtr<void>(), token_ids_host.data(), token_ids.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE);

  Tensor logits(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {B, V});
  std::vector<float> logits_init(B * V);
  for (size_t i = 0; i < logits_init.size(); i++) logits_init[i] = 0.01f * static_cast<float>(i);
  Memcpy(logits.GetPtr<void>(), logits_init.data(), logits.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE);

  // Two scratch buffers: embed [B, R], bias [B, V].
  std::vector<Tensor> embed{Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {B, R})};
  std::vector<Tensor> bias{Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {B, V})};

  EXPECT_TRUE(head.Forward(token_ids, logits, embed, bias).OK());

  StreamSynchronize(context->device->GetPrimaryComputeStream());

  std::vector<float> logits_out(B * V);
  Memcpy(logits_out.data(), logits.GetPtr<void>(), logits.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);

  // CPU reference: bias[b, v] = sum_r bf16(w1[tok, r]) * bf16(w2[v, r]); logits[b, v] += bias.
  auto to_bf16 = [](float f) { return static_cast<float>(static_cast<bfloat16>(f)); };
  for (size_t b = 0; b < B; b++) {
    int32_t tok = token_ids_host[b];
    for (size_t v = 0; v < V; v++) {
      float acc = 0.0f;
      for (size_t r = 0; r < R; r++) {
        acc += to_bf16(w1_host[tok * R + r]) * to_bf16(w2_host[v * R + r]);
      }
      float expected = logits_init[b * V + v] + to_bf16(acc);
      EXPECT_NEAR(expected, logits_out[b * V + v], 5e-3f) << "b=" << b << " v=" << v;
    }
  }
}

}  // namespace ksana_llm
