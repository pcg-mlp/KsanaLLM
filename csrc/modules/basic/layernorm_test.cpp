/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/layernorm.h"

#include "configure/context.h"
#include "configure/environment.h"
#include "models/base/fake_weight_for_test.h"
#include "tests/test.h"
#include "utils/tensor.h"

using namespace ksana_llm;

class LayernormTest : public testing::Test {
 protected:
  void SetUp() override {
    model_config.hidden_units = 256;
    model_config.layernorm_eps = 1e-6;
    model_config.head_num = 8;
    runtime_config.max_step_token_num = 10;

    context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(1, 1, 1, Singleton<Environment>::GetInstance()));
    context->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context->device = std::make_shared<DeviceContext>(1);
    context->device->ext->InitNcclParam(nccl_uid);
  }

  void TearDown() override {}

  LayerCreationContext CreateCtx(std::shared_ptr<FakeWeight> fake_weight, DataType dtype) {
    model_config.weight_data_type = dtype;
    runtime_config.inter_data_type = dtype;
    Singleton<Environment>::GetInstance()->schedule_config_parser_.runtime_config_ = runtime_config;
    SetDevice(0);
    buffer_mgr.SetRank(0);

    LayerCreationContext ctx;
    ctx.Init(fake_weight, context, 0, pipeline_config, model_config, runtime_config, &buffer_mgr);
    return ctx;
  }

  void FillInputData(Tensor& tensor) {
    size_t num_elements = tensor.GetElementNumber();
    std::vector<float> host_data(num_elements);
    CustomRandomGenerator rng(42);
    for (size_t i = 0; i < num_elements; i++) {
      host_data[i] = rng.Normal(0.0f, 0.5f);
    }
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, tensor.dtype, tensor.shape);
    CopyVectorToHostTensor(host_tensor, tensor.dtype, host_data);
    Memcpy(tensor.GetPtr<void>(), host_tensor.GetPtr<void>(), tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE);
  }

  void VerifyOutput(const Tensor& output, DataType dtype) {
    size_t n = output.GetElementNumber();
    size_t bytes = output.GetTotalBytes();
    bool has_nonzero = false;

    auto check = [&](auto* ptr) {
      using T = std::remove_pointer_t<decltype(ptr)>;
      std::vector<T> host(n);
      Memcpy(host.data(), output.GetPtr<void>(), bytes, MEMCPY_DEVICE_TO_HOST);
      for (size_t i = 0; i < n; i++) {
        float val = static_cast<float>(host[i]);
        EXPECT_FALSE(std::isnan(val)) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(val)) << "Inf at index " << i;
        if (val != 0.0f) {
          has_nonzero = true;
        }
      }
    };

    if (dtype == TYPE_FP16) {
      check(static_cast<half*>(nullptr));
    } else if (dtype == TYPE_BF16) {
      check(static_cast<__nv_bfloat16*>(nullptr));
    } else {
      check(static_cast<float*>(nullptr));
    }

    EXPECT_TRUE(has_nonzero) << "Output is all zeros";
  }

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  PipelineConfig pipeline_config;
  BufferManager buffer_mgr;
  std::shared_ptr<Context> context;
};

// Forward with weight, no bias (RMSNorm path).
TEST_F(LayernormTest, ForwardRMSNorm) {
  struct Config {
    DataType dtype;
    size_t token_num;
    size_t hidden_size;
  };
  Config configs[] = {
      {TYPE_FP16, 4, 256}, {TYPE_BF16, 4, 256}, {TYPE_FP32, 4, 256}, {TYPE_FP16, 1, 128}, {TYPE_FP16, 64, 1024}};

  for (auto& cfg : configs) {
    const std::string weight_name = "test.layernorm.weight";
    auto fake_weight = std::make_shared<FakeWeight>(0);
    fake_weight->CreateWeight(weight_name, {cfg.hidden_size}, cfg.dtype, MemoryLocation::LOCATION_DEVICE);

    auto ctx = CreateCtx(fake_weight, cfg.dtype);
    Layernorm layernorm(weight_name, model_config.layernorm_eps, ctx);

    Tensor input(MemoryLocation::LOCATION_DEVICE, cfg.dtype, {cfg.token_num, cfg.hidden_size});
    FillInputData(input);
    Tensor output(MemoryLocation::LOCATION_DEVICE, cfg.dtype, {cfg.token_num, cfg.hidden_size});

    EXPECT_TRUE(layernorm.Forward(input, output).OK());
    VerifyOutput(output, cfg.dtype);
  }
}

// Forward with weight + bias (LayerNorm path).
TEST_F(LayernormTest, ForwardWithBias) {
  struct Config {
    DataType dtype;
    size_t token_num;
    size_t hidden_size;
  };
  Config configs[] = {{TYPE_FP16, 4, 256}, {TYPE_BF16, 4, 256}, {TYPE_FP32, 4, 256}};

  for (auto& cfg : configs) {
    const std::string weight_name = "test.layernorm.weight";
    const std::string bias_name = "test.layernorm.bias";
    auto fake_weight = std::make_shared<FakeWeight>(0);
    fake_weight->CreateWeight(weight_name, {cfg.hidden_size}, cfg.dtype, MemoryLocation::LOCATION_DEVICE);
    fake_weight->CreateWeight(bias_name, {cfg.hidden_size}, cfg.dtype, MemoryLocation::LOCATION_DEVICE);

    auto ctx = CreateCtx(fake_weight, cfg.dtype);
    Layernorm layernorm(weight_name, model_config.layernorm_eps, ctx, bias_name);

    Tensor input(MemoryLocation::LOCATION_DEVICE, cfg.dtype, {cfg.token_num, cfg.hidden_size});
    FillInputData(input);
    Tensor output(MemoryLocation::LOCATION_DEVICE, cfg.dtype, {cfg.token_num, cfg.hidden_size});

    EXPECT_TRUE(layernorm.Forward(input, output).OK());
    VerifyOutput(output, cfg.dtype);
  }
}

// Forward without weight (per-head RMSNorm). Kernel only supports head_dim=512.
TEST_F(LayernormTest, ForwardPerHeadRMSNorm) {
  struct Config {
    DataType dtype;
    size_t token_num;
    uint32_t num_heads;
  };
  Config configs[] = {{TYPE_FP16, 4, 4}, {TYPE_BF16, 4, 8}, {TYPE_FP16, 1, 4}};
  const uint32_t head_dim = 512;

  for (auto& cfg : configs) {
    auto fake_weight = std::make_shared<FakeWeight>(0);
    auto ctx = CreateCtx(fake_weight, cfg.dtype);
    Layernorm layernorm(model_config.layernorm_eps, cfg.num_heads, head_dim, ctx);

    size_t hidden_size = cfg.num_heads * head_dim;
    Tensor input(MemoryLocation::LOCATION_DEVICE, cfg.dtype, {cfg.token_num, hidden_size});
    FillInputData(input);
    Tensor output(MemoryLocation::LOCATION_DEVICE, cfg.dtype, {cfg.token_num, hidden_size});

    EXPECT_TRUE(layernorm.Forward(input, output).OK());
    VerifyOutput(output, cfg.dtype);
  }
}

// ForwardParallel: dual weight parallel norm, with and without stride.
TEST_F(LayernormTest, ForwardParallel) {
  struct Config {
    DataType dtype;
    size_t token_num;
    size_t hidden_size;
    int32_t stride;
  };
  Config configs[] = {
      {TYPE_FP16, 4, 256, 0},
      {TYPE_BF16, 4, 256, 0},
      {TYPE_FP16, 1, 256, 0},
      {TYPE_FP16, 4, 128, 256},  // stride > hidden_size
  };

  for (auto& cfg : configs) {
    const std::string weight_name = "test.input_layernorm.weight";
    const std::string weight2_name = "test.post_attention_layernorm.weight";
    auto fake_weight = std::make_shared<FakeWeight>(0);
    fake_weight->CreateWeight(weight_name, {cfg.hidden_size}, cfg.dtype, MemoryLocation::LOCATION_DEVICE);
    fake_weight->CreateWeight(weight2_name, {cfg.hidden_size}, cfg.dtype, MemoryLocation::LOCATION_DEVICE);

    auto ctx = CreateCtx(fake_weight, cfg.dtype);
    Layernorm layernorm(weight_name, weight2_name, model_config.layernorm_eps, ctx, cfg.stride);

    // Allocate with stride-sized rows when stride > 0.
    size_t alloc_cols = (cfg.stride > 0) ? static_cast<size_t>(cfg.stride) : cfg.hidden_size;
    Tensor input1(MemoryLocation::LOCATION_DEVICE, cfg.dtype, {cfg.token_num, alloc_cols});
    Tensor input2(MemoryLocation::LOCATION_DEVICE, cfg.dtype, {cfg.token_num, alloc_cols});
    FillInputData(input1);
    FillInputData(input2);
    input1.shape = {cfg.token_num, cfg.hidden_size};
    input2.shape = {cfg.token_num, cfg.hidden_size};

    Tensor output1(MemoryLocation::LOCATION_DEVICE, cfg.dtype, {cfg.token_num, alloc_cols});
    Tensor output2(MemoryLocation::LOCATION_DEVICE, cfg.dtype, {cfg.token_num, alloc_cols});

    EXPECT_TRUE(layernorm.ForwardParallel(input1, input2, output1, output2).OK());

    // Output shape should always be 2D regardless of stride.
    EXPECT_EQ(output1.shape.size(), 2ul);
    EXPECT_EQ(output1.shape[1], cfg.hidden_size);
    VerifyOutput(output1, cfg.dtype);
  }
}
