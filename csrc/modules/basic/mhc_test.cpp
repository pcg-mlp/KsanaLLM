/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/mhc.h"
#include "models/common/common_weight.h"
#include "modules/basic/layernorm.h"
#include "tests/test.h"
#include "utils/random_generator.h"

using namespace ksana_llm;

template <typename T>
class MhcWeight : public CommonWeight<T> {
  using CommonWeight<T>::weights_map_;
  using CommonWeight<T>::context_;

 public:
  MhcWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
            std::shared_ptr<Context> context)
      : CommonWeight<T>(model_config, runtime_config, rank, context) {}

  ~MhcWeight() override = default;

  void AddWeight() {
    const size_t hidden_size = 7168;
    std::unordered_map<std::string, std::vector<size_t>> add_tensor_map;
    // mhc weight
    add_tensor_map["model.layers.0.hc_attn_fn"] = {24, hidden_size * 4};
    add_tensor_map["model.layers.0.hc_attn_scale"] = {3};
    add_tensor_map["model.layers.0.hc_attn_base"] = {24};
    add_tensor_map["hc_head_fn"] = {8 /*pad from 4 to 8*/, hidden_size * 4};
    add_tensor_map["hc_head_scale"] = {1};
    add_tensor_map["hc_head_base"] = {4};
    for (const auto& [tensor_name, shape] : add_tensor_map) {
      Tensor tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, shape);
      weights_map_.emplace(tensor_name, tensor);
      std::vector<float> input_data(tensor.GetElementNumber());
      for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = 1.0f / (i % 97 * 0.1f + 1.0f) * pow(-1, (i % 7));
      }
      Memcpy(tensor.template GetPtr<void>(), input_data.data(), tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE);
    }
    // norm weight for fused test (bf16)
    Tensor tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_BF16, {hidden_size});
    weights_map_.emplace("model.layers.0.input_layernorm.weight", tensor);
    std::vector<bfloat16> norm_data(hidden_size);
    for (size_t i = 0; i < hidden_size; i++) {
      norm_data[i] = static_cast<bfloat16>(1.0f / (i % 37 * 0.05f + 0.5f));
    }
    Memcpy(tensor.template GetPtr<void>(), norm_data.data(), tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE);
  }
};

class MhcTest : public testing::Test {
 protected:
  void SetUp() override {
    // Initialize the relevant
    model_config.hidden_units = 7168;
    model_config.layernorm_eps = 1e-6;
    model_config.mhc_config.hc_mult = 4;
    model_config.mhc_config.hc_sinkhorn_iters = 20;
    model_config.mhc_config.hc_eps = 1e-6;

    runtime_config.max_step_token_num = 10;

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
  std::shared_ptr<Context> context;

  template <typename T>
  void TestMhcForward(const int rank = 0) {
    static_assert(std::is_same_v<T, bfloat16>, "Only support bfloat16");

    SetDevice(rank);

    // Prepare model weight
    auto mhc_weight = std::make_shared<MhcWeight<T>>(model_config, runtime_config, rank, context);
    mhc_weight->AddWeight();

    BufferManager buffer_mgr;
    buffer_mgr.SetRank(rank);

    LayerCreationContext creation_ctx;
    creation_ctx.Init(mhc_weight, context, rank, pipeline_config, model_config, runtime_config, &buffer_mgr);

    // Allocate buffers
    MhcBuffers mhc_buffers;
    Mhc::CreateBuffers(&buffer_mgr, model_config.mhc_config, runtime_config, mhc_buffers);

    // Prepare test data
    const size_t token_num = 6;
    const size_t hidden_size = model_config.hidden_units;
    Tensor input(MemoryLocation::LOCATION_DEVICE, model_config.weight_data_type, {token_num, hidden_size});
    std::vector<T> input_host(token_num * hidden_size);
    static RandomGenerator<float> gen(42);
    for (auto& val : input_host) {
      val = static_cast<T>(gen());
    }
    MemcpyAsync(input.GetPtr<void>(), input_host.data(), input.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context->device->GetPrimaryComputeStream());

    const size_t hc_mult = model_config.mhc_config.hc_mult;
    std::vector<Tensor> output{
        Tensor(MemoryLocation::LOCATION_DEVICE, model_config.weight_data_type, {token_num, hc_mult * hidden_size})};

    // Test initial forward
    Mhc initial_mhc("", creation_ctx, mhc_buffers);
    EXPECT_TRUE(initial_mhc.InitialForward({input}, output).OK());
    EXPECT_EQ(output[0].shape[0], token_num);
    EXPECT_EQ(output[0].shape[1], hc_mult * hidden_size);
    std::vector<T> output_host(output[0].GetElementNumber());
    MemcpyAsync(output_host.data(), output[0].GetPtr<void>(), output[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context->device->GetPrimaryComputeStream());
    StreamSynchronize(context->device->GetPrimaryComputeStream());
    // Verify some results
    std::vector<float> output_ref{0.753906, 0.640625, 0.753906, 0.136719, 0.902344};
    for (size_t i = 0; i < output_ref.size(); i++) {
      EXPECT_NEAR(output_ref[i], static_cast<float>(output_host[i]), 1e-3);
    }
    std::swap(output[0], input);

    // Test pre/post forward
    Mhc input_mhc("model.layers.0.hc_attn", creation_ctx, mhc_buffers);
    EXPECT_TRUE(input_mhc.PreForward({input}, output).OK());
    EXPECT_EQ(output[0].shape[0], token_num);
    EXPECT_EQ(output[0].shape[1], hidden_size);
    output_host.resize(output[0].GetElementNumber());
    MemcpyAsync(output_host.data(), output[0].GetPtr<void>(), output[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context->device->GetPrimaryComputeStream());
    StreamSynchronize(context->device->GetPrimaryComputeStream());
    // Verify some results
    output_ref = {3.015625, 2.5625, 3.015625, 0.546875, 3.609375};
    for (size_t i = 0; i < output_ref.size(); i++) {
      EXPECT_NEAR(output_ref[i], static_cast<float>(output_host[i]), 1e-3);
    }
    std::swap(output[0], input);

    EXPECT_TRUE(input_mhc.PostForward({input}, output).OK());
    EXPECT_EQ(output[0].shape[0], token_num);
    EXPECT_EQ(output[0].shape[1], hc_mult * hidden_size);
    output_host.resize(output[0].GetElementNumber());
    MemcpyAsync(output_host.data(), output[0].GetPtr<void>(), output[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context->device->GetPrimaryComputeStream());
    StreamSynchronize(context->device->GetPrimaryComputeStream());
    // Verify some results
    output_ref = {0.75390625, 0.640625, 0.75390625, 0.13671875, 0.90234375};
    for (size_t i = 0; i < output_ref.size(); i++) {
      EXPECT_NEAR(output_ref[i], static_cast<float>(output_host[i]), 1e-3);
    }
    std::swap(output[0], input);

    // Test final forward
    Mhc final_mhc("hc_head", creation_ctx, mhc_buffers);
    EXPECT_TRUE(final_mhc.FinalForward({input}, output).OK());
    EXPECT_EQ(output[0].shape[0], token_num);
    EXPECT_EQ(output[0].shape[1], hidden_size);
    output_host.resize(output[0].GetElementNumber());
    MemcpyAsync(output_host.data(), output[0].GetPtr<void>(), output[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context->device->GetPrimaryComputeStream());
    StreamSynchronize(context->device->GetPrimaryComputeStream());
    output_ref = {3.015625, 2.5625, 3.015625, 0.546875, 3.609375};
    for (size_t i = 0; i < output_ref.size(); i++) {
      EXPECT_NEAR(output_ref[i], static_cast<float>(output_host[i]), 1e-3);
    }
  }

  // Test that PreForwardWithNorm produces the same result as PreForward + separate RMSNorm.
  template <typename T>
  void TestMhcPreForwardWithNorm(const int rank = 0) {
    static_assert(std::is_same_v<T, bfloat16>, "Only support bfloat16");

    SetDevice(rank);

    auto mhc_weight = std::make_shared<MhcWeight<T>>(model_config, runtime_config, rank, context);
    mhc_weight->AddWeight();

    BufferManager buffer_mgr;
    buffer_mgr.SetRank(rank);

    LayerCreationContext creation_ctx;
    creation_ctx.Init(mhc_weight, context, rank, pipeline_config, model_config, runtime_config, &buffer_mgr);

    MhcBuffers mhc_buffers;
    Mhc::CreateBuffers(&buffer_mgr, model_config.mhc_config, runtime_config, mhc_buffers);

    const size_t token_num = 6;
    const size_t hidden_size = model_config.hidden_units;
    const size_t hc_mult = model_config.mhc_config.hc_mult;
    const auto dtype = model_config.weight_data_type;
    const auto& stream = context->device->GetPrimaryComputeStream();

    // Prepare input [t, hc * d]
    Tensor input(MemoryLocation::LOCATION_DEVICE, dtype, {token_num, hc_mult * hidden_size});
    std::vector<T> input_host(token_num * hc_mult * hidden_size);
    static RandomGenerator<float> gen(123);
    for (auto& val : input_host) {
      val = static_cast<T>(gen());
    }
    MemcpyAsync(input.GetPtr<void>(), input_host.data(), input.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE, stream);
    StreamSynchronize(stream);

    // Helper: clone input and allocate output
    void* input_ptr = input.GetPtr<void>();
    auto make_path = [&]() -> std::pair<Tensor, std::vector<Tensor>> {
      Tensor in_copy(MemoryLocation::LOCATION_DEVICE, dtype, {token_num, hc_mult * hidden_size});
      MemcpyAsync(in_copy.GetPtr<void>(), input_ptr, input.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE, stream);
      std::vector<Tensor> out{Tensor(MemoryLocation::LOCATION_DEVICE, dtype, {token_num, hidden_size})};
      return {std::move(in_copy), std::move(out)};
    };

    // Path A: PreForward + separate RMSNorm (reference)
    auto [input_a, output_a] = make_path();
    Mhc mhc_a("model.layers.0.hc_attn", creation_ctx, mhc_buffers);
    EXPECT_TRUE(mhc_a.PreForward({input_a}, output_a).OK());
    Layernorm norm("model.layers.0.input_layernorm.weight", model_config.layernorm_eps, creation_ctx);
    EXPECT_TRUE(norm.Forward(output_a[0], output_a[0]).OK());

    // Path B: Fused PreForwardWithNorm
    auto [input_b, output_b] = make_path();
    Mhc mhc_b("model.layers.0.hc_attn", creation_ctx, mhc_buffers, "model.layers.0.input_layernorm.weight");
    EXPECT_TRUE(mhc_b.PreForwardWithNorm({input_b}, output_b).OK());

    StreamSynchronize(stream);

    // Compare
    EXPECT_EQ(output_a[0].shape, output_b[0].shape);
    const size_t num_elements = output_a[0].GetElementNumber();
    std::vector<T> result_a(num_elements);
    std::vector<T> result_b(num_elements);
    void* ptr_a = output_a[0].template GetPtr<void>();
    void* ptr_b = output_b[0].template GetPtr<void>();
    MemcpyAsync(result_a.data(), ptr_a, output_a[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST, stream);
    MemcpyAsync(result_b.data(), ptr_b, output_b[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST, stream);
    StreamSynchronize(stream);

    // Fused path computes sumsq in float32 before bf16 truncation, so it's slightly
    // more accurate than the separate path. Allow a handful of 1-ULP bf16 differences.
    constexpr float kAtol = 5e-3f;
    constexpr float kRtol = 5e-3f;
    size_t mismatch_count = 0;
    for (size_t i = 0; i < result_a.size(); i++) {
      float a = static_cast<float>(result_a[i]);
      float b = static_cast<float>(result_b[i]);
      if (std::abs(a - b) > kAtol + kRtol * std::abs(a)) {
        if (mismatch_count < 5) {
          KLLM_LOG_WARNING << fmt::format("Mismatch at index {}: ref={}, fused={}", i, a, b);
        }
        mismatch_count++;
      }
    }
    EXPECT_LE(mismatch_count, result_a.size() / 1000) << mismatch_count << " values exceed tolerance (expect <0.1%)";
  }
};

TEST_F(MhcTest, ForwardBf16Test) {
  // Set data type
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = TYPE_BF16;
  auto env = Singleton<Environment>::GetInstance();
  env->schedule_config_parser_.runtime_config_ = runtime_config;

  TestMhcForward<bfloat16>();
}

TEST_F(MhcTest, PreForwardWithNormBf16Test) {
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = TYPE_BF16;
  auto env = Singleton<Environment>::GetInstance();
  env->schedule_config_parser_.runtime_config_ = runtime_config;

  TestMhcPreForwardWithNorm<bfloat16>();
}
