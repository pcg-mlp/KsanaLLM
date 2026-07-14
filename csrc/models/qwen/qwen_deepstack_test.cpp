/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cstring>
#include <filesystem>
#include <vector>

#ifdef ENABLE_CUDA
#  include "device/nvidia/nccl_utils.h"
#endif
#include "configure/environment.h"
#include "configure/global_context.h"
#include "models/base/model_input.h"
#include "models/qwen/qwen_model.h"
#include "modules/basic/add.h"
#include "tests/test.h"
#include "utils/singleton.h"

namespace ksana_llm {

class QwenDeepstackTest : public testing::Test {
 protected:
  void SetUp() override {
    // Parse config to initialise Environment / Context.
    const auto& env = Singleton<Environment>::GetInstance();
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();
    env->ParseConfig(config_path);

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

    SetDevice(0);

    // Build a minimal LayerCreationContext for the Add module.
    runtime_config_.inter_data_type = TYPE_FP32;
    creation_ctx_.context = context_;
    creation_ctx_.runtime_config = runtime_config_;
    creation_ctx_.rank = 0;
  }

  void TearDown() override { DeviceSynchronize(); }

  // Helper: create a GPU Tensor filled with a constant float value.
  Tensor MakeGpuTensor(const std::vector<size_t>& shape, float fill_value) {
    size_t numel = 1;
    for (auto s : shape) numel *= s;
    std::vector<float> host_data(numel, fill_value);
    Tensor gpu_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, shape);
    Memcpy(gpu_tensor.GetPtr<void>(), host_data.data(), numel * sizeof(float), MEMCPY_HOST_TO_DEVICE);
    return gpu_tensor;
  }

  // Helper: read GPU Tensor back to host floats.
  std::vector<float> ReadGpuTensor(const Tensor& tensor) {
    size_t numel = tensor.GetElementNumber();
    std::vector<float> host_data(numel);
    Memcpy(host_data.data(), tensor.GetPtr<void>(), numel * sizeof(float), MEMCPY_DEVICE_TO_HOST);
    return host_data;
  }

 protected:
  std::shared_ptr<Context> context_;
  RuntimeConfig runtime_config_;
  LayerCreationContext creation_ctx_;
};

// ---- Test 1: layer_idx not in map → residual unchanged ----
TEST_F(QwenDeepstackTest, InjectDeepstackSkipsUnmappedLayer) {
  constexpr size_t kSeqLen = 4;
  constexpr size_t kHidden = 8;

  // Prepare Qwen instance with Add module.
  Qwen qwen;
  qwen.deepstack_add_ = std::make_shared<Add>(creation_ctx_);

  // Residual buffer filled with 1.0.
  Tensor residual = MakeGpuTensor({kSeqLen, kHidden}, 1.0f);
  std::vector<Tensor> residual_buffer = {residual};

  // DeepstackData: map contains layer 5, but we query layer 99 → skip.
  ModelInput::DeepstackData deepstack;
  deepstack.active = true;
  deepstack.num_layers = 1;
  deepstack.layer_to_deepstack_idx[5] = 0;
  deepstack.layer_embeds.push_back(MakeGpuTensor({kSeqLen, kHidden}, 2.0f));
  ModelInput::DeepstackData::Segment seg{0, 0, static_cast<int>(kSeqLen)};
  deepstack.segments.push_back(seg);

  // Inject at layer 99 (not in map).
  Status status = qwen.InjectDeepstack(residual_buffer, deepstack, 99);
  EXPECT_TRUE(status.OK());

  // Residual should be unchanged (all 1.0).
  auto data = ReadGpuTensor(residual_buffer[0]);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(data[i], 1.0f) << "index " << i;
  }
}

// ---- Test 2: single segment → residual at injection position += embed ----
TEST_F(QwenDeepstackTest, InjectDeepstackSingleSegment) {
  // Layout: seq_len=6, hidden=4.  Segment at batch_pos=2, length=2.
  constexpr size_t kSeqLen = 6;
  constexpr size_t kHidden = 4;
  constexpr int kBatchPos = 2;
  constexpr int kLength = 2;
  constexpr int kEmbedOffset = 0;

  Qwen qwen;
  qwen.deepstack_add_ = std::make_shared<Add>(creation_ctx_);

  // Residual: all 1.0.
  Tensor residual = MakeGpuTensor({kSeqLen, kHidden}, 1.0f);
  std::vector<Tensor> residual_buffer = {residual};

  // Embed tensor: 2 visual tokens × hidden, filled with 3.0.
  Tensor embed = MakeGpuTensor({static_cast<size_t>(kLength), kHidden}, 3.0f);

  ModelInput::DeepstackData deepstack;
  deepstack.active = true;
  deepstack.num_layers = 1;
  deepstack.layer_to_deepstack_idx[0] = 0;
  deepstack.layer_embeds.push_back(embed);
  deepstack.segments.push_back({kBatchPos, kEmbedOffset, kLength});

  Status status = qwen.InjectDeepstack(residual_buffer, deepstack, 0);
  EXPECT_TRUE(status.OK());
  DeviceSynchronize();

  auto data = ReadGpuTensor(residual_buffer[0]);

  for (size_t row = 0; row < kSeqLen; ++row) {
    for (size_t col = 0; col < kHidden; ++col) {
      float expected = 1.0f;
      if (row >= static_cast<size_t>(kBatchPos) && row < static_cast<size_t>(kBatchPos + kLength)) {
        expected = 1.0f + 3.0f;  // residual + embed
      }
      EXPECT_FLOAT_EQ(data[row * kHidden + col], expected) << "row=" << row << " col=" << col;
    }
  }
}

// ---- Test 3: multiple segments → each segment's position is correctly updated ----
TEST_F(QwenDeepstackTest, InjectDeepstackMultipleSegments) {
  // Layout: seq_len=10, hidden=4.
  // Segment 0: batch_pos=1, length=2, embed_offset=0.
  // Segment 1: batch_pos=6, length=3, embed_offset=2.
  constexpr size_t kSeqLen = 10;
  constexpr size_t kHidden = 4;
  constexpr int kTotalVisual = 5;  // 2 + 3

  Qwen qwen;
  qwen.deepstack_add_ = std::make_shared<Add>(creation_ctx_);

  Tensor residual = MakeGpuTensor({kSeqLen, kHidden}, 1.0f);
  std::vector<Tensor> residual_buffer = {residual};

  // Embed: 5 visual tokens with value 5.0.
  Tensor embed = MakeGpuTensor({static_cast<size_t>(kTotalVisual), kHidden}, 5.0f);

  ModelInput::DeepstackData deepstack;
  deepstack.active = true;
  deepstack.num_layers = 1;
  deepstack.layer_to_deepstack_idx[2] = 0;
  deepstack.layer_embeds.push_back(embed);
  deepstack.segments.push_back({1, 0, 2});  // batch_pos=1, embed_offset=0, length=2
  deepstack.segments.push_back({6, 2, 3});  // batch_pos=6, embed_offset=2, length=3

  Status status = qwen.InjectDeepstack(residual_buffer, deepstack, 2);
  EXPECT_TRUE(status.OK());
  DeviceSynchronize();

  auto data = ReadGpuTensor(residual_buffer[0]);

  // Build expected: rows 1-2 and 6-8 should be 1.0+5.0=6.0, rest stays 1.0.
  for (size_t row = 0; row < kSeqLen; ++row) {
    bool is_injected = (row >= 1 && row < 3) || (row >= 6 && row < 9);
    float expected = is_injected ? 6.0f : 1.0f;
    for (size_t col = 0; col < kHidden; ++col) {
      EXPECT_FLOAT_EQ(data[row * kHidden + col], expected) << "row=" << row << " col=" << col;
    }
  }
}

}  // namespace ksana_llm
