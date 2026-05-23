/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

// Unit tests for TopKLayer (smoke tests).
//
// Tests exercise Init and ForwardRadix paths for k=512 and k=1024.
// Tensors are allocated on GPU, the layer is initialised, and Forward()
// is called to confirm the kernel launches without error.

#include <cuda_runtime.h>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "device/common_device.h"
#include "device/device_types.h"
#include "device/nvidia/cuda_utils.h"
#include "layers/topk_layer.h"
#include "test.h"
#include "utils/singleton.h"

namespace ksana_llm {

// ===========================================================================
// Test fixture
// ===========================================================================
class TopKLayerTest : public testing::Test {
 protected:
  void SetUp() override {
    BlockManagerConfig block_manager_config;
    block_manager_config.host_allocator_config.blocks_num = 2;
    block_manager_config.host_allocator_config.block_token_num = 16;
    block_manager_config.host_allocator_config.block_size = 1024 * 1024;
    block_manager_config.host_allocator_config.device = MEMORY_HOST;
    block_manager_config.device_allocator_config.blocks_num = 2;
    block_manager_config.device_allocator_config.block_token_num = 16;
    block_manager_config.device_allocator_config.block_size = 1024 * 1024;
    block_manager_config.device_allocator_config.device = MEMORY_DEVICE;

    runtime_config_.inter_data_type = DataType::TYPE_FP32;
    runtime_config_.parallel_basic_config.tensor_parallel_size = 1;
    runtime_config_.max_batch_size = 8;
    runtime_config_.max_seq_len = 4096;

    const auto& env = Singleton<Environment>::GetInstance();
    env->SetBlockManagerConfig(block_manager_config);

    context_ = std::make_shared<Context>();
    context_->global.reset(new GlobalContext(1, 1, 1, env));
    context_->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context_->device = std::make_shared<DeviceContext>(1);
    context_->device->ext->InitNcclParam(nccl_uid);
  }

  void TearDown() override {}

  // Allocate a device tensor and zero-fill it.
  Tensor MakeDeviceTensor(DataType dtype, const std::vector<size_t>& shape) {
    Tensor t(MemoryLocation::LOCATION_DEVICE, dtype, shape);
    CUDA_CHECK(cudaMemset(t.GetPtr<void>(), 0, t.GetTotalBytes()));
    return t;
  }

  // Build Init parameters: [k, page_size]
  std::vector<std::any> MakeInitParams(uint32_t k, uint32_t page_size = 16) {
    return {std::any(k), std::any(page_size)};
  }

  // ---------------------------------------------------------------------------
  // Build the radix-path input tensors matching ForwardRadix expectations:
  //   [0] scores      [num_works, score_stride]       float
  //   [1] row_starts  [num_works]                     int32_t
  //   [2] row_ends    [num_works]                     int32_t
  //   [3] page_table  [num_batches, page_table_stride] int32_t
  // ---------------------------------------------------------------------------
  std::vector<Tensor> MakeRadixInputs(uint32_t num_batches, uint32_t seq_len, uint32_t page_size) {
    // For simplicity: num_works == num_batches (one work item per batch).
    const uint32_t num_works = num_batches;
    const uint32_t score_stride = seq_len;
    const uint32_t page_table_stride = (seq_len + page_size - 1) / page_size;

    std::vector<Tensor> inputs;

    // [0] scores
    inputs.push_back(MakeDeviceTensor(TYPE_FP32, {num_works, score_stride}));

    // [1] row_starts: all zeros (each row starts at 0)
    inputs.push_back(MakeDeviceTensor(TYPE_INT32, {num_works}));

    // [2] row_ends: each row ends at seq_len
    Tensor row_ends_t = MakeDeviceTensor(TYPE_INT32, {num_works});
    {
      std::vector<int32_t> ends_h(num_works, static_cast<int32_t>(seq_len));
      CUDA_CHECK(cudaMemcpy(row_ends_t.GetPtr<void>(), ends_h.data(),
                            num_works * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    inputs.push_back(row_ends_t);

    // [3] page_table (identity mapping)
    Tensor pt_t = MakeDeviceTensor(TYPE_INT32, {num_batches, page_table_stride});
    {
      std::vector<int32_t> pt_h(num_batches * page_table_stride);
      std::iota(pt_h.begin(), pt_h.end(), 0);
      CUDA_CHECK(cudaMemcpy(pt_t.GetPtr<void>(), pt_h.data(),
                            pt_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    inputs.push_back(pt_t);

    return inputs;
  }

 protected:
  std::shared_ptr<Context> context_;
  RuntimeConfig runtime_config_;
};

// ===========================================================================
// Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Init tests
// ---------------------------------------------------------------------------
TEST_F(TopKLayerTest, InitK512) {
#ifndef ENABLE_CUDA
  GTEST_SKIP();
#endif
  TopKLayer layer;
  EXPECT_TRUE(layer.Init(MakeInitParams(512u), runtime_config_, context_, 0).OK());
}

TEST_F(TopKLayerTest, InitK1024) {
#ifndef ENABLE_CUDA
  GTEST_SKIP();
#endif
  TopKLayer layer;
  EXPECT_TRUE(layer.Init(MakeInitParams(1024u), runtime_config_, context_, 0).OK());
}

// ---------------------------------------------------------------------------
// Forward – k=512
// ---------------------------------------------------------------------------
TEST_F(TopKLayerTest, ForwardK512) {
#ifndef ENABLE_CUDA
  GTEST_SKIP();
#endif
  constexpr uint32_t num_batches = 2;
  constexpr uint32_t seq_len = 2048;
  constexpr uint32_t page_size = 16;
  constexpr uint32_t k = 512;

  TopKLayer layer;
  ASSERT_TRUE(layer.Init(MakeInitParams(k, page_size), runtime_config_, context_, 0).OK());

  auto inputs = MakeRadixInputs(num_batches, seq_len, page_size);

  std::vector<Tensor> outputs;
  outputs.push_back(MakeDeviceTensor(TYPE_INT32, {num_batches, k}));

  EXPECT_TRUE(layer.Forward(inputs, outputs).OK());
  CUDA_CHECK(cudaStreamSynchronize(context_->device->GetPrimaryComputeStream().Get()));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// ---------------------------------------------------------------------------
// Forward – k=1024
// ---------------------------------------------------------------------------
TEST_F(TopKLayerTest, ForwardK1024) {
#ifndef ENABLE_CUDA
  GTEST_SKIP();
#endif
  constexpr uint32_t num_batches = 2;
  constexpr uint32_t seq_len = 4096;
  constexpr uint32_t page_size = 16;
  constexpr uint32_t k = 1024;

  TopKLayer layer;
  ASSERT_TRUE(layer.Init(MakeInitParams(k, page_size), runtime_config_, context_, 0).OK());

  auto inputs = MakeRadixInputs(num_batches, seq_len, page_size);

  std::vector<Tensor> outputs;
  outputs.push_back(MakeDeviceTensor(TYPE_INT32, {num_batches, k}));

  EXPECT_TRUE(layer.Forward(inputs, outputs).OK());
  CUDA_CHECK(cudaStreamSynchronize(context_->device->GetPrimaryComputeStream().Get()));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// ---------------------------------------------------------------------------
// Init with non-power-of-two page_size must throw.
// ---------------------------------------------------------------------------
TEST_F(TopKLayerTest, InvalidPageSizeThrows) {
#ifndef ENABLE_CUDA
  GTEST_SKIP();
#endif
  TopKLayer layer;
  EXPECT_THROW(layer.Init(MakeInitParams(512u, /*page_size=*/12u), runtime_config_, context_, 0),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// Init with unsupported k must throw.
// ---------------------------------------------------------------------------
TEST_F(TopKLayerTest, InvalidKThrows) {
#ifndef ENABLE_CUDA
  GTEST_SKIP();
#endif
  TopKLayer layer;
  EXPECT_THROW(layer.Init(MakeInitParams(/*k=*/256u), runtime_config_, context_, 0),
               std::runtime_error);
}

}  // namespace ksana_llm
