/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/common_model_weight_loader.h"

#include <memory>
#include <random>
#include <vector>

#include "3rdparty/half/include/half.hpp"
#include "configure/environment.h"
#include "configure/global_context.h"
#include "device/common_device.h"
#include "device/device_context.h"
#include "tests/test.h"
#include "utils/singleton.h"

#ifdef ENABLE_CUDA
#  include "device/nvidia/nccl_utils.h"
#endif

namespace ksana_llm {

class CommonModelWeightLoaderTest : public testing::Test {
 protected:
  void SetUp() override {
    model_config_ = std::make_shared<BaseModelConfig>();
    model_config_->model_dir = "/model/test";

    BlockManagerConfig block_manager_config;
    block_manager_config.host_allocator_config.blocks_num = 2;
    block_manager_config.host_allocator_config.block_token_num = 16;
    block_manager_config.host_allocator_config.block_size = 1024;
    block_manager_config.host_allocator_config.device = MEMORY_HOST;
    block_manager_config.device_allocator_config.blocks_num = 2;
    block_manager_config.device_allocator_config.block_token_num = 16;
    block_manager_config.device_allocator_config.block_size = 1024;
    block_manager_config.device_allocator_config.device = MEMORY_DEVICE;

    env_ = Singleton<Environment>::GetInstance();
    env_->SetBlockManagerConfig(block_manager_config);

    // 创建 Context，参考 deepseek_v3_model_test.cpp
    context_ = std::make_shared<Context>();
    context_->global.reset(new GlobalContext(1, 1, 1, env_));
    context_->global->InitDistributedConfig();

#ifdef ENABLE_CUDA
    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context_->device = std::make_shared<DeviceContext>(1);
    context_->device->ext->InitNcclParam(nccl_uid);
#endif

#ifdef ENABLE_KUNLUN
    context_->device = std::make_shared<DeviceContext>(1);
#endif
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<BaseModelConfig> model_config_;
  std::shared_ptr<Environment> env_;
  std::shared_ptr<Context> context_;
};

TEST_F(CommonModelWeightLoaderTest, Permute2DTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = uint8_t;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: 1024x2048 张量转置
  {
    const size_t rows = 1024;
    const size_t cols = 2048;

    // 创建输入张量
    Tensor input_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {rows, cols});

    // 准备测试数据
    std::vector<dtype> input_host(rows * cols);
    std::default_random_engine eng;
    std::uniform_int_distribution<int> random_range(0, 255);
    for (size_t i = 0; i < rows * cols; ++i) {
      input_host[i] = static_cast<dtype>(random_range(eng));
    }

    // 拷贝到设备
    MemcpyAsync(input_tensor.GetPtr<void>(), input_host.data(), input_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 执行 Permute2D
    Status status = weight_loader.Permute2D(input_tensor, kDeviceRank);
    EXPECT_TRUE(status.OK());

    // 同步 stream
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证形状已经转置
    EXPECT_EQ(input_tensor.shape[0], cols);
    EXPECT_EQ(input_tensor.shape[1], rows);

    // 验证数据正确性
    std::vector<dtype> output_host(rows * cols);
    MemcpyAsync(output_host.data(), input_tensor.GetPtr<void>(), input_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context_->device->GetMemoryManageStream());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 完全遍历验证转置结果：output[j][i] 应该等于 input[i][j]
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        size_t original_idx = i * cols + j;
        size_t transposed_idx = j * rows + i;
        EXPECT_EQ(output_host[transposed_idx], input_host[original_idx]);
      }
    }
  }

  // 测试用例 2: 错误情况 - 非2D张量应该抛出异常
  {
    // 1D 张量
    Tensor tensor_1d(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {100});
    EXPECT_THROW(weight_loader.Permute2D(tensor_1d, kDeviceRank), std::exception);

    // 3D 张量
    Tensor tensor_3d(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {10, 20, 30});
    EXPECT_THROW(weight_loader.Permute2D(tensor_3d, kDeviceRank), std::exception);
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, GetTorchTensorFromTensorTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: HOST 上的 FP32 张量
  {
    const size_t rows = 10;
    const size_t cols = 20;

    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {rows, cols});
    torch::Tensor torch_tensor = weight_loader.GetTorchTensorFromTensor(host_tensor);

    EXPECT_TRUE(torch_tensor.device().is_cpu());
    EXPECT_EQ(torch_tensor.scalar_type(), torch::kFloat32);
    EXPECT_EQ(torch_tensor.size(0), rows);
    EXPECT_EQ(torch_tensor.size(1), cols);
  }

  // 测试用例 2: DEVICE 上的 FP16 张量
  {
    const size_t rows = 8;
    const size_t cols = 16;

    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {rows, cols});
    torch::Tensor torch_tensor = weight_loader.GetTorchTensorFromTensor(device_tensor);

    EXPECT_TRUE(torch_tensor.device().is_cuda());
    EXPECT_EQ(torch_tensor.device().index(), kDeviceRank);
    EXPECT_EQ(torch_tensor.scalar_type(), torch::kFloat16);
    EXPECT_EQ(torch_tensor.size(0), rows);
    EXPECT_EQ(torch_tensor.size(1), cols);
  }

  // 测试用例 3: 1D 张量
  {
    const size_t size = 100;

    Tensor tensor_1d(MemoryLocation::LOCATION_HOST, TYPE_FP32, {size});
    torch::Tensor torch_tensor = weight_loader.GetTorchTensorFromTensor(tensor_1d);

    EXPECT_EQ(torch_tensor.dim(), 1);
    EXPECT_EQ(torch_tensor.size(0), size);
    EXPECT_TRUE(torch_tensor.device().is_cpu());
    EXPECT_EQ(torch_tensor.scalar_type(), torch::kFloat32);
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, GetTensorFromTorchTensorTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: CPU 上的 FP32 torch::Tensor
  {
    const size_t rows = 10;
    const size_t cols = 20;

    torch::Tensor torch_tensor = torch::randn({static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    Tensor tensor = weight_loader.GetTensorFromTorchTensor(torch_tensor);

    EXPECT_EQ(tensor.location, MemoryLocation::LOCATION_HOST);
    EXPECT_EQ(tensor.dtype, TYPE_FP32);
    EXPECT_EQ(tensor.shape.size(), 2);
    EXPECT_EQ(tensor.shape[0], rows);
    EXPECT_EQ(tensor.shape[1], cols);
  }

  // 测试用例 2: CUDA 上的 FP16 torch::Tensor
  {
    const size_t rows = 8;
    const size_t cols = 16;

    torch::Tensor torch_tensor =
        torch::randn({static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                     torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, kDeviceRank));
    Tensor tensor = weight_loader.GetTensorFromTorchTensor(torch_tensor);
    StreamSynchronize(context_->device->GetMemoryManageStream());

    EXPECT_EQ(tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(tensor.dtype, TYPE_FP16);
    EXPECT_EQ(tensor.shape.size(), 2);
    EXPECT_EQ(tensor.shape[0], rows);
    EXPECT_EQ(tensor.shape[1], cols);
  }

  // 测试用例 3: 1D torch::Tensor
  {
    const size_t size = 100;

    torch::Tensor torch_tensor =
        torch::randn({static_cast<int64_t>(size)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    Tensor tensor = weight_loader.GetTensorFromTorchTensor(torch_tensor);

    EXPECT_EQ(tensor.shape.size(), 1);
    EXPECT_EQ(tensor.shape[0], size);
    EXPECT_EQ(tensor.location, MemoryLocation::LOCATION_HOST);
    EXPECT_EQ(tensor.dtype, TYPE_FP32);
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, GetMemcpyKindTest) {
#ifdef ENABLE_CUDA
  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: HOST → HOST
  {
    Tensor host_tensor1(MemoryLocation::LOCATION_HOST, TYPE_FP32, {10, 20});
    Tensor host_tensor2(MemoryLocation::LOCATION_HOST, TYPE_FP32, {10, 20});
    MemcpyKind kind = weight_loader.GetMemcpyKind(host_tensor1, host_tensor2);
    EXPECT_EQ(kind, MEMCPY_HOST_TO_HOST);
  }

  // 测试用例 2: HOST → DEVICE
  {
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {10, 20});
    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {10, 20});
    MemcpyKind kind = weight_loader.GetMemcpyKind(host_tensor, device_tensor);
    EXPECT_EQ(kind, MEMCPY_HOST_TO_DEVICE);
  }

  // 测试用例 3: DEVICE → HOST
  {
    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {10, 20});
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {10, 20});
    MemcpyKind kind = weight_loader.GetMemcpyKind(device_tensor, host_tensor);
    EXPECT_EQ(kind, MEMCPY_DEVICE_TO_HOST);
  }

  // 测试用例 4: DEVICE → DEVICE
  {
    Tensor device_tensor1(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {10, 20});
    Tensor device_tensor2(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {10, 20});
    MemcpyKind kind = weight_loader.GetMemcpyKind(device_tensor1, device_tensor2);
    EXPECT_EQ(kind, MEMCPY_DEVICE_TO_DEVICE);
  }

  // 测试用例 5: 不同数据类型的张量（应该仍然正常工作，因为只检查 location）
  {
    Tensor host_fp32(MemoryLocation::LOCATION_HOST, TYPE_FP32, {10, 20});
    Tensor device_fp16(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {10, 20});
    MemcpyKind kind = weight_loader.GetMemcpyKind(host_fp32, device_fp16);
    EXPECT_EQ(kind, MEMCPY_HOST_TO_DEVICE);
  }

  // 测试用例 6: 不同形状的张量（应该仍然正常工作，因为只检查 location）
  {
    Tensor device_small(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {5, 10});
    Tensor host_large(MemoryLocation::LOCATION_HOST, TYPE_FP32, {100, 200});
    MemcpyKind kind = weight_loader.GetMemcpyKind(device_small, host_large);
    EXPECT_EQ(kind, MEMCPY_DEVICE_TO_HOST);
  }
#endif
}

TEST_F(CommonModelWeightLoaderTest, MoveToDeviceTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例: FP16 类型的 2D 张量
  {
    const size_t rows = 10;
    const size_t cols = 20;

    // 创建 HOST 张量并填充数据
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {rows, cols});
    std::vector<half_float::half> host_data(rows * cols);
    std::default_random_engine eng;
    std::uniform_real_distribution<float> random_range(0.0f, 1.0f);
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = half_float::half(random_range(eng));
    }
    std::memcpy(host_tensor.GetPtr<void>(), host_data.data(), host_tensor.GetTotalBytes());

    // 执行 MoveToDevice
    Tensor device_tensor = weight_loader.MoveToDevice(host_tensor, kDeviceRank);
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证输出张量的属性
    EXPECT_EQ(device_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(device_tensor.dtype, TYPE_FP16);
    EXPECT_EQ(device_tensor.shape.size(), 2);
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols);

    // 验证数据正确性：将 device 数据拷贝回 host 并比较
    std::vector<half_float::half> device_data(rows * cols);
    MemcpyAsync(device_data.data(), device_tensor.GetPtr<void>(), device_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context_->device->GetMemoryManageStream());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    for (size_t i = 0; i < rows * cols; ++i) {
      EXPECT_EQ(device_data[i], host_data[i]);
    }
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, CastDeviceTensorTypeTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: FP32 → FP16
  {
    const size_t rows = 10;
    const size_t cols = 20;

    // 创建 DEVICE 上的 FP32 张量并填充数据
    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {rows, cols});
    std::vector<float> host_data(rows * cols);
    std::default_random_engine eng(42);
    std::uniform_real_distribution<float> random_range(-1.0f, 1.0f);
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = random_range(eng);
    }
    MemcpyAsync(device_tensor.GetPtr<void>(), host_data.data(), device_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 执行类型转换
    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_FP16, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证输出张量的属性
    EXPECT_EQ(device_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(device_tensor.dtype, TYPE_FP16);
    EXPECT_EQ(device_tensor.shape.size(), 2);
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols);

    // 验证数据正确性：将转换后的数据拷贝回 host 并比较
    std::vector<half_float::half> result_data(rows * cols);
    MemcpyAsync(result_data.data(), device_tensor.GetPtr<void>(), device_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context_->device->GetMemoryManageStream());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    for (size_t i = 0; i < rows * cols; ++i) {
      float expected = host_data[i];
      float actual = static_cast<float>(result_data[i]);
      EXPECT_NEAR(actual, expected, 1e-3);  // FP16 精度损失
    }
  }

  // 测试用例 2: FP32 → BF16
  {
    const size_t rows = 8;
    const size_t cols = 16;

    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {rows, cols});
    std::vector<float> host_data(rows * cols);
    std::default_random_engine eng(123);
    std::uniform_real_distribution<float> random_range(-1.0f, 1.0f);
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = random_range(eng);
    }
    MemcpyAsync(device_tensor.GetPtr<void>(), host_data.data(), device_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_BF16, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    EXPECT_EQ(device_tensor.dtype, TYPE_BF16);
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols);
  }

  // 测试用例 3: FP16 → FP32
  {
    const size_t rows = 12;
    const size_t cols = 24;

    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {rows, cols});
    std::vector<half_float::half> host_data(rows * cols);
    std::default_random_engine eng(456);
    std::uniform_real_distribution<float> random_range(-1.0f, 1.0f);
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = half_float::half(random_range(eng));
    }
    MemcpyAsync(device_tensor.GetPtr<void>(), host_data.data(), device_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_FP32, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    EXPECT_EQ(device_tensor.dtype, TYPE_FP32);
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols);

    // 验证数据正确性
    std::vector<float> result_data(rows * cols);
    MemcpyAsync(result_data.data(), device_tensor.GetPtr<void>(), device_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context_->device->GetMemoryManageStream());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    for (size_t i = 0; i < rows * cols; ++i) {
      float expected = static_cast<float>(host_data[i]);
      float actual = result_data[i];
      EXPECT_NEAR(actual, expected, 1e-3);
    }
  }

  // 测试用例 4: BF16 → FP32
  {
    const size_t size = 100;

    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {size});

    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_FP32, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    EXPECT_EQ(device_tensor.dtype, TYPE_FP32);
    EXPECT_EQ(device_tensor.shape[0], size);
  }

  // 测试用例 5: FP16 → BF16 (原地转换)
  {
    const size_t rows = 16;
    const size_t cols = 32;

    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {rows, cols});
    void* original_ptr = device_tensor.GetPtr<void>();

    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_BF16, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    EXPECT_EQ(device_tensor.dtype, TYPE_BF16);
    EXPECT_EQ(device_tensor.GetPtr<void>(), original_ptr);  // 原地转换，指针不变
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols);
  }

  // 测试用例 6: BF16 → FP16 (原地转换)
  {
    const size_t rows = 20;
    const size_t cols = 40;

    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {rows, cols});
    void* original_ptr = device_tensor.GetPtr<void>();

    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_FP16, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    EXPECT_EQ(device_tensor.dtype, TYPE_FP16);
    EXPECT_EQ(device_tensor.GetPtr<void>(), original_ptr);  // 原地转换，指针不变
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols);
  }

  // 测试用例 7: 相同类型转换（应该直接返回成功）
  {
    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {10, 20});

    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_FP32, kDeviceRank);
    EXPECT_TRUE(status.OK());

    EXPECT_EQ(device_tensor.dtype, TYPE_FP32);
  }

  // 测试用例 8: 不支持的类型转换（应该抛出异常）
  {
    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {10, 20});

    // 假设 TYPE_UINT8 到 TYPE_FP16 的转换不被支持
    EXPECT_THROW(weight_loader.CastDeviceTensorType(device_tensor, TYPE_UINT8, kDeviceRank), std::exception);
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, TensorParallelSplitTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  constexpr size_t kParaSize = 2;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: 1D 张量 RowPara 切分
  {
    const size_t size = 100;

    // 创建 HOST 上的 1D 张量
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {size});

    // 执行 TensorParallelSplit
    Tensor device_tensor =
        weight_loader.TensorParallelSplit(host_tensor, kDeviceRank, kParaSize, TensorParallelMode::RowPara);
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证输出张量的属性
    EXPECT_EQ(device_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(device_tensor.dtype, TYPE_FP32);
    EXPECT_EQ(device_tensor.shape.size(), 1);
    EXPECT_EQ(device_tensor.shape[0], size / kParaSize);
  }

  // 测试用例 2: 2D 张量 RowPara 切分
  {
    const size_t rows = 100;
    const size_t cols = 50;

    // 创建 HOST 上的 2D 张量
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {rows, cols});

    // 执行 TensorParallelSplit
    Tensor device_tensor =
        weight_loader.TensorParallelSplit(host_tensor, kDeviceRank, kParaSize, TensorParallelMode::RowPara);
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证输出张量的属性
    EXPECT_EQ(device_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(device_tensor.dtype, TYPE_FP16);
    EXPECT_EQ(device_tensor.shape.size(), 2);
    EXPECT_EQ(device_tensor.shape[0], rows / kParaSize);
    EXPECT_EQ(device_tensor.shape[1], cols);
  }

  // 测试用例 3: 2D 张量 ColPara 切分
  {
    const size_t rows = 80;
    const size_t cols = 60;

    // 创建 HOST 上的 2D 张量
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {rows, cols});

    // 执行 TensorParallelSplit
    Tensor device_tensor =
        weight_loader.TensorParallelSplit(host_tensor, kDeviceRank, kParaSize, TensorParallelMode::ColPara);
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证输出张量的属性
    EXPECT_EQ(device_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(device_tensor.dtype, TYPE_FP32);
    EXPECT_EQ(device_tensor.shape.size(), 2);
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols / kParaSize);
  }

  // 测试用例 4: 1D 张量使用 ColPara 模式（当前实现会忽略 tp_mode，按 RowPara 处理）
  {
    const size_t size = 100;
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {size});

    // 1D 张量使用 ColPara 模式时，实际按 RowPara 模式处理
    Tensor device_tensor =
        weight_loader.TensorParallelSplit(host_tensor, kDeviceRank, kParaSize, TensorParallelMode::ColPara);
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证输出张量的属性（与 RowPara 模式相同）
    EXPECT_EQ(device_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(device_tensor.dtype, TYPE_FP32);
    EXPECT_EQ(device_tensor.shape.size(), 1);
    EXPECT_EQ(device_tensor.shape[0], size / kParaSize);
  }

  // 测试用例 5: 错误情况 - 张量大小不能被 para_size 整除（应该抛出异常）
  {
    // 1D 张量
    Tensor host_tensor_1d(MemoryLocation::LOCATION_HOST, TYPE_FP32, {101});
    EXPECT_THROW(weight_loader.TensorParallelSplit(host_tensor_1d, kDeviceRank, kParaSize, TensorParallelMode::RowPara),
                 std::exception);

    // 2D 张量 RowPara
    Tensor host_tensor_2d_row(MemoryLocation::LOCATION_HOST, TYPE_FP32, {101, 50});
    EXPECT_THROW(
        weight_loader.TensorParallelSplit(host_tensor_2d_row, kDeviceRank, kParaSize, TensorParallelMode::RowPara),
        std::exception);

    // 2D 张量 ColPara
    Tensor host_tensor_2d_col(MemoryLocation::LOCATION_HOST, TYPE_FP32, {100, 51});
    EXPECT_THROW(
        weight_loader.TensorParallelSplit(host_tensor_2d_col, kDeviceRank, kParaSize, TensorParallelMode::ColPara),
        std::exception);
  }

  // 测试用例 6: 错误情况 - 3D 张量（应该抛出异常）
  {
    Tensor host_tensor_3d(MemoryLocation::LOCATION_HOST, TYPE_FP32, {10, 20, 30});
    EXPECT_THROW(weight_loader.TensorParallelSplit(host_tensor_3d, kDeviceRank, kParaSize, TensorParallelMode::RowPara),
                 std::exception);
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, LmHeadTensorParallelSplitTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  constexpr size_t kParaSize = 2;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: 正常情况 - vocab_size 可以被 para_size 整除
  {
    const size_t vocab_size = 1000;
    const size_t hidden_size = 512;

    // 创建 HOST 上的 2D 张量
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {vocab_size, hidden_size});

    // 执行 LmHeadTensorParallelSplit
    Tensor device_tensor =
        weight_loader.LmHeadTensorParallelSplit(host_tensor, kDeviceRank, kParaSize, TensorParallelMode::RowPara);
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证输出张量的属性
    EXPECT_EQ(device_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(device_tensor.dtype, TYPE_FP32);
    EXPECT_EQ(device_tensor.shape.size(), 2);
    EXPECT_EQ(device_tensor.shape[0], vocab_size / kParaSize);
    EXPECT_EQ(device_tensor.shape[1], hidden_size);
  }

  // 测试用例 2: vocab_size 不能被 para_size 整除的情况
  {
    const size_t vocab_size = 1001;  // 不能被kParaSize整除
    const size_t hidden_size = 256;

    // 创建 HOST 上的 2D 张量
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {vocab_size, hidden_size});
    std::vector<half_float::half> host_data(vocab_size * hidden_size);
    std::default_random_engine eng(123);
    std::uniform_real_distribution<float> random_range(-1.0f, 1.0f);
    for (size_t i = 0; i < vocab_size * hidden_size; ++i) {
      host_data[i] = half_float::half(random_range(eng));
    }
    std::memcpy(host_tensor.GetPtr<void>(), host_data.data(), host_tensor.GetTotalBytes());

    // 执行 LmHeadTensorParallelSplit - rank 0
    Tensor device_tensor_rank0 =
        weight_loader.LmHeadTensorParallelSplit(host_tensor, 0, kParaSize, TensorParallelMode::RowPara);
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证 rank 0 的 shape (使用 DivRoundUp)
    const size_t vocab_size_per_tp = DivRoundUp(vocab_size, kParaSize);  // 501
    EXPECT_EQ(device_tensor_rank0.shape[0], vocab_size_per_tp);
    EXPECT_EQ(device_tensor_rank0.shape[1], hidden_size);

    // 执行 LmHeadTensorParallelSplit - rank 1 (最后一个rank)
    Tensor device_tensor_rank1 =
        weight_loader.LmHeadTensorParallelSplit(host_tensor, 1, kParaSize, TensorParallelMode::RowPara);
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证 rank 1 的 shape (仍然是 vocab_size_per_tp，但实际拷贝的数据更少)
    EXPECT_EQ(device_tensor_rank1.shape[0], vocab_size_per_tp);
    EXPECT_EQ(device_tensor_rank1.shape[1], hidden_size);

    // 验证 rank 1 的数据正确性（只验证实际拷贝的范围）
    const size_t split_vocab_size = vocab_size - 1 * vocab_size_per_tp;  // 1001 - 501 = 500
    std::vector<half_float::half> device_data_rank1(vocab_size_per_tp * hidden_size);
    MemcpyAsync(device_data_rank1.data(), device_tensor_rank1.GetPtr<void>(), device_tensor_rank1.GetTotalBytes(),
                MEMCPY_DEVICE_TO_HOST, context_->device->GetMemoryManageStream());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 只验证实际拷贝的数据：从 host_data 的 vocab_size_per_tp * hidden_size 位置开始
    const size_t host_offset = vocab_size_per_tp * hidden_size;  // 501 * 256
    for (size_t i = 0; i < split_vocab_size * hidden_size; ++i) {
      EXPECT_EQ(device_data_rank1[i], host_data[host_offset + i]);
    }
  }

  // 测试用例 3: 错误情况 - 1D 张量应该抛出异常
  {
    Tensor host_tensor_1d(MemoryLocation::LOCATION_HOST, TYPE_FP32, {1000});
    EXPECT_THROW(
        weight_loader.LmHeadTensorParallelSplit(host_tensor_1d, kDeviceRank, kParaSize, TensorParallelMode::RowPara),
        std::exception);
  }

  // 测试用例 4: 错误情况 - ColPara 模式应该抛出异常
  {
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {1000, 512});
    EXPECT_THROW(
        weight_loader.LmHeadTensorParallelSplit(host_tensor, kDeviceRank, kParaSize, TensorParallelMode::ColPara),
        std::exception);
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, AutoMergeWeightTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: 合并两个 2D 张量
  {
    const size_t rows1 = 100;
    const size_t rows2 = 200;
    const size_t cols = 50;

    // 创建两个 DEVICE 张量
    Tensor tensor1(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {rows1, cols});
    Tensor tensor2(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {rows2, cols});

    // 创建 dev_weights_map
    std::unordered_map<std::string, Tensor> dev_weights_map;
    dev_weights_map["weight1"] = tensor1;
    dev_weights_map["weight2"] = tensor2;

    // 执行 AutoMergeWeight
    std::vector<std::string> input_names = {"weight1", "weight2"};
    Status status = weight_loader.AutoMergeWeight(input_names, "merged_weight", dev_weights_map, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证合并后的张量存在
    EXPECT_TRUE(dev_weights_map.find("merged_weight") != dev_weights_map.end());

    // 验证原始张量已被删除
    EXPECT_TRUE(dev_weights_map.find("weight1") == dev_weights_map.end());
    EXPECT_TRUE(dev_weights_map.find("weight2") == dev_weights_map.end());

    // 验证合并后的张量 shape
    const Tensor& merged_tensor = dev_weights_map.at("merged_weight");
    EXPECT_EQ(merged_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(merged_tensor.dtype, TYPE_FP32);
    EXPECT_EQ(merged_tensor.shape.size(), 2);
    EXPECT_EQ(merged_tensor.shape[0], rows1 + rows2);
    EXPECT_EQ(merged_tensor.shape[1], cols);
  }

  // 测试用例 2: 合并三个 2D 张量
  {
    const size_t rows1 = 50;
    const size_t rows2 = 75;
    const size_t rows3 = 100;
    const size_t cols = 64;

    // 创建三个 DEVICE 张量
    Tensor tensor1(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {rows1, cols});
    Tensor tensor2(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {rows2, cols});
    Tensor tensor3(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {rows3, cols});

    // 创建 dev_weights_map
    std::unordered_map<std::string, Tensor> dev_weights_map;
    dev_weights_map["q_proj"] = tensor1;
    dev_weights_map["k_proj"] = tensor2;
    dev_weights_map["v_proj"] = tensor3;

    // 执行 AutoMergeWeight
    std::vector<std::string> input_names = {"q_proj", "k_proj", "v_proj"};
    Status status = weight_loader.AutoMergeWeight(input_names, "qkv_proj", dev_weights_map, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证合并后的张量存在
    EXPECT_TRUE(dev_weights_map.find("qkv_proj") != dev_weights_map.end());

    // 验证原始张量已被删除
    EXPECT_TRUE(dev_weights_map.find("q_proj") == dev_weights_map.end());
    EXPECT_TRUE(dev_weights_map.find("k_proj") == dev_weights_map.end());
    EXPECT_TRUE(dev_weights_map.find("v_proj") == dev_weights_map.end());

    // 验证合并后的张量 shape
    const Tensor& merged_tensor = dev_weights_map.at("qkv_proj");
    EXPECT_EQ(merged_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(merged_tensor.dtype, TYPE_FP16);
    EXPECT_EQ(merged_tensor.shape.size(), 2);
    EXPECT_EQ(merged_tensor.shape[0], rows1 + rows2 + rows3);
    EXPECT_EQ(merged_tensor.shape[1], cols);
  }

  // 测试用例 3: 错误情况 - 输入权重不存在
  {
    std::unordered_map<std::string, Tensor> dev_weights_map;
    Tensor tensor1(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {100, 50});
    dev_weights_map["weight1"] = tensor1;

    // 尝试合并不存在的权重
    std::vector<std::string> input_names = {"weight1", "non_existent_weight"};
    Status status = weight_loader.AutoMergeWeight(input_names, "merged_weight", dev_weights_map, kDeviceRank);
    EXPECT_FALSE(status.OK());
    EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);
  }

  // 测试用例 4: 合并不同数据类型的张量（应该使用第一个张量的类型）
  {
    const size_t rows1 = 80;
    const size_t rows2 = 120;
    const size_t cols = 32;

    // 创建两个不同类型的 DEVICE 张量
    Tensor tensor1(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {rows1, cols});
    Tensor tensor2(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {rows2, cols});

    // 创建 dev_weights_map
    std::unordered_map<std::string, Tensor> dev_weights_map;
    dev_weights_map["gate_proj"] = tensor1;
    dev_weights_map["up_proj"] = tensor2;

    // 执行 AutoMergeWeight
    std::vector<std::string> input_names = {"gate_proj", "up_proj"};
    Status status = weight_loader.AutoMergeWeight(input_names, "gate_up_proj", dev_weights_map, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证合并后的张量 shape 和类型
    const Tensor& merged_tensor = dev_weights_map.at("gate_up_proj");
    EXPECT_EQ(merged_tensor.dtype, TYPE_BF16);
    EXPECT_EQ(merged_tensor.shape[0], rows1 + rows2);
    EXPECT_EQ(merged_tensor.shape[1], cols);
  }

  // 测试用例 5: 合并单个张量（边界情况）
  {
    const size_t rows = 100;
    const size_t cols = 50;

    Tensor tensor1(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {rows, cols});

    std::unordered_map<std::string, Tensor> dev_weights_map;
    dev_weights_map["single_weight"] = tensor1;

    // 执行 AutoMergeWeight
    std::vector<std::string> input_names = {"single_weight"};
    Status status = weight_loader.AutoMergeWeight(input_names, "merged_single", dev_weights_map, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证合并后的张量 shape（应该与原始张量相同）
    const Tensor& merged_tensor = dev_weights_map.at("merged_single");
    EXPECT_EQ(merged_tensor.shape[0], rows);
    EXPECT_EQ(merged_tensor.shape[1], cols);
    EXPECT_TRUE(dev_weights_map.find("single_weight") == dev_weights_map.end());
  }

#endif
}

// 测试 Host侧预切分 + 整块H2D 优化路径的正确性
// 验证方法: 对比优化路径（HostTensorParallelSlice2D + MemcpyAsync）与原始路径（Memcpy2DAsync）的结果一致性
TEST_F(CommonModelWeightLoaderTest, HostTensorParallelSlice2DConsistencyTest) {
#ifdef ENABLE_KUNLUN
  constexpr int kDeviceRank0 = 0;
  constexpr int kDeviceRank1 = 1;
  constexpr size_t kParaSize = 2;

  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // ==================== 测试用例1: RowPara (dim=0切分) ====================
  // RowPara切分后数据连续，优化路径使用零拷贝指针偏移
  {
    const size_t rows = 128;
    const size_t cols = 64;

    // 创建host tensor并填充确定性数据
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {rows, cols});
    float* host_data = host_tensor.GetPtr<float>();
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = static_cast<float>(i);  // 使用索引作为值，方便验证
    }

    // 路径A: 使用优化后的 TensorParallelSplit2D（内部走 HostTensorParallelSlice2D + 整块H2D）
    Tensor dev_tensor_rank0 =
        weight_loader.TensorParallelSplit2D(host_tensor, kDeviceRank0, kParaSize, TensorParallelMode::RowPara);
    Tensor dev_tensor_rank1 =
        weight_loader.TensorParallelSplit2D(host_tensor, kDeviceRank1, kParaSize, TensorParallelMode::RowPara);
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证shape
    EXPECT_EQ(dev_tensor_rank0.shape[0], rows / kParaSize);
    EXPECT_EQ(dev_tensor_rank0.shape[1], cols);
    EXPECT_EQ(dev_tensor_rank1.shape[0], rows / kParaSize);
    EXPECT_EQ(dev_tensor_rank1.shape[1], cols);

    // 将device数据拷回host验证
    const size_t slice_rows = rows / kParaSize;
    std::vector<float> result_rank0(slice_rows * cols);
    std::vector<float> result_rank1(slice_rows * cols);
    MemcpyAsync(result_rank0.data(), dev_tensor_rank0.GetPtr<void>(), dev_tensor_rank0.GetTotalBytes(),
                MEMCPY_DEVICE_TO_HOST, context_->device->GetMemoryManageStream());
    MemcpyAsync(result_rank1.data(), dev_tensor_rank1.GetPtr<void>(), dev_tensor_rank1.GetTotalBytes(),
                MEMCPY_DEVICE_TO_HOST, context_->device->GetMemoryManageStream());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 路径B: 手动计算期望结果（模拟原始Memcpy2D逻辑）
    // RowPara rank0: 取前 slice_rows 行
    for (size_t i = 0; i < slice_rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        float expected = static_cast<float>(i * cols + j);
        EXPECT_EQ(result_rank0[i * cols + j], expected) << "RowPara rank0 mismatch at [" << i << "," << j << "]";
      }
    }
    // RowPara rank1: 取后 slice_rows 行
    for (size_t i = 0; i < slice_rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        float expected = static_cast<float>((i + slice_rows) * cols + j);
        EXPECT_EQ(result_rank1[i * cols + j], expected) << "RowPara rank1 mismatch at [" << i << "," << j << "]";
      }
    }
  }

  // ==================== 测试用例2: ColPara (dim=1切分) ====================
  // ColPara切分后数据不连续，优化路径在host侧gather到连续buffer
  {
    const size_t rows = 64;
    const size_t cols = 128;

    // 创建host tensor并填充确定性数据
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {rows, cols});
    float* host_data = host_tensor.GetPtr<float>();
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = static_cast<float>(i);
    }

    // 使用优化后的 TensorParallelSplit2D
    Tensor dev_tensor_rank0 =
        weight_loader.TensorParallelSplit2D(host_tensor, kDeviceRank0, kParaSize, TensorParallelMode::ColPara);
    Tensor dev_tensor_rank1 =
        weight_loader.TensorParallelSplit2D(host_tensor, kDeviceRank1, kParaSize, TensorParallelMode::ColPara);
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证shape
    const size_t slice_cols = cols / kParaSize;
    EXPECT_EQ(dev_tensor_rank0.shape[0], rows);
    EXPECT_EQ(dev_tensor_rank0.shape[1], slice_cols);
    EXPECT_EQ(dev_tensor_rank1.shape[0], rows);
    EXPECT_EQ(dev_tensor_rank1.shape[1], slice_cols);

    // 将device数据拷回host验证
    std::vector<float> result_rank0(rows * slice_cols);
    std::vector<float> result_rank1(rows * slice_cols);
    MemcpyAsync(result_rank0.data(), dev_tensor_rank0.GetPtr<void>(), dev_tensor_rank0.GetTotalBytes(),
                MEMCPY_DEVICE_TO_HOST, context_->device->GetMemoryManageStream());
    MemcpyAsync(result_rank1.data(), dev_tensor_rank1.GetPtr<void>(), dev_tensor_rank1.GetTotalBytes(),
                MEMCPY_DEVICE_TO_HOST, context_->device->GetMemoryManageStream());
    StreamSynchronize(context_->device->GetMemoryManageStream());

    // 验证: ColPara rank0取每行的前 slice_cols 列
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < slice_cols; ++j) {
        float expected = static_cast<float>(i * cols + j);
        EXPECT_EQ(result_rank0[i * slice_cols + j], expected) << "ColPara rank0 mismatch at [" << i << "," << j << "]";
      }
    }
    // 验证: ColPara rank1取每行的后 slice_cols 列
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < slice_cols; ++j) {
        float expected = static_cast<float>(i * cols + (j + slice_cols));
        EXPECT_EQ(result_rank1[i * slice_cols + j], expected) << "ColPara rank1 mismatch at [" << i << "," << j << "]";
      }
    }
  }

  // ==================== 测试用例3: ColPara 大tensor (模拟真实权重) ====================
  // 模拟 Qwen MoE down_proj [4096, 2048] TP=4
  {
    const size_t rows = 4096;
    const size_t cols = 2048;
    const size_t para_size = 4;

    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {rows, cols});
    uint16_t* host_data = host_tensor.GetPtr<uint16_t>();
    // 用低16位填充确定性模式
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = static_cast<uint16_t>(i & 0xFFFF);
    }

    // 对每个rank做切分
    const size_t slice_cols = cols / para_size;
    for (size_t rank = 0; rank < para_size; ++rank) {
      Tensor dev_tensor = weight_loader.TensorParallelSplit2D(host_tensor, static_cast<int>(rank), para_size,
                                                              TensorParallelMode::ColPara);
      StreamSynchronize(context_->device->GetMemoryManageStream());

      EXPECT_EQ(dev_tensor.shape[0], rows);
      EXPECT_EQ(dev_tensor.shape[1], slice_cols);

      // 拷回验证
      std::vector<uint16_t> result(rows * slice_cols);
      MemcpyAsync(result.data(), dev_tensor.GetPtr<void>(), dev_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                  context_->device->GetMemoryManageStream());
      StreamSynchronize(context_->device->GetMemoryManageStream());

      // 抽样验证（全量验证太慢）
      const size_t col_offset = slice_cols * rank;
      for (size_t i = 0; i < rows; i += 256) {         // 每256行验证一次
        for (size_t j = 0; j < slice_cols; j += 64) {  // 每64列验证一次
          size_t src_idx = i * cols + (j + col_offset);
          uint16_t expected = static_cast<uint16_t>(src_idx & 0xFFFF);
          EXPECT_EQ(result[i * slice_cols + j], expected)
              << "Large ColPara rank" << rank << " mismatch at [" << i << "," << j << "]";
        }
      }
    }
  }

  // ==================== 测试用例4: RowPara 大tensor TP=4 ====================
  {
    const size_t rows = 4096;
    const size_t cols = 2048;
    const size_t para_size = 4;

    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {rows, cols});
    uint16_t* host_data = host_tensor.GetPtr<uint16_t>();
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = static_cast<uint16_t>(i & 0xFFFF);
    }

    const size_t slice_rows = rows / para_size;
    for (size_t rank = 0; rank < para_size; ++rank) {
      Tensor dev_tensor = weight_loader.TensorParallelSplit2D(host_tensor, static_cast<int>(rank), para_size,
                                                              TensorParallelMode::RowPara);
      StreamSynchronize(context_->device->GetMemoryManageStream());

      EXPECT_EQ(dev_tensor.shape[0], slice_rows);
      EXPECT_EQ(dev_tensor.shape[1], cols);

      // 拷回验证
      std::vector<uint16_t> result(slice_rows * cols);
      MemcpyAsync(result.data(), dev_tensor.GetPtr<void>(), dev_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                  context_->device->GetMemoryManageStream());
      StreamSynchronize(context_->device->GetMemoryManageStream());

      // 抽样验证
      const size_t row_offset = slice_rows * rank;
      for (size_t i = 0; i < slice_rows; i += 256) {
        for (size_t j = 0; j < cols; j += 256) {
          size_t src_idx = (i + row_offset) * cols + j;
          uint16_t expected = static_cast<uint16_t>(src_idx & 0xFFFF);
          EXPECT_EQ(result[i * cols + j], expected)
              << "Large RowPara rank" << rank << " mismatch at [" << i << "," << j << "]";
        }
      }
    }
  }

  // ==================== 测试用例5: HostTensorParallelSlice2D 直接验证 ====================
  // 直接验证host侧切分结果的数据正确性（不经过device）
  {
    const size_t rows = 8;
    const size_t cols = 16;

    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {rows, cols});
    float* host_data = host_tensor.GetPtr<float>();
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = static_cast<float>(i);
    }

    // RowPara: 验证零拷贝指针正确性
    {
      Tensor slice = weight_loader.HostTensorParallelSlice2D(host_tensor, 0, 2, TensorParallelMode::RowPara);
      EXPECT_EQ(slice.shape[0], rows / 2);
      EXPECT_EQ(slice.shape[1], cols);
      // 零拷贝: 指针应该指向原始tensor偏移处
      EXPECT_EQ(slice.GetPtr<float>(), host_data);

      Tensor slice1 = weight_loader.HostTensorParallelSlice2D(host_tensor, 1, 2, TensorParallelMode::RowPara);
      EXPECT_EQ(slice1.GetPtr<float>(), host_data + (rows / 2) * cols);
    }

    // ColPara: 验证gather结果
    {
      Tensor slice = weight_loader.HostTensorParallelSlice2D(host_tensor, 0, 2, TensorParallelMode::ColPara);
      EXPECT_EQ(slice.shape[0], rows);
      EXPECT_EQ(slice.shape[1], cols / 2);
      // ColPara分配了新buffer，指针不同
      EXPECT_NE(static_cast<void*>(slice.GetPtr<float>()), static_cast<void*>(host_data));

      // 验证数据: rank0取每行前半列
      const size_t slice_cols = cols / 2;
      const float* slice_data = slice.GetPtr<float>();
      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < slice_cols; ++j) {
          float expected = static_cast<float>(i * cols + j);
          EXPECT_EQ(slice_data[i * slice_cols + j], expected)
              << "Direct ColPara rank0 mismatch at [" << i << "," << j << "]";
        }
      }

      // rank1取每行后半列
      Tensor slice1 = weight_loader.HostTensorParallelSlice2D(host_tensor, 1, 2, TensorParallelMode::ColPara);
      const float* slice1_data = slice1.GetPtr<float>();
      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < slice_cols; ++j) {
          float expected = static_cast<float>(i * cols + j + slice_cols);
          EXPECT_EQ(slice1_data[i * slice_cols + j], expected)
              << "Direct ColPara rank1 mismatch at [" << i << "," << j << "]";
        }
      }
    }
  }
#endif
}

}  // namespace ksana_llm
