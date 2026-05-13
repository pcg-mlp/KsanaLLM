/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#include "layers/layer_test_base.h"

// specific includes for matmul tests
#include "layers/batched_matmul_layer.h"
#include "layers/cutlass_matmul_layer.h"
#include "layers/finegrained_mixed_dtype_gemm_layer.h"
#include "layers/machete_matmul_layer.h"
#include "layers/marlin_matmul_layer.h"
#include "layers/matmul_layer.h"
#include "layers/matmul_layer_factory.h"
#include "utils/search_status.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
torch::Tensor GetRefGptq(torch::Tensor a, torch::Tensor pack_b, torch::Tensor b_scale) {
  int n = b_scale.size(1);
  int k = a.size(1);
  int groupsize = a.size(1) / b_scale.size(0);

  torch::Tensor w_packed_int4x2 = pack_b.t().contiguous().view(torch::kUInt8);
  torch::Tensor w_unpacked = torch::zeros({w_packed_int4x2.size(0), w_packed_int4x2.size(1) * 2},
                                          torch::TensorOptions().device(w_packed_int4x2.device()).dtype(torch::kInt8));
  w_unpacked.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)},
                        w_packed_int4x2 % 16);
  w_unpacked.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)},
                        w_packed_int4x2 / 16);
  w_unpacked = w_unpacked.t().contiguous();

  b_scale = b_scale.unsqueeze(1).repeat({1, groupsize, 1}).reshape({k, n}).contiguous();

  torch::Tensor b = b_scale * (w_unpacked - 8);

  return torch::matmul(a, b);
}
#endif

TEST_F(LayerTest, MacheteMatMulLayerTest) {
#ifdef ENABLE_CUDA

  if (context_->device->ext->GetComputeCapacity() != 90) {
    return;
  }

  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;
  using device_type = half;
  runtime_config.inter_data_type = TYPE_FP16;

  // 初始化参数
  const size_t max_m = 1024;
  const size_t max_n = 8192;
  const size_t max_k = 28672;
  const size_t groupsize = 128;
  const bool is_awq = false;
  const bool is_gptq_desc = false;
  const bool is_k_full = false;
  const bool cutlass_use_gemv_cuda_core = false;

  // 创建MacheteMatMulLayer实例
  MacheteMatMulLayer machete_matmul_layer;
  machete_matmul_layer.Init(
      {max_m, max_n, max_k, groupsize, is_awq, is_gptq_desc, is_k_full, cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
      runtime_config, context_, kDeviceRank);

  // 获取工作空间大小并分配
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  machete_matmul_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(machete_matmul_layer.GetWorkspaceSize()));

  // 准备输入张量
  const size_t m = 96;  // 实际使用的m值，小于max_m
  const size_t bits = 4;
  const size_t pack_factor = 32 / bits;

  // 创建输入数据
  Tensor input_activation = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_k});
  Tensor weight = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_k / pack_factor, max_n});
  Tensor scales = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {max_k / groupsize, max_n});

  // 赋初值并拷贝到device，必要的步骤
  {
    std::default_random_engine eng;
    std::uniform_real_distribution<float> random_range(-1, 1);

    std::vector<dtype> input_activation_host(input_activation.GetElementNumber());
    for (size_t i = 0; i < input_activation.GetElementNumber(); ++i) {
      input_activation_host[i] = static_cast<dtype>(random_range(eng));
    }
    MemcpyAsync(input_activation.GetPtr<void>(), input_activation_host.data(), input_activation.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

    std::vector<dtype> scales_host(scales.GetElementNumber());
    for (size_t i = 0; i < scales.GetElementNumber(); ++i) {
      scales_host[i] = static_cast<dtype>(random_range(eng));
    }
    MemcpyAsync(scales.GetPtr<void>(), scales_host.data(), scales.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());

    std::vector<int32_t> weight_host(weight.GetElementNumber());
    for (size_t i = 0; i < weight.GetElementNumber(); ++i) {
      weight_host[i] = static_cast<int32_t>(1000 * random_range(eng));
    }
    MemcpyAsync(weight.GetPtr<void>(), weight_host.data(), weight.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());

    StreamSynchronize(context_->device->GetPrimaryComputeStream());
  }

  // 权重预处理
  Tensor weightT = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_k / pack_factor, max_n});
  Tensor weightPrePack = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_k / pack_factor, max_n});
  llm_kernels::nvidia::InvokePermute<2ul, sizeof(int32_t)>(weight.GetPtr<void>(), weightT.GetPtr<void>(),
                                                           {max_k / pack_factor, max_n}, {1, 0},
                                                           context_->device->GetPrimaryComputeStream().Get());
  llm_kernels::nvidia::machete::machete_prepack_weight(
      weightT.GetPtr<void>(), {max_k / pack_factor, max_n}, weightPrePack.GetPtr<void>(),
      llm_kernels::nvidia::vllm_dtype::kHalf, llm_kernels::nvidia::vllm_dtype::kU4B8,
      llm_kernels::nvidia::vllm_dtype::kHalf, context_->device->GetPrimaryComputeStream().Get());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // 绑定scale
  weightPrePack.scales = &scales;

  // 执行默认矩阵计算
  Tensor output0 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_n});
  int64_t default_duration = 0;
  {
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    auto begin_time = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> output_tensors = {output0};
    for (size_t it = 0; it < 100; it++) {
      machete_matmul_layer.Forward({input_activation, weightPrePack}, output_tensors);
    }
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    auto end_time = std::chrono::high_resolution_clock::now();
    default_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
  }

  machete_matmul_layer.Preprocess(model_config, runtime_config);
  machete_matmul_layer.Preprocess(model_config, runtime_config);

  // 执行最优矩阵计算
  Tensor output1 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_n});
  int64_t best_duration = 0;
  {
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    auto begin_time = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> output_tensors = {output1};
    for (size_t it = 0; it < 100; it++) {
      machete_matmul_layer.Forward({input_activation, weightPrePack}, output_tensors);
    }
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    auto end_time = std::chrono::high_resolution_clock::now();
    best_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
  }

  printf("MacheteMatMulLayerTest default time: %ld ms, best time: %ld ms\n", default_duration, best_duration);
  EXPECT_TRUE((best_duration < default_duration) ||
              (std::abs(best_duration - default_duration) <
               0.01 * best_duration));  // 最优配置有可能刚好是默认配置，因此要允许1%的误差，不能强制小于

  // 按照GPTQ计算逻辑计算基准结果
  torch::Tensor t_weight =
      torch::from_blob(weight.GetPtr<void>(), {max_k / pack_factor, max_n}, torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::Device(torch::kCUDA, kDeviceRank));
  torch::Tensor t_scale = torch::from_blob(scales.GetPtr<void>(), {max_k / groupsize, max_n},
                                           torch::TensorOptions().dtype(GetTorchDataType<device_type>()))
                              .to(torch::Device(torch::kCUDA, kDeviceRank));
  torch::Tensor t_a = torch::from_blob(input_activation.GetPtr<void>(), {m, max_k},
                                       torch::TensorOptions().dtype(GetTorchDataType<device_type>()))
                          .to(torch::Device(torch::kCUDA, kDeviceRank));
  torch::Tensor ref = GetRefGptq(t_a, t_weight, t_scale);

  // 验证输出形状
  EXPECT_EQ(output0.shape[0], m);
  EXPECT_EQ(output0.shape[1], max_n);
  EXPECT_EQ(output1.shape[0], m);
  EXPECT_EQ(output1.shape[1], max_n);

  // 验证结果
  std::vector<dtype> ref_host(m * max_n);
  std::vector<dtype> output0_host(m * max_n);
  std::vector<dtype> output1_host(m * max_n);
  Memcpy(ref_host.data(), ref.data_ptr(), ref_host.size() * sizeof(dtype), MEMCPY_DEVICE_TO_HOST);
  Memcpy(output0_host.data(), output0.GetPtr<void>(), output0_host.size() * sizeof(dtype), MEMCPY_DEVICE_TO_HOST);
  Memcpy(output1_host.data(), output1.GetPtr<void>(), output1_host.size() * sizeof(dtype), MEMCPY_DEVICE_TO_HOST);
  StreamSynchronize(context_->device->GetMemoryManageStream());
  for (size_t idx = 0; idx < m * max_n; idx++) {
    EXPECT_NEAR(ref_host[idx], output0_host[idx], 1.0);  // 算出来的结果数量级较大，不能用EXPECT_FLOAT_EQ完全比较
    EXPECT_NEAR(ref_host[idx], output1_host[idx], 1.0);
  }

#endif
}

TEST_F(LayerTest, CutlassMatMulLayerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;
  runtime_config.inter_data_type = TYPE_FP16;

  // 初始化参数
  const size_t max_m = 1024;
  const size_t max_n = 8192;
  const size_t max_k = 28672;
  const size_t groupsize = 128;
  const bool is_awq = false;
  const bool is_gptq_desc = false;
  const bool is_k_full = false;
  const bool cutlass_use_gemv_cuda_core = true;

  // 创建CutlassMatMulLayer实例
  CutlassMatMulLayer cutlass_matmul_layer;
  cutlass_matmul_layer.Init(
      {max_m, max_n, max_k, groupsize, is_awq, is_gptq_desc, is_k_full, cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
      runtime_config, context_, kDeviceRank);

  // 获取工作空间大小并分配
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  cutlass_matmul_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(cutlass_matmul_layer.GetWorkspaceSize()));

  // 准备输入张量
  const size_t m = 96;  // 实际使用的m值，小于max_m
  const size_t bits = 4;
  const size_t pack_factor = 32 / bits;

  Tensor input_activation = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_k});
  Tensor output0 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_n});
  Tensor output1 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_n});
  Tensor weight = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {max_k, max_n / 2});
  Tensor scales = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {max_k / groupsize, max_n});
  weight.scales = &scales;

  // 执行默认矩阵计算
  int64_t default_duration = 0;
  {
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    auto begin_time = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> output_tensors = {output0};
    for (size_t it = 0; it < 100; it++) {
      cutlass_matmul_layer.Forward({input_activation, weight}, output_tensors);
    }
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    auto end_time = std::chrono::high_resolution_clock::now();
    default_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
  }

  cutlass_matmul_layer.Preprocess(model_config, runtime_config);
  cutlass_matmul_layer.Preprocess(model_config, runtime_config);

  // 执行最优矩阵计算
  int64_t best_duration = 0;
  {
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    auto begin_time = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> output_tensors = {output1};
    for (size_t it = 0; it < 100; it++) {
      cutlass_matmul_layer.Forward({input_activation, weight}, output_tensors);
    }
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    auto end_time = std::chrono::high_resolution_clock::now();
    best_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
  }

  printf("CutlassMatMulLayerTest default time: %ld ms, best time: %ld ms\n", default_duration, best_duration);
  EXPECT_TRUE((best_duration < default_duration) ||
              (std::abs(best_duration - default_duration) <
               0.01 * best_duration));  // 最优配置有可能刚好只默认配置，因此要允许1%的误差，不能强制小于

  // 验证输出形状
  EXPECT_EQ(output0.shape[0], m);
  EXPECT_EQ(output0.shape[1], max_n);
  EXPECT_EQ(output1.shape[0], m);
  EXPECT_EQ(output1.shape[1], max_n);

  // 验证结果
  std::vector<dtype> output0_host(output0.GetElementNumber());
  std::vector<dtype> output1_host(output1.GetElementNumber());
  Memcpy(output0_host.data(), output0.GetPtr<void>(), output0.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  Memcpy(output1_host.data(), output1.GetPtr<void>(), output1.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  StreamSynchronize(context_->device->GetMemoryManageStream());
  for (size_t idx = 0; idx < m * max_n; idx++) {
    EXPECT_FLOAT_EQ(output0_host[idx], output1_host[idx]);
  }
#endif
}

TEST_F(LayerTest, MarlinMatMulLayerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;
  using device_type = half;
  runtime_config.inter_data_type = TYPE_FP16;

  // 初始化参数
  const size_t max_m = 1024;
  const size_t max_n = 8192;
  const size_t max_k = 28672;
  const size_t groupsize = 128;
  const bool is_awq = false;
  const bool is_gptq_desc = false;
  const bool is_k_full = true;
  const bool cutlass_use_gemv_cuda_core = false;

  // 创建MarlinMatMulLayer实例
  MarlinMatMulLayer marlin_matmul_layer;
  marlin_matmul_layer.Init(
      {max_m, max_n, max_k, groupsize, is_awq, is_gptq_desc, is_k_full, cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
      runtime_config, context_, kDeviceRank);

  // 获取工作空间大小并分配
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  marlin_matmul_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(marlin_matmul_layer.GetWorkspaceSize()));

  // 准备输入张量
  const size_t m = 96;  // 实际使用的m值，小于max_m
  const size_t bits = 4;
  const size_t pack_factor = 32 / bits;

  Tensor input_activation = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_k});
  Tensor weight = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_k / pack_factor, max_n});
  Tensor scales = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {max_k / groupsize, max_n});

  // 赋初值并拷贝到device，必要的步骤
  {
    std::default_random_engine eng;
    std::uniform_real_distribution<float> random_range(-1, 1);

    std::vector<dtype> input_activation_host(input_activation.GetElementNumber());
    for (size_t i = 0; i < input_activation.GetElementNumber(); ++i) {
      input_activation_host[i] = static_cast<dtype>(random_range(eng));
    }
    MemcpyAsync(input_activation.GetPtr<void>(), input_activation_host.data(), input_activation.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

    std::vector<dtype> scales_host(scales.GetElementNumber());
    for (size_t i = 0; i < scales.GetElementNumber(); ++i) {
      scales_host[i] = static_cast<dtype>(random_range(eng));
    }
    MemcpyAsync(scales.GetPtr<void>(), scales_host.data(), scales.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());

    std::vector<int32_t> weight_host(weight.GetElementNumber());
    for (size_t i = 0; i < weight.GetElementNumber(); ++i) {
      weight_host[i] = static_cast<int32_t>(1000 * random_range(eng));
    }
    MemcpyAsync(weight.GetPtr<void>(), weight_host.data(), weight.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());

    StreamSynchronize(context_->device->GetPrimaryComputeStream());
  }

  // 权重预处理
  std::vector<int64_t> repack_shape = GetMarlinGptqRepackMeta(max_k, max_n, bits);
  Tensor weightPrePack = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                                {static_cast<uint64_t>(repack_shape[0]), static_cast<uint64_t>(repack_shape[1])});
  InvokeMarlinGptqRepack(weight.GetPtr<void>(), nullptr, weightPrePack.GetPtr<void>(), 1, max_k, max_n, bits, false,
                         kDeviceRank, context_->device->GetPrimaryComputeStream().Get());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // scale预处理
  Tensor scalesPrePack = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {max_k / groupsize, max_n});
  InvokeMarlinPermuteScales<device_type>(context_->device->GetPrimaryComputeStream().Get(), scales.GetPtr<void>(),
                                         scalesPrePack.GetPtr<void>(), max_k, max_n, groupsize);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // 绑定scale
  weightPrePack.scales = &scalesPrePack;

  // 执行矩阵计算
  Tensor output = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_n});
  int64_t default_duration = 0;
  {
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    auto begin_time = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> output_tensors = {output};
    for (size_t it = 0; it < 100; it++) {
      marlin_matmul_layer.Forward({input_activation, weightPrePack}, output_tensors);
    }
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    auto end_time = std::chrono::high_resolution_clock::now();
    default_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
  }

  printf("MarlinMatMulLayerTest time: %ld ms\n", default_duration);

  // 按照GPTQ计算逻辑计算基准结果
  torch::Tensor t_weight =
      torch::from_blob(weight.GetPtr<void>(), {max_k / pack_factor, max_n}, torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::Device(torch::kCUDA, kDeviceRank));
  torch::Tensor t_scale = torch::from_blob(scales.GetPtr<void>(), {max_k / groupsize, max_n},
                                           torch::TensorOptions().dtype(GetTorchDataType<device_type>()))
                              .to(torch::Device(torch::kCUDA, kDeviceRank));
  torch::Tensor t_a = torch::from_blob(input_activation.GetPtr<void>(), {m, max_k},
                                       torch::TensorOptions().dtype(GetTorchDataType<device_type>()))
                          .to(torch::Device(torch::kCUDA, kDeviceRank));
  torch::Tensor ref = GetRefGptq(t_a, t_weight, t_scale);

  // 验证输出形状
  EXPECT_EQ(output.shape[0], m);
  EXPECT_EQ(output.shape[1], max_n);

  // 验证结果
  std::vector<dtype> ref_host(m * max_n);
  std::vector<dtype> output_host(m * max_n);
  Memcpy(ref_host.data(), ref.data_ptr(), ref_host.size() * sizeof(dtype), MEMCPY_DEVICE_TO_HOST);
  Memcpy(output_host.data(), output.GetPtr<void>(), output_host.size() * sizeof(dtype), MEMCPY_DEVICE_TO_HOST);
  StreamSynchronize(context_->device->GetMemoryManageStream());
  for (size_t idx = 0; idx < m * max_n; idx++) {
    EXPECT_NEAR(ref_host[idx], output_host[idx], 1.0);  // 算出来的结果数量级较大，不能用EXPECT_FLOAT_EQ完全比较
  }

#endif
}

TEST_F(LayerTest, BatchedMatMulLayerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;
  using device_type = half;
  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;
  cublasCreate(&cublas_handle);
  cublasLtCreate(&cublaslt_handle);

  const int batch_size = 4;
  const int m = 32;
  const int n = 64;
  const int k = 128;

  BatchedMatMulLayer batched_matmul_layer;
  batched_matmul_layer.Init({}, runtime_config, context_, kDeviceRank);

  Tensor input_a = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, batch_size, k});
  Tensor input_b = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {batch_size, k, n});
  Tensor output = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, batch_size, n});

  std::vector<dtype> input_a_host(input_a.GetElementNumber());
  std::vector<dtype> input_b_host(input_b.GetElementNumber());
  std::default_random_engine eng;
  std::uniform_real_distribution<float> random_range(-1, 1);

  for (size_t i = 0; i < input_a.GetElementNumber(); ++i) {
    input_a_host[i] = static_cast<dtype>(random_range(eng));
  }
  MemcpyAsync(input_a.GetPtr<void>(), input_a_host.data(), input_a.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());

  for (size_t i = 0; i < input_b.GetElementNumber(); ++i) {
    input_b_host[i] = static_cast<dtype>(random_range(eng));
  }
  MemcpyAsync(input_b.GetPtr<void>(), input_b_host.data(), input_b.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());

  StreamSynchronize(context_->device->GetMemoryManageStream());

  std::vector<Tensor> output_tensors = {output};
  EXPECT_TRUE(batched_matmul_layer.Forward({input_a, input_b}, output_tensors).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // 验证输出形状
  EXPECT_EQ(output.shape.size(), 3);
  EXPECT_EQ(output.shape[0], m);
  EXPECT_EQ(output.shape[1], batch_size);
  EXPECT_EQ(output.shape[2], n);

  // 使用cublas计算参考结果进行验证
  Tensor ref_output = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, batch_size, n});

  // 对每个batch单独计算矩阵乘法作为参考
  for (int b = 0; b < batch_size; ++b) {
    float alpha = 1.0f;
    float beta = 0.0f;

    void* a_ptr = static_cast<char*>(input_a.GetPtr<void>()) + b * k * sizeof(device_type);
    void* b_ptr = static_cast<char*>(input_b.GetPtr<void>()) + b * k * n * sizeof(device_type);
    void* c_ptr = static_cast<char*>(ref_output.GetPtr<void>()) + b * n * sizeof(device_type);

    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b_ptr, CUDA_R_16F, n, a_ptr, CUDA_R_16F,
                 batch_size * k, &beta, c_ptr, CUDA_R_16F, batch_size * n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
  }

  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // 比较结果
  std::vector<dtype> output_host(output.GetElementNumber());
  std::vector<dtype> ref_output_host(ref_output.GetElementNumber());

  Memcpy(output_host.data(), output.GetPtr<void>(), output.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  Memcpy(ref_output_host.data(), ref_output.GetPtr<void>(), ref_output.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);

  for (size_t i = 0; i < output.GetElementNumber(); ++i) {
    EXPECT_NEAR(static_cast<float>(output_host[i]), static_cast<float>(ref_output_host[i]), 1e-2);
  }

  cublasDestroy(cublas_handle);
  cublasLtDestroy(cublaslt_handle);
#endif
}

TEST_F(LayerTest, MacheteSearchStatusTest) {
#ifdef ENABLE_CUDA
  if (context_->device->ext->GetComputeCapacity() != 90) {
    return;
  }
  constexpr int kDeviceRank = 0;

  // 初始化参数
  const size_t max_m = 32768;  // 32K
  std::vector<std::pair<const size_t, const size_t>> n_k_pairs = {
      {1024, 2048}, {2048, 1024}, {1024, 1024}, {2048, 2048}};
  const size_t groupsize = 128;
  const bool is_awq = false;
  const bool is_gptq_desc = false;
  const bool is_k_full = false;
  const bool cutlass_use_gemv_cuda_core = false;

  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  auto t1 = std::chrono::high_resolution_clock::now();
  {
    std::shared_ptr<MatMulLayerFactory> matmul_layer_factory =
        std::make_shared<MatMulLayerFactory>(model_config, runtime_config, kDeviceRank, context_);
    for (const auto& nk : n_k_pairs) {
      std::shared_ptr<BaseLayer> layer =
          matmul_layer_factory->CreateLayer(TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16,
                                            {max_m, nk.first, nk.second, groupsize, is_awq, is_gptq_desc, is_k_full,
                                             cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
                                            QUANT_GPTQ, MACHETE_LINEAR_BACKEND);
      layer->SetWorkspaceBuffer(workspace_mgr->GetWorkspace(layer->GetWorkspaceSize()));
      layer->Preprocess(model_config, runtime_config);
    }
  }

  Singleton<MacheteSearchStatus>::GetInstance()->ClearMacheteSchedule();
  Singleton<MacheteSearchStatus>::GetInstance()->ClearMacheteWorkspace();

  auto t2 = std::chrono::high_resolution_clock::now();
  {
    std::shared_ptr<MatMulLayerFactory> matmul_layer_factory =
        std::make_shared<MatMulLayerFactory>(model_config, runtime_config, kDeviceRank, context_);

    auto func = [&]() {
      for (size_t layer_idx = 0; layer_idx < model_config.num_layer; layer_idx++) {
        for (const auto& nk : n_k_pairs) {
          std::shared_ptr<BaseLayer> layer =
              matmul_layer_factory->CreateLayer(TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16,
                                                {max_m, nk.first, nk.second, groupsize, is_awq, is_gptq_desc, is_k_full,
                                                 cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
                                                QUANT_GPTQ, MACHETE_LINEAR_BACKEND);
          layer->SetWorkspaceBuffer(workspace_mgr->GetWorkspace(layer->GetWorkspaceSize()));
          layer->Preprocess(model_config, runtime_config);
        }
      }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
      threads.emplace_back(func);
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  auto t3 = std::chrono::high_resolution_clock::now();
  auto duration12 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  auto duration23 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);

  printf("time1: %ld, time2: %ld\n", duration12.count(), duration23.count());

  // 有缓存，创建多次耗时不应该增加太多
  EXPECT_TRUE(2 * duration12.count() > duration23.count());

#endif
}

TEST_F(LayerTest, CutlassSearchStatusTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;

  // 初始化参数
  const size_t max_m = 8192;  // 8K
  std::vector<std::pair<const size_t, const size_t>> n_k_pairs = {
      {1024, 2048}, {2048, 1024}, {1024, 1024}, {2048, 2048}};
  const size_t groupsize = 128;
  const bool is_awq = false;
  const bool is_gptq_desc = false;
  const bool is_k_full = false;
  const bool cutlass_use_gemv_cuda_core = true;
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  auto t1 = std::chrono::high_resolution_clock::now();
  {
    std::shared_ptr<MatMulLayerFactory> matmul_layer_factory =
        std::make_shared<MatMulLayerFactory>(model_config, runtime_config, kDeviceRank, context_);
    for (const auto& nk : n_k_pairs) {
      std::shared_ptr<BaseLayer> layer =
          matmul_layer_factory->CreateLayer(TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16,
                                            {max_m, nk.first, nk.second, groupsize, is_awq, is_gptq_desc, is_k_full,
                                             cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
                                            QUANT_GPTQ, CUTLASS_LINEAR_BACKEND);
      layer->SetWorkspaceBuffer(workspace_mgr->GetWorkspace(layer->GetWorkspaceSize()));
      layer->Preprocess(model_config, runtime_config);
    }
  }

  Singleton<CutlassSearchStatus>::GetInstance()->ClearCutlassSchedule();
  Singleton<CutlassSearchStatus>::GetInstance()->ClearCutlassWorkspace();

  auto t2 = std::chrono::high_resolution_clock::now();
  {
    std::shared_ptr<MatMulLayerFactory> matmul_layer_factory =
        std::make_shared<MatMulLayerFactory>(model_config, runtime_config, kDeviceRank, context_);

    auto func = [&]() {
      for (size_t layer_idx = 0; layer_idx < model_config.num_layer; layer_idx++) {
        for (const auto& nk : n_k_pairs) {
          std::shared_ptr<BaseLayer> layer =
              matmul_layer_factory->CreateLayer(TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16,
                                                {max_m, nk.first, nk.second, groupsize, is_awq, is_gptq_desc, is_k_full,
                                                 cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
                                                QUANT_GPTQ, CUTLASS_LINEAR_BACKEND);
          layer->SetWorkspaceBuffer(workspace_mgr->GetWorkspace(layer->GetWorkspaceSize()));
          layer->Preprocess(model_config, runtime_config);
        }
      }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
      threads.emplace_back(func);
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  auto t3 = std::chrono::high_resolution_clock::now();
  auto duration12 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  auto duration23 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);

  printf("time1: %ld, time2: %ld\n", duration12.count(), duration23.count());

  // 有缓存，创建多次耗时不应该增加太多
  EXPECT_TRUE(2 * duration12.count() > duration23.count());

#endif
}

TEST_F(LayerTest, FinegrainedMixedDtypeGemmLayerTest) {
#ifdef ENABLE_CUDA
  // 仅在 SM90 架构上运行此测试
  if (context_->device->ext->GetComputeCapacity() != 90) {
    return;
  }

  constexpr int kDeviceRank = 0;
  runtime_config.inter_data_type = TYPE_BF16;

  // ============================================================================
  // 第一部分：初始化层参数和工作空间
  // ============================================================================

  // GEMM 参数配置
  // - max_m: 最大 batch size，用于分配工作空间
  // - n, k: 矩阵维度
  // - group_size: 量化分组大小
  // - decode_tactic.size() = max_batch_size + 1 = 129，用于区分 decode/prefill 分支
  const size_t max_m = 1024;
  const size_t n = 1024;
  const size_t k = 1024;
  const size_t group_size = 128;
  const bool has_zero = false;

  // 创建层实例并初始化
  FinegrainedMixedDtypeGemmLayer finegrained_gemm_layer;
  FinegrainedMixedDtypeGemmLayerParameters params{.m = max_m,
                                                  .n = n,
                                                  .k = k,
                                                  .group_size = group_size,
                                                  .has_zero = has_zero,
                                                  .activation_type = TYPE_FP8_E4M3,
                                                  .output_type = TYPE_BF16};
  finegrained_gemm_layer.Init({params}, runtime_config, context_, kDeviceRank);

  // 分配工作空间
  size_t workspace_size = finegrained_gemm_layer.GetWorkspaceSize();
  Tensor workspace = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, {workspace_size});
  std::shared_ptr<Tensor> workspace_ptr = std::make_shared<Tensor>(workspace);
  finegrained_gemm_layer.SetWorkspaceBuffer(workspace_ptr);

  // ============================================================================
  // 第二部分：准备共享的权重张量
  // ============================================================================

  Tensor weight = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, {k, n / 2});
  Tensor input_scale = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1});
  Tensor weight_scale = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {k / group_size, n});

  // 绑定 scales 和 alpha 到权重张量
  weight.input_scales = &input_scale;
  weight.weight_scales = &weight_scale;
  weight.alpha = 1.0f;

  // ============================================================================
  // 第三部分：测试默认 tactic（跳过算子搜索）
  // ============================================================================

  // 设置环境变量跳过算子搜索，使用默认 tactic
  setenv("QUANT_PROFILE", "0", 1);
  finegrained_gemm_layer.Preprocess(model_config, runtime_config);
  unsetenv("QUANT_PROFILE");

  // 测试 decode 分支：m=16 < max_batch_size=128
  const size_t m_decode = 16;
  Tensor input_decode = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {m_decode, k});
  Tensor output_default = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {m_decode, n});
  {
    std::vector<Tensor> output_tensors = {output_default};
    EXPECT_TRUE(finegrained_gemm_layer.Forward({input_decode, weight}, output_tensors).OK());
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
  }

  // 验证输出形状
  EXPECT_EQ(output_default.shape[0], m_decode);
  EXPECT_EQ(output_default.shape[1], n);

  // ============================================================================
  // 第四部分：测试算子搜索和最优 tactic
  // ============================================================================

  // 执行算子搜索
  finegrained_gemm_layer.Preprocess(model_config, runtime_config);
  // 再次调用验证缓存复用
  finegrained_gemm_layer.Preprocess(model_config, runtime_config);

  // 使用最优 tactic 执行 Forward
  Tensor output_optimized = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {m_decode, n});
  {
    std::vector<Tensor> output_tensors = {output_optimized};
    EXPECT_TRUE(finegrained_gemm_layer.Forward({input_decode, weight}, output_tensors).OK());
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
  }

  // 验证输出形状
  EXPECT_EQ(output_optimized.shape[0], m_decode);
  EXPECT_EQ(output_optimized.shape[1], n);

  // ============================================================================
  // 第五部分：测试带 pre_quant_scales 的情况
  // ============================================================================

  Tensor pre_quant_scales = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {k});
  weight.pre_quant_scales = &pre_quant_scales;

  Tensor output_with_prequant = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {m_decode, n});
  {
    std::vector<Tensor> output_tensors = {output_with_prequant};
    EXPECT_TRUE(finegrained_gemm_layer.Forward({input_decode, weight}, output_tensors).OK());
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
  }

  // 验证输出形状
  EXPECT_EQ(output_with_prequant.shape[0], m_decode);
  EXPECT_EQ(output_with_prequant.shape[1], n);

  // 重置 pre_quant_scales
  weight.pre_quant_scales = nullptr;

  // ============================================================================
  // 第六部分：测试 prefill 分支
  // ============================================================================

  // m=256 > max_batch_size=128，命中 prefill 分支
  const size_t m_prefill = 256;
  Tensor input_prefill = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {m_prefill, k});
  Tensor output_prefill = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {m_prefill, n});
  {
    std::vector<Tensor> output_tensors = {output_prefill};
    EXPECT_TRUE(finegrained_gemm_layer.Forward({input_prefill, weight}, output_tensors).OK());
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
  }

  // 验证输出形状
  EXPECT_EQ(output_prefill.shape[0], m_prefill);
  EXPECT_EQ(output_prefill.shape[1], n);
#endif
}

}  // namespace ksana_llm
