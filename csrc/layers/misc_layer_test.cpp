/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/layer_test_base.h"

#include "layers/add_layer.h"
#include "layers/all_reduce_residual_add_norm_layer.h"
#include "layers/assemble_tokens_hidden_layer.h"
#include "layers/input_refit_layer.h"
#include "layers/nccl_all_reduce_sum_layer.h"

namespace ksana_llm {

TEST_F(LayerTest, AddLayerTest) {
#ifndef ENABLE_TOPS

  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;

  // 初始化tensor
  Tensor input, bias_a, bias_b;
  std::vector<Tensor> output(1);
  input = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {32, 64});
  bias_a = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {32, 64});
  bias_b = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {1, 4, 16});
  output[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {32, 64});

  // 赋初值并拷贝到device
  std::vector<dtype> input_host(input.GetElementNumber());
  std::vector<dtype> bias_a_host(bias_a.GetElementNumber());
  std::vector<dtype> bias_b_host(bias_b.GetElementNumber());
  std::default_random_engine eng;
  std::uniform_real_distribution<float> random_range(-1, 1);
  for (size_t i = 0; i < input.GetElementNumber(); ++i) {
    input_host[i] = static_cast<dtype>(random_range(eng));
  }
  MemcpyAsync(input.GetPtr<void>(), input_host.data(), input.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  for (size_t i = 0; i < bias_a.GetElementNumber(); ++i) {
    bias_a_host[i] = static_cast<dtype>(random_range(eng));
  }
  MemcpyAsync(bias_a.GetPtr<void>(), bias_a_host.data(), bias_a.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  for (size_t i = 0; i < bias_b.GetElementNumber(); ++i) {
    bias_b_host[i] = static_cast<dtype>(random_range(eng));
  }
  MemcpyAsync(bias_b.GetPtr<void>(), bias_b_host.data(), bias_b.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  // 测试相同shape
  AddLayer add_layer = AddLayer();
  add_layer.Init({}, runtime_config, context_, kDeviceRank);
  add_layer.Forward({input, bias_a}, output);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
  // 验证结果
  std::vector<dtype> output_host(output[0].GetElementNumber());
  Memcpy(output_host.data(), output[0].GetPtr<void>(), output[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  for (size_t i = 0; i < output[0].GetElementNumber(); ++i) {
    EXPECT_FLOAT_EQ(output_host[i], input_host[i] + bias_a_host[i]);
  }

  // 测试broadcast shape
  add_layer.Forward({input, bias_b}, output);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
  // 验证结果
  Memcpy(output_host.data(), output[0].GetPtr<void>(), output[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  const size_t bias_b_elements = bias_b.GetElementNumber();
  for (size_t i = 0; i < output[0].GetElementNumber(); ++i) {
    EXPECT_FLOAT_EQ(output_host[i], input_host[i] + bias_b_host[i % bias_b_elements]);
  }
#endif
}

TEST_F(LayerTest, AssembleAcceptedTokensHiddenTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;
  using device_type = half;

  // 初始化tensor
  constexpr int batch_size = 2;
  constexpr size_t hidden_size = 1;
  Tensor input, accepted_tokens_idx;
  std::vector<Tensor> output(1);
  std::vector<int32_t> input_tokens_host = {2, 3};
  std::vector<int32_t> accepted_tokens_num_host = {2, 1};
  std::vector<size_t> accepted_tokens_idx_host;
  const size_t input_tokens_size = std::accumulate(input_tokens_host.begin(), input_tokens_host.end(), 0);
  const size_t accepted_tokens_size =
      std::accumulate(accepted_tokens_num_host.begin(), accepted_tokens_num_host.end(), 0);
  accepted_tokens_idx_host.reserve(accepted_tokens_size);
  int32_t input_token_offset = 0;
  int32_t output_token_offset = 0;
  std::vector<size_t> input_token_offsets;
  std::vector<size_t> output_token_offsets;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < accepted_tokens_num_host[i]; ++j) {
      accepted_tokens_idx_host.push_back(input_token_offset + j);
      printf("accepted_tokens_idx_host %d\n", input_token_offset + j);
    }
    input_token_offsets.push_back(input_token_offset);
    output_token_offsets.push_back(output_token_offset);
    input_token_offset += input_tokens_host[i];
    output_token_offset += accepted_tokens_num_host[i];
  }

  input = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {input_tokens_size, hidden_size});
  accepted_tokens_idx = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {accepted_tokens_size});
  output[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {accepted_tokens_size, hidden_size});

  // 赋初值并拷贝到device
  std::vector<dtype> input_host(input.GetElementNumber());
  std::default_random_engine eng;
  std::uniform_real_distribution<float> random_range(-1, 1);
  for (size_t i = 0; i < input.GetElementNumber(); ++i) {
    input_host[i] = static_cast<dtype>(random_range(eng));
  }
  MemcpyAsync(input.GetPtr<void>(), input_host.data(), input.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(accepted_tokens_idx.GetPtr<void>(), accepted_tokens_idx_host.data(), accepted_tokens_idx.GetTotalBytes(),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  AssembleTokensHiddenLayer test_layer = AssembleTokensHiddenLayer();
  test_layer.Init({}, runtime_config, context_, kDeviceRank);
  test_layer.Forward({input, accepted_tokens_idx}, output);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // 验证结果
  std::vector<dtype> output_host(output[0].GetElementNumber());
  Memcpy(output_host.data(), output[0].GetPtr<void>(), output[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  StreamSynchronize(context_->device->GetMemoryManageStream());
  for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
    size_t accepted_num = accepted_tokens_num_host[sample_idx];
    for (size_t h = 0; h < hidden_size * accepted_num; ++h) {
      size_t in_start_offset = input_token_offsets[sample_idx] * hidden_size;
      size_t ou_start_offset = output_token_offsets[sample_idx] * hidden_size;
      EXPECT_FLOAT_EQ(input_host[in_start_offset + h], output_host[ou_start_offset + h]);
    }
  }

#endif
}

TEST_F(LayerTest, InputRefitLayerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;
  using device_type = half;

  // 测试参数
  const size_t num_tokens = 4;
  const size_t hidden_size = 128;
  const size_t num_embeddings = 2;  // 测试2个embedding的refit

  // 创建 InputRefitLayer
  InputRefitLayer input_refit_layer;
  EXPECT_TRUE(input_refit_layer.Init({}, runtime_config, context_, kDeviceRank).OK());

  // 准备输出张量 - 初始化为全0
  Tensor output_tensor;
  output_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {num_tokens, hidden_size});
  Memset(output_tensor.GetPtr<void>(), 0, output_tensor.GetTotalBytes());

  // 准备位置张量 - 指定每个embedding要写入的位置
  Tensor pos_tensor;
  pos_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {num_embeddings});
  int64_t* pos_ptr = pos_tensor.GetPtr<int64_t>();
  pos_ptr[0] = 1;  // 第一个embedding写入位置1
  pos_ptr[1] = 3;  // 第二个embedding写入位置3

  // 准备embedding张量 - 在host上创建，因为InputRefitLayer会从host拷贝到device
  std::vector<dtype> emb0_host(hidden_size);
  std::vector<dtype> emb1_host(hidden_size);
  std::default_random_engine eng(42);
  std::uniform_real_distribution<float> random_range(-1, 1);

  for (size_t i = 0; i < hidden_size; ++i) {
    emb0_host[i] = static_cast<dtype>(random_range(eng));
    emb1_host[i] = static_cast<dtype>(random_range(eng));
  }

  Tensor emb0_tensor;
  emb0_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_size});
  Memcpy(emb0_tensor.GetPtr<void>(), emb0_host.data(), hidden_size * sizeof(dtype), MEMCPY_HOST_TO_HOST);

  Tensor emb1_tensor;
  emb1_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {hidden_size});
  Memcpy(emb1_tensor.GetPtr<void>(), emb1_host.data(), hidden_size * sizeof(dtype), MEMCPY_HOST_TO_HOST);

  // 构建输入: emb0, emb1, pos
  std::vector<Tensor> input_tensors = {emb0_tensor, emb1_tensor, pos_tensor};
  std::vector<Tensor> output_tensors = {output_tensor};

  // 执行Forward
  EXPECT_TRUE(input_refit_layer.Forward(input_tensors, output_tensors).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // 验证结果
  std::vector<dtype> output_host(num_tokens * hidden_size);
  Memcpy(output_host.data(), output_tensor.GetPtr<void>(), output_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);

  // 验证位置0应该是0（未被写入）
  for (size_t i = 0; i < hidden_size; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(output_host[0 * hidden_size + i]), 0.0f);
  }

  // 验证位置1应该是emb0的值
  for (size_t i = 0; i < hidden_size; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(output_host[1 * hidden_size + i]), static_cast<float>(emb0_host[i]));
  }

  // 验证位置2应该是0（未被写入）
  for (size_t i = 0; i < hidden_size; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(output_host[2 * hidden_size + i]), 0.0f);
  }

  // 验证位置3应该是emb1的值
  for (size_t i = 0; i < hidden_size; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(output_host[3 * hidden_size + i]), static_cast<float>(emb1_host[i]));
  }

  // 测试Clear功能
  input_refit_layer.Clear();
  EXPECT_TRUE(input_refit_layer.cast_cache_.empty());

  // 测试空输入情况（只有pos张量）
  std::vector<Tensor> empty_input_tensors = {pos_tensor};
  EXPECT_TRUE(input_refit_layer.Forward(empty_input_tensors, output_tensors).OK());

  // 测试数据类型转换功能 - 使用FP32输入，FP16输出
  Tensor emb_fp32_tensor;
  emb_fp32_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {hidden_size});
  std::vector<float> emb_fp32_host(hidden_size);
  for (size_t i = 0; i < hidden_size; ++i) {
    emb_fp32_host[i] = random_range(eng);
  }
  Memcpy(emb_fp32_tensor.GetPtr<void>(), emb_fp32_host.data(), hidden_size * sizeof(float), MEMCPY_HOST_TO_HOST);

  // 重置输出张量
  Memset(output_tensor.GetPtr<void>(), 0, output_tensor.GetTotalBytes());

  // 只使用一个embedding进行类型转换测试
  Tensor single_pos_tensor;
  single_pos_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {1});
  single_pos_tensor.GetPtr<int64_t>()[0] = 2;  // 写入位置2

  std::vector<Tensor> cast_input_tensors = {emb_fp32_tensor, single_pos_tensor};
  EXPECT_TRUE(input_refit_layer.Forward(cast_input_tensors, output_tensors).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // 验证类型转换后的结果
  Memcpy(output_host.data(), output_tensor.GetPtr<void>(), output_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  for (size_t i = 0; i < hidden_size; ++i) {
    // FP32转FP16会有精度损失，使用EXPECT_NEAR
    EXPECT_NEAR(static_cast<float>(output_host[2 * hidden_size + i]), emb_fp32_host[i], 1e-2);
  }

  // 验证cast_cache不为空（因为进行了类型转换）
  EXPECT_FALSE(input_refit_layer.cast_cache_.empty());

  KLLM_LOG_INFO << "InputRefitLayerTest passed successfully!";
#else
  GTEST_SKIP() << "InputRefitLayer requires CUDA support";
#endif
}

TEST_F(LayerTest, AllReduceResidualAddNormLayerTest) {
  int device_count = 0;
  GetDeviceCount(&device_count);
  if (device_count < 2) {
    GTEST_SKIP() << "Skip all reduce test since device count less than 2";
  }
  device_count = std::min(2, device_count);
#ifdef ENABLE_CUDA
  using dtype = half_float::half;
  for (int rank = 0; rank < device_count; rank++) {
    int multicast_supported = 0;
    CU_CHECK(cuDeviceGetAttribute(&multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, rank));
    if (!multicast_supported) {
      GTEST_SKIP() << "Skip all reduce test since device does not support multicast";
    }
  }

  ncclUniqueId nccl_uid = GenerateNCCLUniqueID();

  // Collect ipc handles from all ranks
  std::vector<IpcMemHandle> global_ipc_handles(device_count + 1);

  const float rms_norm_eps = 1e-6;
  const size_t token_num = 4;
  const size_t hidden_size = 1024;
  auto RunAllReduce = [&](const int rank) {
    SetDevice(rank);
    std::default_random_engine eng;
    std::uniform_real_distribution<float> random_range(-1, 1);

    auto context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(device_count, 1, 1, Singleton<Environment>::GetInstance()));
    context->global->InitDistributedConfig();

    context->device = std::make_shared<DeviceContext>(device_count);
    context->device->ext->InitNcclParam(nccl_uid);
    // Force multicast for testing
    context->device->ext->is_multicast_supported_ = true;

    Tensor rms_norm_weight(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {hidden_size});
    std::vector<dtype> rms_norm_weight_host(hidden_size);
    for (auto& val : rms_norm_weight_host) {
      val = static_cast<dtype>(random_range(eng));
    }
    MemcpyAsync(rms_norm_weight.GetPtr<void>(), rms_norm_weight_host.data(), rms_norm_weight.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE, context->device->GetPrimaryComputeStream());

    // Use a local instance instead of the global singleton
    NvlsMcastMemory nvls_mcast_memory;
    nvls_mcast_memory.Initialize(device_count);
    // Wait NCCL init
    static_cast<void>(context->device->ext->GetNCCLParam());

    nvls_mcast_memory.AllocMcastMemory(token_num * hidden_size * GetTypeSize(TYPE_FP16));
    Tensor input(MemoryLocation::LOCATION_MULTICAST, TYPE_FP16, {token_num, hidden_size},
                 reinterpret_cast<void*>(nvls_mcast_memory.GetNvlsHandle()->uc_ptr));
    Tensor residual(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {token_num, hidden_size});
    std::vector<dtype> input_host(token_num * hidden_size);
    for (auto& val : input_host) {
      val = static_cast<dtype>(random_range(eng));
    }
    MemcpyAsync(input.GetPtr<void>(), input_host.data(), input.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context->device->GetPrimaryComputeStream());
    std::vector<dtype> residual_host(token_num * hidden_size);
    for (auto& val : residual_host) {
      val = static_cast<dtype>(random_range(eng));
    }
    MemcpyAsync(residual.GetPtr<void>(), residual_host.data(), residual.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context->device->GetPrimaryComputeStream());

    auto& ipc_handles = nvls_mcast_memory.GetIpcHandles();
    EXPECT_TRUE(ipc_handles.size() == static_cast<size_t>(device_count + 1));
    global_ipc_handles[rank].fd = ipc_handles[rank].fd;
    if (rank == 0) {
      global_ipc_handles[device_count].fd = ipc_handles[device_count].fd;
    }
    // Sync after setting global_ipc_handles
    static std::atomic<size_t> wg_ipc_handle{0};
    for (wg_ipc_handle.fetch_add(1, std::memory_order_relaxed);
         wg_ipc_handle.load(std::memory_order_relaxed) % device_count > 0;) {
      std::this_thread::yield();
    }

    for (int peer = 0; peer < device_count; peer++) {
      if (peer != rank) {
        ipc_handles[peer].fd = global_ipc_handles[peer].fd;
      }
    }
    if (rank != 0) {
      ipc_handles[device_count].fd = global_ipc_handles[device_count].fd;
    }
    // Init multicast memory before forwarding
    nvls_mcast_memory.InitMcastMemory();
    // Sync after initializing multicast memory
    static std::atomic<size_t> wg_mcast_init{0};
    for (wg_mcast_init.fetch_add(1, std::memory_order_relaxed);
         wg_mcast_init.load(std::memory_order_relaxed) % device_count > 0;) {
      std::this_thread::yield();
    }

    AllReduceResidualAddNormLayer all_reduce_residual_add_norm_layer;
    EXPECT_TRUE(
        all_reduce_residual_add_norm_layer.Init({rms_norm_eps, rms_norm_weight}, runtime_config, context, rank).OK());
    all_reduce_residual_add_norm_layer.nvls_mcast_memory_ = &nvls_mcast_memory;

    // Reduce + Residual
    std::vector<Tensor> outputs{residual};
    EXPECT_TRUE(all_reduce_residual_add_norm_layer.Forward({input}, outputs).OK());
    // Verify output
    std::vector<dtype> output_host(token_num * hidden_size);
    MemcpyAsync(output_host.data(), outputs[0].GetPtr<void>(), outputs[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context->device->GetPrimaryComputeStream());
    StreamSynchronize(context->device->GetPrimaryComputeStream());
    for (size_t i = 0; i < token_num * hidden_size; i++) {
      EXPECT_NEAR(output_host[i], input_host[i] * device_count + residual_host[i], 1e-3);
    }

    // Reduce + Residual + Norm
    outputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {token_num, hidden_size});
    EXPECT_TRUE(all_reduce_residual_add_norm_layer.Forward({input, residual}, outputs).OK());
    // Validate the first few elements
    MemcpyAsync(output_host.data(), outputs[0].GetPtr<void>(), outputs[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context->device->GetPrimaryComputeStream());
    std::vector<float> output_ref_host{0.58203,  -0.88671, 0.00791,  0.16882,  0.03298,
                                       0.108398, -0.10589, -0.30102, -0.31982, 1.52344};
    StreamSynchronize(context->device->GetPrimaryComputeStream());
    for (size_t i = 0; i < output_ref_host.size(); i++) {
      EXPECT_NEAR(output_host[i], output_ref_host[i], 1e-3);
    }

    nvls_mcast_memory.FreeMcastMemory();
  };

  std::vector<std::unique_ptr<std::thread>> run_threads;
  for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
    run_threads.emplace_back(std::make_unique<std::thread>(RunAllReduce, cur_rank));
  }
  for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
    run_threads[cur_rank]->join();
  }
#endif
}

}  // namespace ksana_llm
