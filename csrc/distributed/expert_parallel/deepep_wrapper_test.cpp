/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

// DeepEP Wrapper 单元测试
// 验证两种模式的 Dispatch + Identity Expert + Combine 端到端正确性：
//   1. Normal 模式 (Intranode) - 2 GPU 节点内 NVLink 通信
//   2. Low Latency 模式 - 2 GPU 低延迟 RDMA/nvshmem 通信
//
// 测试架构：fork() + pipe 进程间通信 + setenv GPU 隔离
// 参考 nccl_data_channel_test.cpp 和 communication_performance_runner_test.cpp 的模式
// 父进程 fork 两个子进程 (rank 0 / rank 1)，通过 pipe 交换 IPC handles，
// 父进程只负责等待子进程完成并检查退出状态。
//
// GPU 隔离策略：
//   每个子进程通过 CUDA_VISIBLE_DEVICES 只看到一张 GPU (device 0)。
//   DeepEPWrapper 的 local_rank_ 始终为 0，避免与可见设备数不匹配。
//   通过 node_rank 参数区分 DeepEP 的通信 rank:
//     process 0: rank=0, node_rank=0 → local_rank_=0, deepep_rank_=0
//     process 1: rank=0, node_rank=1 → local_rank_=0, deepep_rank_=1

#include <gtest/gtest.h>

#ifdef KSANA_WITH_DEEPEP

#  include <cuda_runtime.h>
#  include <dirent.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>

// 声明 GCC 内置的覆盖率数据刷新函数，用于在 _exit() 前手动写入 .gcda 文件。
// _exit() 不会触发 atexit() 回调，导致 gcov 计数器数据丢失。
extern "C" void __gcov_dump(void) __attribute__((weak));


#  include <algorithm>
#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#  include <vector>

#  include "distributed/control_message.h"
#  include "distributed/expert_parallel/deepep_wrapper.h"
#  include "utils/tensor.h"

namespace ksana_llm {

// ============================================================
// Pipe 辅助函数 - 用于进程间交换 IPC handles
// ============================================================

// 通过 pipe 发送固定大小的数据
static bool PipeWrite(int fd, const void* data, size_t size) {
  const char* ptr = static_cast<const char*>(data);
  size_t remaining = size;
  while (remaining > 0) {
    ssize_t written = write(fd, ptr, remaining);
    if (written <= 0) return false;
    ptr += written;
    remaining -= written;
  }
  return true;
}

// 通过 pipe 接收固定大小的数据
static bool PipeRead(int fd, void* data, size_t size) {
  char* ptr = static_cast<char*>(data);
  size_t remaining = size;
  while (remaining > 0) {
    ssize_t bytes_read = read(fd, ptr, remaining);
    if (bytes_read <= 0) return false;
    ptr += bytes_read;
    remaining -= bytes_read;
  }
  return true;
}

// ============================================================
// BF16 <-> Float 转换辅助
// ============================================================
static uint16_t FloatToBF16(float f) {
  uint32_t val;
  memcpy(&val, &f, sizeof(float));
  // 简单截断 (不做 round-to-nearest)
  return static_cast<uint16_t>(val >> 16);
}

static float BF16ToFloat(uint16_t bf16) {
  uint32_t val = static_cast<uint32_t>(bf16) << 16;
  float f;
  memcpy(&f, &val, sizeof(float));
  return f;
}

// ============================================================
// GPU Tensor 辅助函数
// ============================================================

// 计算 DataType 的字节大小
static size_t GetDTypeSizeHelper(DataType dtype) {
  switch (dtype) {
    case TYPE_BF16:
      return 2;
    case TYPE_FP16:
      return 2;
    case TYPE_FP32:
      return 4;
    case TYPE_INT32:
      return 4;
    case TYPE_INT64:
      return 8;
    case TYPE_FP8_E4M3:
      return 1;
    default:
      return 2;
  }
}

// 分配 GPU 内存并用 host 数据初始化，返回 Tensor (shared buffer 模式，不自动释放)
static Tensor AllocateGPUTensor(DataType dtype, const std::vector<size_t>& shape, const void* host_data,
                                cudaStream_t stream) {
  size_t num_elements = 1;
  for (auto s : shape) {
    num_elements *= s;
  }
  size_t total_bytes = num_elements * GetDTypeSizeHelper(dtype);

  void* device_ptr = nullptr;
  cudaMalloc(&device_ptr, total_bytes);

  if (host_data) {
    cudaMemcpyAsync(device_ptr, host_data, total_bytes, cudaMemcpyHostToDevice, stream);
  } else {
    cudaMemsetAsync(device_ptr, 0, total_bytes, stream);
  }

  // 使用 data_ptr 构造 Tensor，is_shared_buffer_ = true，析构时不释放
  return Tensor(LOCATION_DEVICE, dtype, shape, device_ptr);
}

// 手动释放 GPU Tensor 的内存
static void FreeGPUTensor(Tensor& tensor) {
  void* ptr = tensor.GetPtr<void>(/*check_empty=*/false);
  if (ptr) {
    cudaFree(ptr);
  }
}

// 将 GPU BF16 数据拷贝到 host float 数组
static std::vector<float> CopyBF16ToHostFloat(const Tensor& tensor, size_t num_elements, cudaStream_t stream) {
  std::vector<uint16_t> host_bf16(num_elements);
  cudaMemcpyAsync(host_bf16.data(), tensor.GetPtr<void>(), num_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);

  std::vector<float> host_float(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    host_float[i] = BF16ToFloat(host_bf16[i]);
  }
  return host_float;
}

// ============================================================
// Normal 模式测试 - 单个 rank 的执行逻辑
//
// 参数:
//   deep_rank: DeepEP 通信 rank (0 或 1)
//   node_rank: 节点编号，用于 DeepEPWrapper 构造 (等于 deep_rank)
//   pipe_read_fd: 从对端读取 config 的 fd
//   pipe_write_fd: 向对端写入 config 的 fd
// ============================================================
static void RunNormalModeTest(int deep_rank, int node_rank, int pipe_read_fd, int pipe_write_fd) {
  constexpr int kNumRanks = 2;
  constexpr int kWorldSize = 2;
  constexpr int kNumTokens = 256;
  constexpr int kHiddenSize = 512;
  constexpr int kNumExperts = 8;
  constexpr int kNumTopk = 2;
  constexpr float kTolerance = 1e-2f;  // BF16 精度容差

  // 1. CUDA_VISIBLE_DEVICES 已在父进程中为每个子进程隔离到单卡，
  //    子进程内始终使用 device 0。
  cudaSetDevice(0);
  cudaStream_t cuda_stream;
  cudaStreamCreate(&cuda_stream);

  // 2. 构造 DeepEPWrapper (Normal 模式)
  //    rank=0 (local_rank_=0，匹配唯一可见的 device 0)
  //    node_rank 用于区分 DeepEP 通信 rank:
  //      node_rank=0 → deepep_rank_ = 0 + 0*2/2 = 0
  //      node_rank=1 → deepep_rank_ = 0 + 1*2/2 = 1
  DeepEPWrapper wrapper(/*rank=*/0, kNumRanks, kWorldSize, node_rank, kHiddenSize, kNumExperts,
                        /*low_latency_mode=*/false,
                        /*max_token_num=*/0);

  // 3. 通过 pipe 交换 IPC handles (仿照 nccl_data_channel_test.cpp 的 pipe 通信模式)
  EPCommunicationConfig my_config = wrapper.GetCommunicationConfig();
  EPCommunicationConfig peer_config;

  // 向对端发送自己的 config
  if (!PipeWrite(pipe_write_fd, &my_config, sizeof(my_config))) {
    fprintf(stderr, "[Rank %d] Failed to write config to pipe\n", deep_rank);
    if (__gcov_dump) __gcov_dump();
    _exit(1);
  }
  close(pipe_write_fd);

  // 从对端接收 config (pipe 的阻塞读天然实现同步)
  if (!PipeRead(pipe_read_fd, &peer_config, sizeof(peer_config))) {
    fprintf(stderr, "[Rank %d] Failed to read config from pipe\n", deep_rank);
    if (__gcov_dump) __gcov_dump();
    _exit(1);
  }
  close(pipe_read_fd);

  // 4. Initialize with all configs (按 deep_rank 顺序排列)
  std::vector<EPCommunicationConfig> all_configs(2);
  all_configs[deep_rank] = my_config;
  all_configs[1 - deep_rank] = peer_config;
  wrapper.Initialize(all_configs);

  // 5. 准备有意义的测试数据
  //
  // hidden_states: (num_tokens, hidden_size) BF16
  //   每个 token 填充不同的值: hidden[i][j] = bf16(i + 1.0)
  //   token 0 全部 1.0, token 1 全部 2.0, ..., token 255 全部 256.0
  //   这样每个 token 有唯一可辨识的数据指纹，可追踪跨 GPU 传输是否正确
  std::vector<uint16_t> hidden_host(kNumTokens * kHiddenSize);
  for (int i = 0; i < kNumTokens; ++i) {
    uint16_t val = FloatToBF16(static_cast<float>(i + 1));
    for (int j = 0; j < kHiddenSize; ++j) {
      hidden_host[i * kHiddenSize + j] = val;
    }
  }

  // topk_ids: (num_tokens, num_topk) INT32
  // 关键：确保每个 token 的两个 topk 选择分别在不同 rank 上。
  //   rank 0 拥有 experts 0..3, rank 1 拥有 experts 4..7
  //   这样每个 token 必须经过跨 GPU 通信才能完成 dispatch + combine
  //   如果两个 topk 都在同一 rank，该 token 不需要跨 GPU 通信，无法验证数据传输正确性
  constexpr int kExpertsPerRank = kNumExperts / kNumRanks;
  std::vector<int32_t> topk_ids_host(kNumTokens * kNumTopk);
  srand(42 + deep_rank);
  for (int i = 0; i < kNumTokens; ++i) {
    // topk[0] 选 rank 0 的 expert (0..3)
    topk_ids_host[i * kNumTopk + 0] = rand() % kExpertsPerRank;
    // topk[1] 选 rank 1 的 expert (4..7)
    topk_ids_host[i * kNumTopk + 1] = kExpertsPerRank + (rand() % kExpertsPerRank);
  }

  // topk_weights: (num_tokens, num_topk) FP32, 全部填充 1.0
  std::vector<float> topk_weights_host(kNumTokens * kNumTopk, 1.0f);

  // 分配 GPU tensors
  Tensor hidden_tensor = AllocateGPUTensor(TYPE_BF16, {kNumTokens, kHiddenSize}, hidden_host.data(), cuda_stream);

  Tensor topk_ids_tensor = AllocateGPUTensor(TYPE_INT32, {kNumTokens, kNumTopk}, topk_ids_host.data(), cuda_stream);

  Tensor topk_weights_tensor =
      AllocateGPUTensor(TYPE_FP32, {kNumTokens, kNumTopk}, topk_weights_host.data(), cuda_stream);

  // output buffer: 预分配足够大的空间
  // Normal 模式最坏情况: 所有 token 的所有 topk 都路由到当前 rank
  size_t max_recv = static_cast<size_t>(kNumTokens) * kNumRanks * kNumTopk;
  Tensor output_placeholder = AllocateGPUTensor(TYPE_BF16, {1, static_cast<size_t>(kHiddenSize)}, nullptr, cuda_stream);
  Tensor dispatched_tensor =
      AllocateGPUTensor(TYPE_BF16, {max_recv, static_cast<size_t>(kHiddenSize)}, nullptr, cuda_stream);

  cudaStreamSynchronize(cuda_stream);

  // 6. Dispatch
  std::vector<Tensor> dispatch_inputs = {hidden_tensor, topk_ids_tensor, topk_weights_tensor};
  std::vector<Tensor> dispatch_outputs = {output_placeholder, dispatched_tensor};

  Stream stream;
  wrapper.Dispatch(dispatch_inputs, dispatch_outputs, stream);

  cudaStreamSynchronize(stream.Get());
  cudaStreamSynchronize(cuda_stream);

  int recv_num_tokens = wrapper.GetRecvNumTokens();

  // 7. 模拟专家计算 (identity): expert_output = dispatched_hidden_states
  Tensor expert_output(LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(recv_num_tokens), static_cast<size_t>(kHiddenSize)},
                       dispatch_outputs[1].GetPtr<void>());

  // 8. Combine
  Tensor combined_tensor =
      AllocateGPUTensor(TYPE_BF16, {kNumTokens, static_cast<size_t>(kHiddenSize)}, nullptr, cuda_stream);

  std::vector<Tensor> combine_inputs = {expert_output};
  std::vector<Tensor> combine_outputs = {combined_tensor};

  wrapper.Combine(combine_inputs, combine_outputs, stream);

  cudaStreamSynchronize(stream.Get());
  cudaStreamSynchronize(cuda_stream);

  // 9. 正确性验证
  //
  // Normal 模式 combine 传递 topk_weights=nullopt，不做加权，直接将各 expert 输出求和。
  // 经过 dispatch → identity expert → combine:
  //   combined[i][j] = sum_{k=0}^{topk-1} hidden[i][j] = num_topk * hidden[i][j]
  //
  // 因此 expected[i] = num_topk * (i + 1.0)
  //   token 0 → 2.0,  token 1 → 4.0,  ...,  token 255 → 512.0
  //
  // 如果跨 GPU 通信失败:
  //   - 数据丢失: combined[i] ≈ (i+1) 而非 2*(i+1)（只收到本 rank 的一份）
  //   - 数据错位: combined[i] = 2*(j+1)（j≠i，收到了错误 token 的数据）
  //   - 数据损坏: combined[i] 为随机值
  auto combined_host = CopyBF16ToHostFloat(combined_tensor, kNumTokens * kHiddenSize, cuda_stream);

  fprintf(stdout, "[Rank %d] Normal mode: recv_num_tokens=%d (expected=%d), verifying %d tokens...\n", deep_rank,
          recv_num_tokens, kNumTokens, kNumTokens);

  int num_errors = 0;
  float max_diff = 0.0f;
  constexpr int kMaxReportedErrors = 5;
  for (int i = 0; i < kNumTokens; ++i) {
    float hidden_val = static_cast<float>(i + 1);
    float expected = static_cast<float>(kNumTopk) * hidden_val;
    // BF16 容差: 1% 相对误差或 0.1 绝对误差
    float tolerance = std::max(std::abs(expected) * 0.01f, 0.1f);
    for (int j = 0; j < kHiddenSize; ++j) {
      float actual = combined_host[i * kHiddenSize + j];
      float diff = std::abs(actual - expected);
      max_diff = std::max(max_diff, diff);
      if (diff > tolerance) {
        if (num_errors < kMaxReportedErrors) {
          fprintf(stderr,
                  "[Rank %d] MISMATCH token=%d dim=%d: expected=%.2f (hidden=%.1f * topk=%d), "
                  "actual=%.2f, diff=%.4f\n",
                  deep_rank, i, j, expected, hidden_val, kNumTopk, actual, diff);
        }
        num_errors++;
      }
    }
  }

  fprintf(stdout, "[Rank %d] Normal mode: %d/%d elements OK, %d errors, max_diff=%.4f\n", deep_rank,
          kNumTokens * kHiddenSize - num_errors, kNumTokens * kHiddenSize, num_errors, max_diff);

  // 10. 清理
  stream.Destroy();

  if (num_errors > 0) {
    fprintf(stderr, "[Rank %d] Normal mode FAILED: %d elements mismatched\n", deep_rank, num_errors);
    cudaStreamDestroy(cuda_stream);
    if (__gcov_dump) __gcov_dump();
    _exit(1);
  }

  cudaStreamDestroy(cuda_stream);
  // 子进程成功后必须直接 _exit(0)，避免触发 DeepEPWrapper 析构中的
  // nvshmem_finalize() 导致 IBGDA transport 堆损坏。参见 LowLatency 测试同样处理。
  // 在 _exit() 前手动刷新 gcov 数据，否则子进程的覆盖率计数器会丢失。
  if (__gcov_dump) __gcov_dump();
  _exit(0);
}

// ============================================================
// Low Latency 模式测试 - 单个 rank 的执行逻辑
// ============================================================
static void RunLowLatencyModeTest(int deep_rank, int node_rank, int pipe_read_fd, int pipe_write_fd) {
  constexpr int kNumRanks = 2;
  constexpr int kWorldSize = 2;
  constexpr int kNumTokens = 128;
  // DeepEP low-latency kernel (SWITCH_HIDDEN macro) 仅支持: 2560, 4096, 5120, 7168
  constexpr int kHiddenSize = 5120;
  constexpr int kNumExperts = 8;
  constexpr int kNumTopk = 2;
  constexpr int kMaxTokenNum = 128;
  constexpr int kExpertsPerNode = kNumExperts / kNumRanks;
  constexpr float kTolerance = 1e-2f;

  // 1. 子进程内始终使用 device 0 (CUDA_VISIBLE_DEVICES 已隔离到单卡)
  cudaSetDevice(0);
  cudaStream_t cuda_stream;
  cudaStreamCreate(&cuda_stream);

  // 2. 构造 DeepEPWrapper (Low Latency 模式)
  DeepEPWrapper wrapper(/*rank=*/0, kNumRanks, kWorldSize, node_rank, kHiddenSize, kNumExperts,
                        /*low_latency_mode=*/true,
                        /*max_token_num=*/kMaxTokenNum);

  // 3. 通过 pipe 交换 IPC handles + nvshmem unique id
  EPCommunicationConfig my_config = wrapper.GetCommunicationConfig();
  EPCommunicationConfig peer_config;

  if (!PipeWrite(pipe_write_fd, &my_config, sizeof(my_config))) {
    fprintf(stderr, "[Rank %d] Failed to write config to pipe\n", deep_rank);
    if (__gcov_dump) __gcov_dump();
    _exit(1);
  }
  close(pipe_write_fd);

  if (!PipeRead(pipe_read_fd, &peer_config, sizeof(peer_config))) {
    fprintf(stderr, "[Rank %d] Failed to read config from pipe\n", deep_rank);
    if (__gcov_dump) __gcov_dump();
    _exit(1);
  }
  close(pipe_read_fd);

  // 4. Initialize
  std::vector<EPCommunicationConfig> all_configs(2);
  all_configs[deep_rank] = my_config;
  all_configs[1 - deep_rank] = peer_config;
  wrapper.Initialize(all_configs);

  // 5. 准备有意义的测试数据
  //
  // hidden_states: 每个 token 填充不同的值: hidden[i][j] = bf16(i + 1.0)
  //   token 0 全部 1.0, token 1 全部 2.0, ..., token 127 全部 128.0
  std::vector<uint16_t> hidden_host(kNumTokens * kHiddenSize);
  for (int i = 0; i < kNumTokens; ++i) {
    uint16_t val = FloatToBF16(static_cast<float>(i + 1));
    for (int j = 0; j < kHiddenSize; ++j) {
      hidden_host[i * kHiddenSize + j] = val;
    }
  }

  // topk_ids: 确保跨 rank 分发
  std::vector<int32_t> topk_ids_host(kNumTokens * kNumTopk);
  srand(42 + deep_rank);
  for (int i = 0; i < kNumTokens; ++i) {
    topk_ids_host[i * kNumTopk + 0] = rand() % kExpertsPerNode;
    topk_ids_host[i * kNumTopk + 1] = kExpertsPerNode + (rand() % kExpertsPerNode);
  }

  // topk_weights: 使用非平凡权重 (1.0, 0.5)，验证 combine 的加权求和功能
  //   Low Latency combine 使用 topk_weights 做加权:
  //   combined[i] = hidden[i] * w[0] + hidden[i] * w[1] = hidden[i] * (1.0 + 0.5) = hidden[i] * 1.5
  constexpr float kWeight0 = 1.0f;
  constexpr float kWeight1 = 0.5f;
  constexpr float kWeightSum = kWeight0 + kWeight1;  // 1.5
  std::vector<float> topk_weights_host(kNumTokens * kNumTopk);
  for (int i = 0; i < kNumTokens; ++i) {
    topk_weights_host[i * kNumTopk + 0] = kWeight0;
    topk_weights_host[i * kNumTopk + 1] = kWeight1;
  }

  Tensor hidden_tensor = AllocateGPUTensor(TYPE_BF16, {kNumTokens, kHiddenSize}, hidden_host.data(), cuda_stream);

  Tensor topk_ids_tensor = AllocateGPUTensor(TYPE_INT32, {kNumTokens, kNumTopk}, topk_ids_host.data(), cuda_stream);

  Tensor topk_weights_tensor =
      AllocateGPUTensor(TYPE_FP32, {kNumTokens, kNumTopk}, topk_weights_host.data(), cuda_stream);

  // Low latency output: 3D shape [experts_per_node, max_tokens * num_ranks, hidden_size]
  size_t ll_dim0 = static_cast<size_t>(kExpertsPerNode);
  size_t ll_dim1 = static_cast<size_t>(kMaxTokenNum) * kNumRanks;
  size_t ll_dim2 = static_cast<size_t>(kHiddenSize);

  Tensor output_placeholder = AllocateGPUTensor(TYPE_BF16, {1, static_cast<size_t>(kHiddenSize)}, nullptr, cuda_stream);
  Tensor dispatched_tensor = AllocateGPUTensor(TYPE_BF16, {ll_dim0 * ll_dim1, ll_dim2}, nullptr, cuda_stream);

  Tensor recv_count_tensor =
      AllocateGPUTensor(TYPE_INT32, {static_cast<size_t>(kExpertsPerNode)}, nullptr, cuda_stream);

  cudaStreamSynchronize(cuda_stream);

  // 6. Dispatch
  std::vector<Tensor> dispatch_inputs = {hidden_tensor, topk_ids_tensor, topk_weights_tensor};
  std::vector<Tensor> dispatch_outputs = {output_placeholder, dispatched_tensor, recv_count_tensor};

  Stream stream;
  wrapper.Dispatch(dispatch_inputs, dispatch_outputs, stream);

  cudaStreamSynchronize(stream.Get());
  cudaStreamSynchronize(cuda_stream);

  // 7. 模拟专家计算 (identity): expert_output = dispatched_hidden_states
  Tensor expert_output(LOCATION_DEVICE, TYPE_BF16, dispatch_outputs[1].shape, dispatch_outputs[1].GetPtr<void>());

  // 8. Combine
  Tensor combined_tensor =
      AllocateGPUTensor(TYPE_BF16, {kNumTokens, static_cast<size_t>(kHiddenSize)}, nullptr, cuda_stream);

  std::vector<Tensor> combine_inputs = {expert_output};
  std::vector<Tensor> combine_outputs = {combined_tensor};

  wrapper.Combine(combine_inputs, combine_outputs, stream);

  cudaStreamSynchronize(stream.Get());
  cudaStreamSynchronize(cuda_stream);

  // 9. 正确性验证
  //
  // Low Latency 模式 combine 使用 topk_weights 做加权求和:
  //   combined[i][j] = sum_{k} hidden[i][j] * topk_weights[i][k]
  //                  = hidden[i][j] * (w[0] + w[1])
  //                  = (i + 1.0) * kWeightSum
  //
  //   token 0 → 1.5,  token 1 → 3.0,  ...,  token 127 → 192.0
  //
  // 与 Normal 模式的区别:
  //   Normal: combine 不加权 (topk_weights=nullopt)，结果 = hidden * num_topk
  //   Low Latency: combine 加权，结果 = hidden * sum(weights)
  auto combined_host = CopyBF16ToHostFloat(combined_tensor, kNumTokens * kHiddenSize, cuda_stream);

  fprintf(stdout, "[Rank %d] LowLatency mode: weights=(%.1f, %.1f), verifying %d tokens...\n", deep_rank, kWeight0,
          kWeight1, kNumTokens);

  int num_errors = 0;
  float max_diff = 0.0f;
  constexpr int kMaxReportedErrors = 5;
  for (int i = 0; i < kNumTokens; ++i) {
    float hidden_val = static_cast<float>(i + 1);
    float expected = hidden_val * kWeightSum;
    // BF16 容差: 1% 相对误差或 0.1 绝对误差
    float tolerance = std::max(std::abs(expected) * 0.01f, 0.1f);
    for (int j = 0; j < kHiddenSize; ++j) {
      float actual = combined_host[i * kHiddenSize + j];
      float diff = std::abs(actual - expected);
      max_diff = std::max(max_diff, diff);
      if (diff > tolerance) {
        if (num_errors < kMaxReportedErrors) {
          fprintf(stderr,
                  "[Rank %d] MISMATCH token=%d dim=%d: expected=%.2f (hidden=%.1f * wsum=%.1f), "
                  "actual=%.2f, diff=%.4f\n",
                  deep_rank, i, j, expected, hidden_val, kWeightSum, actual, diff);
        }
        num_errors++;
      }
    }
  }

  fprintf(stdout, "[Rank %d] LowLatency mode: %d/%d elements OK, %d errors, max_diff=%.4f\n", deep_rank,
          kNumTokens * kHiddenSize - num_errors, kNumTokens * kHiddenSize, num_errors, max_diff);

  stream.Destroy();

  if (num_errors > 0) {
    fprintf(stderr, "[Rank %d] LowLatency mode FAILED: %d elements mismatched\n", deep_rank, num_errors);
    cudaStreamDestroy(cuda_stream);
    if (__gcov_dump) __gcov_dump();
    _exit(1);
  }

  cudaStreamDestroy(cuda_stream);
  // 子进程成功后必须直接 _exit(0)，不能正常 return。
  // 因为 wrapper 是栈上局部变量，return 会触发析构 → Buffer::~Buffer()
  // → nvshmem_finalize() → dlclose(nvshmem_transport_ibgda.so) → 全局析构器
  // 中 free() 检测到堆损坏而 SIGABRT（nvshmem IBGDA transport 的已知问题）。
  // _exit() 跳过析构和 atexit 处理，直接退出子进程
  // 在 _exit() 前手动刷新 gcov 数据，否则子进程的覆盖率计数器会丢失。
  if (__gcov_dump) __gcov_dump();
  _exit(0);
}

// ============================================================
// GTest Fixture
// ============================================================
class DeepEPWrapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 检查 GPU 数量，不足 2 卡则跳过
    // 注意：不能在 fork() 之前调用任何 CUDA API (如 cudaGetDeviceCount)，
    // 因为这会初始化父进程的 CUDA 上下文，fork 后子进程继承的 CUDA 上下文是无效的。
    // 这里通过读取 /proc/driver/nvidia/gpus/ 目录来检查 GPU 数量，避免初始化 CUDA。
    int device_count = CountNvidiaGPUs();
    if (device_count < 2) {
      GTEST_SKIP() << "Need at least 2 GPUs, found " << device_count;
    }

    // 读取环境变量中的可用设备列表，默认 "0,1"
    // 仿照 nccl_data_channel_test.cpp 通过 CUDA_VISIBLE_DEVICES 隔离 GPU
    const char* all_devices = getenv("CUDA_VISIBLE_DEVICES");
    if (all_devices != nullptr) {
      std::string devs(all_devices);
      size_t pos = devs.find(',');
      if (pos != std::string::npos) {
        device_ids_[0] = devs.substr(0, pos);
        device_ids_[1] = devs.substr(pos + 1);
      }
    }
  }

  // 不通过 CUDA API 检查 GPU 数量，避免在 fork 前初始化 CUDA 上下文
  static int CountNvidiaGPUs() {
    // 方法1: 通过 nvidia-smi 查询
    FILE* pipe = popen("nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1", "r");
    if (pipe) {
      char buffer[128];
      if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        int count = atoi(buffer);
        pclose(pipe);
        if (count > 0) return count;
      } else {
        pclose(pipe);
      }
    }
    // 方法2: 通过 /proc 文件系统
    int count = 0;
    DIR* dir = opendir("/proc/driver/nvidia/gpus");
    if (dir) {
      struct dirent* entry;
      while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] != '.') count++;
      }
      closedir(dir);
      if (count > 0) return count;
    }
    // 方法3: 回退到 CUDA API (如果上面的方法都失败了)
    cudaGetDeviceCount(&count);
    return count;
  }

  // 通用的双进程 fork + pipe 交换 + 等待逻辑
  //
  // test_fn 签名: void(int deep_rank, int node_rank, int pipe_read_fd, int pipe_write_fd)
  //   deep_rank:     DeepEP 通信 rank (0 或 1)
  //   node_rank:     传给 DeepEPWrapper 的 node_rank (等于 deep_rank)
  //   pipe_read_fd:  从对端读取 EPCommunicationConfig 的 fd
  //   pipe_write_fd: 向对端写入 EPCommunicationConfig 的 fd
  using TestFunction = void (*)(int deep_rank, int node_rank, int pipe_read_fd, int pipe_write_fd);

  void RunMultiProcessTest(TestFunction test_fn) {
    // 创建两个 pipe: pipe_0to1 (rank0 写 -> rank1 读), pipe_1to0 (rank1 写 -> rank0 读)
    int pipe_0to1[2];  // [0]=read, [1]=write
    int pipe_1to0[2];
    ASSERT_EQ(pipe(pipe_0to1), 0) << "Failed to create pipe_0to1";
    ASSERT_EQ(pipe(pipe_1to0), 0) << "Failed to create pipe_1to0";

    // Fork rank 0 子进程
    pid_t pid_rank0 = fork();
    ASSERT_GE(pid_rank0, 0) << "Failed to fork rank 0 process";

    if (pid_rank0 == 0) {
      // Rank 0 子进程
      close(pipe_0to1[0]);
      close(pipe_1to0[1]);

      // 通过 CUDA_VISIBLE_DEVICES 隔离到单卡，确保子进程只看到 device 0
      setenv("CUDA_VISIBLE_DEVICES", device_ids_[0].c_str(), 1);

      test_fn(/*deep_rank=*/0, /*node_rank=*/0, /*pipe_read_fd=*/pipe_1to0[0], /*pipe_write_fd=*/pipe_0to1[1]);
      fflush(stdout);
      fflush(stderr);
      if (__gcov_dump) __gcov_dump();
      _exit(0);
    }

    // Fork rank 1 子进程
    pid_t pid_rank1 = fork();
    ASSERT_GE(pid_rank1, 0) << "Failed to fork rank 1 process";

    if (pid_rank1 == 0) {
      // Rank 1 子进程
      close(pipe_0to1[1]);
      close(pipe_1to0[0]);

      // 通过 CUDA_VISIBLE_DEVICES 隔离到单卡
      setenv("CUDA_VISIBLE_DEVICES", device_ids_[1].c_str(), 1);

      test_fn(/*deep_rank=*/1, /*node_rank=*/1, /*pipe_read_fd=*/pipe_0to1[0], /*pipe_write_fd=*/pipe_1to0[1]);
      fflush(stdout);
      fflush(stderr);
      if (__gcov_dump) __gcov_dump();
      _exit(0);
    }

    // 父进程: 关闭所有 pipe fd，只负责等待子进程
    close(pipe_0to1[0]);
    close(pipe_0to1[1]);
    close(pipe_1to0[0]);
    close(pipe_1to0[1]);

    // 等待两个子进程完成
    int status_rank0, status_rank1;
    waitpid(pid_rank0, &status_rank0, 0);
    waitpid(pid_rank1, &status_rank1, 0);

    bool rank0_success = WIFEXITED(status_rank0) && WEXITSTATUS(status_rank0) == 0;
    bool rank1_success = WIFEXITED(status_rank1) && WEXITSTATUS(status_rank1) == 0;

    if (!rank0_success && !WIFEXITED(status_rank0)) {
      fprintf(stderr, "[Rank 0] terminated abnormally (signal %d)\n",
              WIFSIGNALED(status_rank0) ? WTERMSIG(status_rank0) : -1);
    }
    if (!rank1_success && !WIFEXITED(status_rank1)) {
      fprintf(stderr, "[Rank 1] terminated abnormally (signal %d)\n",
              WIFSIGNALED(status_rank1) ? WTERMSIG(status_rank1) : -1);
    }

    EXPECT_TRUE(rank0_success) << "Rank 0 process failed with status: "
                               << (WIFEXITED(status_rank0) ? WEXITSTATUS(status_rank0) : -1);
    EXPECT_TRUE(rank1_success) << "Rank 1 process failed with status: "
                               << (WIFEXITED(status_rank1) ? WEXITSTATUS(status_rank1) : -1);
  }

  std::string device_ids_[2] = {"0", "1"};
};

// ============================================================
// 测试用例
// ============================================================

// Normal 模式: 2 卡 Intranode Dispatch + Identity Expert + Combine
TEST_F(DeepEPWrapperTest, NormalModeIntraNode) { RunMultiProcessTest(RunNormalModeTest); }

// Low Latency 模式: 2 卡 Low Latency Dispatch + Identity Expert + Combine
TEST_F(DeepEPWrapperTest, LowLatencyMode) { RunMultiProcessTest(RunLowLatencyModeTest); }

}  // namespace ksana_llm

#else  // !KSANA_WITH_DEEPEP

// 当 DeepEP 不可用时，提供一个跳过的测试
TEST(DeepEPWrapperTest, SkippedWithoutDeepEP) {
  GTEST_SKIP() << "DeepEP is not available (KSANA_WITH_DEEPEP not defined)";
}

#endif  // KSANA_WITH_DEEPEP

// 自定义 main(): 确保 PyTorch 使用 native CUDA 分配器而非 CudaMallocAsync。
// CudaMallocAsync 分配器在 fork() 后子进程中执行 cudaFreeAsync 会返回
// "CUDA error: invalid argument"。libtorch_cuda.so 在库加载阶段（main() 之前）
// 缓存 PYTORCH_CUDA_ALLOC_CONF，因此 setenv() 无法在运行时生效。
// 解决方案：设置环境变量后 re-exec 进程，使新加载的 libtorch_cuda.so 读到 backend:native。
int main(int argc, char** argv) {
#ifdef KSANA_WITH_DEEPEP
  const char* alloc_conf = getenv("PYTORCH_CUDA_ALLOC_CONF");
  if (alloc_conf == nullptr || strstr(alloc_conf, "backend:native") == nullptr) {
    setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:native", 1);
    execv("/proc/self/exe", argv);
    perror("execv failed");
    return 1;
  }
#endif
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
