/*
 * Adapted from
 * https://github.com/vllm-project/vllm/blob/v0.3.1/csrc/custom_all_reduce_test.cu
 * Copyright (c) 2024, Tencent Inc.
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <curand_kernel.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <atomic>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_profiler_api.h>
#include <nccl.h>

#include "csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

#define NCCLCHECK(cmd)                                                                      \
  do {                                                                                      \
    ncclResult_t r = cmd;                                                                   \
    if (r != ncclSuccess) {                                                                 \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

__global__ void dummy_kernel() {
  for (int i = 0; i < 100; i++) __nanosleep(1000000);  // 100ms
}

class LlamaNvidiaCustomAllReduceTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const std::vector<std::pair<int, int>> m_n_pairs = {{2, 4096}};

 protected:
  template <typename T>
  void Run(int myRank, int nRanks, ncclComm_t &comm, int data_size, void **metas, void **data_handles,
           void **input_handles, std::atomic<int> &counter) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }
    cudaStream_t stream;
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    BufferMeta result_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(data_size)}, false);

    T *result = static_cast<T *>(result_meta.data_ptr);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemset(result, 0, data_size * sizeof(T)));

    size_t buffer_size = data_size * sizeof(T);
    BufferMeta buffer_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {buffer_size / sizeof(T)}, false);
    data_handles[myRank] = (char *)buffer_meta.data_ptr;

    BufferMeta meta_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {sizeof(llm_kernels::nvidia::Metadata) / sizeof(T)}, false);
    metas[myRank] = meta_meta.data_ptr;

    BufferMeta self_data_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(data_size)}, true);
    T *self_data = static_cast<T *>(self_data_meta.data_ptr);
    input_handles[myRank] = self_data;

    counter++;
    while (counter != 2)
      ;
    CHECK_NVIDIA_CUDA_ERROR(cudaMemset(data_handles[myRank], 0, buffer_size));

    BufferMeta self_data_copy_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(data_size)}, false);
    T *self_data_copy = static_cast<T *>(self_data_copy_meta.data_ptr);

    size_t rank_data_sz = 8 * 1024 * 1024;
    BufferMeta rank_data_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(rank_data_sz / sizeof(T))}, false);
    void *rank_data = rank_data_meta.data_ptr;

    std::vector<int64_t> offsets(nRanks, 0);
    llm_kernels::nvidia::CustomAllreduce fa(metas, rank_data, rank_data_sz, data_handles, offsets, myRank);
    // hack data_handles[myRank] registration
    {
      std::vector<std::string> handles;
      handles.reserve(nRanks);
      for (int i = 0; i < nRanks; i++) {
        char *begin = (char *)&input_handles[i];
        char *end = (char *)&input_handles[i + 1];
        handles.emplace_back(begin, end);
      }
      std::vector<int64_t> offsets(nRanks, 0);
      fa.RegisterBuffer(handles, offsets, self_data, stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpyAsync(self_data_copy, self_data, data_size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    cudaEvent_t start, stop;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&stop));

    constexpr int warmup_iters = 10;
    constexpr int num_iters = 25;

    ncclDataType_t ncclDtype;
    if (std::is_same<T, half>::value) {
      ncclDtype = ncclFloat16;
    } else if (std::is_same<T, nv_bfloat16>::value) {
      ncclDtype = ncclBfloat16;
    } else {
      ncclDtype = ncclFloat;
    }

    dummy_kernel<<<1, 1, 0, stream>>>();
    // warmup
    for (int i = 0; i < warmup_iters; i++) {
      NCCLCHECK(ncclAllReduce(self_data, self_data_copy, data_size, ncclDtype, ncclSum, comm, stream));
    }
    float allreduce_ms = 0;
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i = 0; i < num_iters; i++) {
      NCCLCHECK(ncclAllReduce(self_data, self_data_copy, data_size, ncclDtype, ncclSum, comm, stream));
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    cudaEventElapsedTime(&allreduce_ms, start, stop);

    dummy_kernel<<<1, 1, 0, stream>>>();
    // warm up
    for (int i = 0; i < warmup_iters; i++) {
      fa.AllReduce<T>(stream, self_data, result, data_size);
    }
    float duration_ms = 0;
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i = 0; i < num_iters; i++) {
      fa.AllReduce<T>(stream, self_data, result, data_size);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    cudaEventElapsedTime(&duration_ms, start, stop);

    if (myRank == 0)
      printf(
          "Rank %d done, nGPUs:%d, sz (kb), %ld, my time,%.2f,us, nccl "
          "time,%.2f,us\n",
          myRank, nRanks, data_size * sizeof(T) / 1024, duration_ms * 1e3 / num_iters, allreduce_ms * 1e3 / num_iters);

    // And wait for all the queued up work to complete
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    NCCLCHECK(ncclAllReduce(self_data, self_data_copy, data_size, ncclDtype, ncclSum, comm, stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    EXPECT_TRUE(CheckResult<T>("custom_all_reduce_" + type_str + "_size_" + std::to_string(data_size * sizeof(T)),
                               self_data_copy_meta, result_meta, 1e-5f, 1e-5f));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
    DeleteBuffer(result_meta);
    DeleteBuffer(self_data_copy_meta);
    DeleteBuffer(rank_data_meta);
    DeleteBuffer(buffer_meta);
    DeleteBuffer(meta_meta);
    DeleteBuffer(self_data_meta);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(stream));
    counter = 0;
    while (counter != 0)
      ;
  }

  template <typename T>
  void RunThread(int myRank, int nRanks, ncclUniqueId nccl_id, void **metas, void **data_handles, void **input_handles,
                 std::atomic<int> &counter) {
    CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(myRank));
    for (int i = 0; i < nRanks; i++) {
      if (i != myRank) {
        auto err = cudaDeviceEnablePeerAccess(i, 0);
        if (err != cudaErrorPeerAccessAlreadyEnabled) {
          CHECK_NVIDIA_CUDA_ERROR(err);
        }
      }
    }
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, nccl_id, myRank));
    for (int sz = 512; sz <= 8192; sz *= 2) {
      Run<T>(myRank, nRanks, comm, sz, metas, data_handles, input_handles, counter);
    }
    for (int i = 0; i < nRanks; ++i) {
      if (i != myRank) {
        CHECK_NVIDIA_CUDA_ERROR(cudaDeviceDisablePeerAccess(i));
      }
    }
  }

  template <typename T>
  void TestCustomAllReduce() {

    int device_count = -1;
    CHECK_NVIDIA_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (device_count != 2) {
      GTEST_SKIP_("This test is just for 2 GPUs");
    }

    int nRanks = 2;
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    CHECK_NVIDIA_CUDA_ERROR(cudaProfilerStart());
    std::vector<std::shared_ptr<std::thread>> run_threads;
    std::atomic<int> counter(0);
    std::vector<void *> metas(8);
    std::vector<void *> data_handles(8);
    std::vector<void *> input_handles(8);
    for (int myRank = 0; myRank < nRanks; ++myRank) {
      run_threads.emplace_back(std::shared_ptr<std::thread>(
          new std::thread(&LlamaNvidiaCustomAllReduceTestSuit::RunThread<T>, this, myRank, nRanks, nccl_id,
                          static_cast<void **>(metas.data()), static_cast<void **>(data_handles.data()),
                          static_cast<void **>(input_handles.data()), std::ref<std::atomic<int>>(counter))));
    }
    for (int myRank = 0; myRank < nRanks; ++myRank) {
      run_threads[myRank]->join();
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaProfilerStop());
  }
};

TEST_F(LlamaNvidiaCustomAllReduceTestSuit, FloatCustomAllReduceTest) { TestCustomAllReduce<float>(); }
TEST_F(LlamaNvidiaCustomAllReduceTestSuit, HalfCustomAllReduceTest) { TestCustomAllReduce<half>(); }
TEST_F(LlamaNvidiaCustomAllReduceTestSuit, BFloat16CustomAllReduceTest) { TestCustomAllReduce<__nv_bfloat16>(); }

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
