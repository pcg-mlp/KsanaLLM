/* Copyright 2026 Tencent Inc.  All rights reserved.

   BKCL communication layer correctness test (standalone, no gtest dependency).

   Multi-threaded test that verifies BkclAllGatherLayer and BkclAllReduceSumLayer
   using std::thread + BKCL collective communication (requires >= 2 XPU cards).

   Build:
     cd /zhongzhicao_ws/KsanaLLM/build_tests
     cmake . && make -j$(nproc)

   Run (requires 2 XPU cards visible):
     ./bkcl_comm_test

   Architecture:
     - Uses std::thread (one per rank) within a single process
     - Each rank sets its own XPU device + initializes BKCL comm
     - Heap-allocated SharedRegion for passing BKCLUniqueId between ranks
     - Validates all_gather (dim 0 + dim 1) and all_reduce (SUM)

   NOTE: BKCL's P2P initialization uses System V shared memory and cross-device
   xpu_memcpy, which requires all ranks to share the same address space (single
   process). Multi-process mode (fork+exec) fails with XPUERR_INVALID_PARAM (-707)
   because device pointers are not valid across process boundaries. This matches
   production code which also uses multi-threading for device parallelism.
==============================================================================*/

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// Include cuda_runtime.h BEFORE any namespace to avoid name collision
// This is the XPU CUDA compat layer header (not real CUDA).
#include <cuda_runtime.h>

// Python.h for runtime initialization (required by XPU CUDA compat layer)
#include <Python.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <thread>
#include <vector>

#include "device/common_device.h"
#include "device/device_types.h"
#include "device/device_utils.h"
#include "device/kunlun/bkcl_utils.h"
#include "device/kunlun/kunlun_device_context_impl.h"
#include "layers/bkcl_all_gather_layer.h"
#include "layers/bkcl_all_reduce_sum_layer.h"

namespace ksana_llm {

// ─── Testable wrappers: expose protected members for direct setup ───

class TestableBkclAllGatherLayer : public BkclAllGatherLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
    tp_size_ = ctx->global->GetTensorParallelSize();
    dp_size_ = 1;
    attn_dp_atp_size_ = 1;
    attn_dp_group_id_ = 0;
    attn_dp_rank_id_ = 0;
  }
};

class TestableBkclAllReduceSumLayer : public BkclAllReduceSumLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
    tp_size_ = ctx->global->GetTensorParallelSize();
    dp_size_ = 1;
    attn_dp_atp_size_ = 1;
    attn_dp_group_id_ = 0;
    attn_dp_rank_id_ = 0;
  }
};

// ─── Shared memory region for BKCL unique ID exchange ───

struct SharedRegion {
  BKCLUniqueId uid;
  volatile int uid_ready;               // 0 = not ready, 1 = ready
  std::atomic<int> init_done_count{0};  // count of ranks that finished device init
};

// ─── fp16 / bf16 host <-> float conversion helpers ───
// These avoid depending on <cuda_fp16.h> availability in the XPU compat layer.

static inline uint16_t F32ToF16(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000);
  uint32_t mant = x & 0x7fffff;
  int exp = static_cast<int>((x >> 23) & 0xff);

  if (exp == 0) {
    // zero or subnormal: round to nearest even (subnormals with mantissa may round up)
    if (mant == 0) return sign;
    // Subnormal in fp32 → denormal in fp16: add implicit leading 1, shift right.
    // The smallest fp32 subnormal (exp=0, mant!=0) maps to fp16 zero via underflow.
    // Compute the fp16 mantissa with round-to-nearest-even.
    uint32_t mant_with_implicit = mant | 0x800000;  // restore implicit 1
    // Shift amount: fp32 subnormal bias adjustment = 127 - 1 = 126, plus 14 for fp16
    // For the first subnormal: shift = 126 - (leading_zero_count + 14) + ...
    // Simplification: shift right by 13, then round. Subnormals that are too small
    // will naturally become 0 after rounding.
    uint32_t shifted = mant_with_implicit >> 13;
    uint32_t round_bit = (mant_with_implicit >> 12) & 1;
    uint32_t sticky = (mant_with_implicit & 0xfff) != 0;
    if (round_bit && (sticky || (shifted & 1))) shifted++;
    return static_cast<uint16_t>(sign | (shifted & 0x3ff));
  }

  exp = exp - 127 + 15;
  if (exp <= 0) {
    // Underflow: shift mantissa right, add round-to-nearest-even
    uint32_t mant_with_implicit = mant | 0x800000;
    int shift = 1 - exp;
    uint32_t shifted = (shift >= 23) ? 0 : (mant_with_implicit >> shift);
    uint32_t round_bit = (shift >= 1 && shift <= 23) ? ((mant_with_implicit >> (shift - 1)) & 1) : 0;
    uint32_t sticky = (shift >= 2) ? ((mant_with_implicit & ((1u << (shift - 1)) - 1)) != 0) : 0;
    if (round_bit && (sticky || (shifted & 1))) shifted++;
    return static_cast<uint16_t>(sign | (shifted & 0x3ff));
  }
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);

  // Normal case: round-to-nearest-even
  uint32_t mant_fp16 = (x >> 13) & 0x3ff;
  uint32_t round_bit = (x >> 12) & 1;
  uint32_t sticky = (x & 0xfff) != 0;
  if (round_bit && (sticky || (mant_fp16 & 1))) mant_fp16++;
  return static_cast<uint16_t>(sign | (exp << 10) | (mant_fp16 & 0x3ff));
}

static inline float F16ToF32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  int exp = (h >> 10) & 0x1f;
  if (exp == 0) return 0.0f;
  uint32_t frac = static_cast<uint32_t>(h & 0x3ff) << 13;
  if (exp == 31) {
    uint32_t r = sign | 0x7f800000 | frac;
    float f;
    memcpy(&f, &r, sizeof(f));
    return f;
  }
  exp = exp - 15 + 127;
  uint32_t r = sign | (static_cast<uint32_t>(exp) << 23) | frac;
  float f;
  memcpy(&f, &r, sizeof(f));
  return f;
}

// BF16: upper 16 bits of float32 (round to nearest even)
static inline uint16_t F32ToBF16(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  uint32_t rounding_bias = ((x & 0x1ffffu) == 0x10000u) ? ((x & 0x20000u) << 1) : 0x10000u;
  x += rounding_bias;
  return static_cast<uint16_t>(x >> 16);
}

static inline float BF16ToF32(uint16_t h) {
  uint32_t x = static_cast<uint32_t>(h) << 16;
  float f;
  memcpy(&f, &x, sizeof(f));
  return f;
}

// ─── Enum for low-precision dtype category ───

enum class LowPrecDtype { kFp16, kBf16 };

// Convert float to uint16 (fp16 or bf16) based on dtype category
static inline uint16_t F32ToLowPrec(float f, LowPrecDtype dt) {
  return (dt == LowPrecDtype::kFp16) ? F32ToF16(f) : F32ToBF16(f);
}

// Convert uint16 (fp16 or bf16) to float based on dtype category
static inline float LowPrecToF32(uint16_t h, LowPrecDtype dt) {
  return (dt == LowPrecDtype::kFp16) ? F16ToF32(h) : BF16ToF32(h);
}

// Map LowPrecDtype to DataType
static inline DataType LowPrecToDataType(LowPrecDtype dt) {
  return (dt == LowPrecDtype::kFp16) ? TYPE_FP16 : TYPE_BF16;
}

// ─── TestContext: shared state for all test functions ───

struct TestContext {
  int rank;
  int world_size;
  std::shared_ptr<Context> context;
  Stream compute_stream;
};

// ─── Device + BKCL initialization ───

// When libtorch_serving.so is loaded, libtorch_python.so's global constructor
// initializes the CUDA compat layer. After that, direct xpu_set_device() fails
// with -101 UNLESS Python + torch_xmlir is initialized first (Py_Initialize +
// "import torch"). The worker process does this init before calling RunRank,
// so SetDevice (xpu_set_device) works correctly here.

static bool InitTestEnvironment(int rank, int world_size, SharedRegion* shared, TestContext& ctx) {
  printf("[Rank %d] Starting (world_size=%d)\n", rank, world_size);

  // Serialize device initialization: each rank waits for all previous ranks
  // to finish their SetDevice + DeviceContext creation. This avoids races in
  // xpu_set_device (which is process-global on XPU, unlike CUDA's per-thread).
  while (shared->init_done_count.load(std::memory_order_acquire) < rank) {
    usleep(100);
  }

  SetDevice(rank);

  // Create DeviceContext
  auto device_ctx = std::make_shared<DeviceContext>(1);

  // Signal that this rank's device init is done
  shared->init_done_count.fetch_add(1, std::memory_order_release);

  // Wait for all ranks to finish device init before proceeding to BKCL
  while (shared->init_done_count.load(std::memory_order_acquire) < world_size) {
    usleep(100);
  }

  // BKCL initialization: rank 0 generates UID, others wait
  if (rank == 0) {
    BKCLUniqueId uid = GenerateBKCLUniqueID();
    shared->uid = uid;
    __sync_synchronize();
    shared->uid_ready = 1;
  } else {
    while (!shared->uid_ready) {
      usleep(1000);
    }
    __sync_synchronize();
  }

  // Initialize BKCL communicator
  device_ctx->ext->SetBKCLUniqueId(shared->uid);
  device_ctx->ext->InitBkcl(rank, world_size);
  printf("[Rank %d] BKCL initialized\n", rank);

  // Create GlobalContext with env=nullptr (safe for Get*() methods we use:
  //   GetTensorParallelSize() returns tensor_parallel_size_ from ctor,
  //   IsRunContextDecodeAndDecodeSerially() returns default true)
  auto global_ctx = std::make_shared<GlobalContext>(static_cast<size_t>(world_size),  // tensor_parallel_size
                                                    1,                                // attn_data_parallel_size
                                                    1,                                // max_multi_batch_size
                                                    nullptr);                         // env (not used in our test path)

  // Create a Context
  auto context = std::make_shared<Context>();
  context->device = device_ctx;
  context->global = global_ctx;

  ctx.rank = rank;
  ctx.world_size = world_size;
  ctx.context = context;
  ctx.compute_stream = device_ctx->GetPrimaryComputeStream();
  return true;
}

// ─── AllReduce SUM test (fp32, uniform data) ───

static int TestAllReduceSumFp32(const TestContext& ctx) {
  printf("[Rank %d] Test 1: AllReduce SUM\n", ctx.rank);
  constexpr size_t kH = 4, kW = 256;
  constexpr size_t kElems = kH * kW;

  Tensor input(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH, kW});
  Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH, kW});

  std::vector<float> h_input(kElems, static_cast<float>(ctx.rank + 1));
  MemcpyAsync(input.GetPtr<void>(), h_input.data(), kElems * sizeof(float), MEMCPY_HOST_TO_DEVICE, ctx.compute_stream);
  StreamSynchronize(ctx.compute_stream);

  TestableBkclAllReduceSumLayer layer;
  layer.SetupForTest(ctx.context, TYPE_FP32);

  std::vector<Tensor> inputs = {input};
  std::vector<Tensor> outputs = {output};
  Status status = layer.Forward(inputs, outputs);
  StreamSynchronize(ctx.compute_stream);

  if (!status.OK()) {
    printf("[Rank %d] FAIL AllReduce: status=%s\n", ctx.rank, status.ToString().c_str());
    return 1;
  }

  float expected_val = static_cast<float>(ctx.world_size * (ctx.world_size + 1)) / 2.0f;
  std::vector<float> h_output(kElems);
  Memcpy(h_output.data(), outputs[0].GetPtr<void>(), kElems * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < kElems; ++i) {
    max_diff = std::max(max_diff, std::fabs(h_output[i] - expected_val));
  }
  if (max_diff > 1e-3f) {
    printf("[Rank %d] FAIL AllReduce: max_diff=%e, expected=%f, got[0]=%f\n", ctx.rank, max_diff, expected_val,
           h_output[0]);
    return 1;
  }
  printf("[Rank %d] PASS AllReduce (max_diff=%e)\n", ctx.rank, max_diff);
  return 0;
}

// ─── AllGather dim 0 test (fp32) ───

static int TestAllGatherDim0Fp32(const TestContext& ctx) {
  printf("[Rank %d] Test 2: AllGather dim 0\n", ctx.rank);
  constexpr size_t kH = 4, kW = 128;
  constexpr size_t kElems = kH * kW;

  Tensor input(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH, kW});
  Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH * static_cast<size_t>(ctx.world_size), kW});

  std::vector<float> h_input(kElems);
  for (size_t i = 0; i < kElems; ++i) {
    h_input[i] = static_cast<float>(ctx.rank) * 100.0f + static_cast<float>(i) * 0.1f;
  }
  MemcpyAsync(input.GetPtr<void>(), h_input.data(), kElems * sizeof(float), MEMCPY_HOST_TO_DEVICE, ctx.compute_stream);
  StreamSynchronize(ctx.compute_stream);

  TestableBkclAllGatherLayer layer;
  layer.SetupForTest(ctx.context, TYPE_FP32);

  std::vector<Tensor> inputs = {input};
  std::vector<Tensor> outputs = {output};
  Status status = layer.Forward(inputs, outputs);
  StreamSynchronize(ctx.compute_stream);

  if (!status.OK()) {
    printf("[Rank %d] FAIL AllGather dim0: status=%s\n", ctx.rank, status.ToString().c_str());
    return 1;
  }

  if (outputs[0].shape[0] != kH * ctx.world_size || outputs[0].shape[1] != kW) {
    printf("[Rank %d] FAIL AllGather dim0: unexpected shape [%zu, %zu]\n", ctx.rank, outputs[0].shape[0],
           outputs[0].shape[1]);
    return 1;
  }

  std::vector<float> h_output(kH * ctx.world_size * kW);
  Memcpy(h_output.data(), outputs[0].GetPtr<void>(), h_output.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (int r = 0; r < ctx.world_size; ++r) {
    for (size_t i = 0; i < kElems; ++i) {
      float expected = static_cast<float>(r) * 100.0f + static_cast<float>(i) * 0.1f;
      float actual = h_output[r * kElems + i];
      max_diff = std::max(max_diff, std::fabs(actual - expected));
    }
  }
  if (max_diff > 1e-3f) {
    printf("[Rank %d] FAIL AllGather dim0: max_diff=%e\n", ctx.rank, max_diff);
    return 1;
  }
  printf("[Rank %d] PASS AllGather dim0 (max_diff=%e)\n", ctx.rank, max_diff);
  return 0;
}

// ─── AllGather dim 1 rearrange test (fp32) ───

static int TestAllGatherDim1RearrangeFp32(const TestContext& ctx) {
  printf("[Rank %d] Test 3: AllGather dim 1 (rearrange)\n", ctx.rank);
  constexpr size_t kH = 8, kWPer = 64;
  constexpr size_t kElems = kH * kWPer;

  Tensor input(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH, kWPer});
  Tensor temp(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH * static_cast<size_t>(ctx.world_size), kWPer});
  Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH, kWPer * static_cast<size_t>(ctx.world_size)});

  std::vector<float> h_input(kElems);
  for (size_t row = 0; row < kH; ++row) {
    for (size_t col = 0; col < kWPer; ++col) {
      h_input[row * kWPer + col] =
          static_cast<float>(ctx.rank) * 1000.0f + static_cast<float>(row) * 10.0f + static_cast<float>(col) * 0.01f;
    }
  }
  MemcpyAsync(input.GetPtr<void>(), h_input.data(), kElems * sizeof(float), MEMCPY_HOST_TO_DEVICE, ctx.compute_stream);
  StreamSynchronize(ctx.compute_stream);

  TestableBkclAllGatherLayer layer;
  layer.SetupForTest(ctx.context, TYPE_FP32);

  std::vector<Tensor> inputs;
  inputs.push_back(input);
  inputs.push_back(temp);
  std::vector<Tensor> outputs;
  outputs.push_back(output);
  printf("[Rank %d] AllGather dim1: inputs.size()=%zu, input.shape=[%zu,%zu], temp.shape=[%zu,%zu]\n", ctx.rank,
         inputs.size(), inputs[0].shape[0], inputs[0].shape[1], inputs[1].shape[0], inputs[1].shape[1]);
  Status status = layer.Forward(inputs, outputs);
  StreamSynchronize(ctx.compute_stream);

  if (!status.OK()) {
    printf("[Rank %d] FAIL AllGather dim1: status=%s\n", ctx.rank, status.ToString().c_str());
    return 1;
  }

  size_t expected_w = kWPer * ctx.world_size;
  if (outputs[0].shape[0] != kH || outputs[0].shape[1] != expected_w) {
    printf("[Rank %d] FAIL AllGather dim1: unexpected shape [%zu, %zu], expected [%zu, %zu]\n", ctx.rank,
           outputs[0].shape[0], outputs[0].shape[1], kH, expected_w);
    return 1;
  }

  std::vector<float> h_output(kH * expected_w);
  Memcpy(h_output.data(), outputs[0].GetPtr<void>(), h_output.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (int r = 0; r < ctx.world_size; ++r) {
    for (size_t row = 0; row < kH; ++row) {
      for (size_t col = 0; col < kWPer; ++col) {
        float expected =
            static_cast<float>(r) * 1000.0f + static_cast<float>(row) * 10.0f + static_cast<float>(col) * 0.01f;
        float actual = h_output[row * expected_w + r * kWPer + col];
        max_diff = std::max(max_diff, std::fabs(actual - expected));
      }
    }
  }
  if (max_diff > 1e-2f) {
    printf("[Rank %d] FAIL AllGather dim1: max_diff=%e\n", ctx.rank, max_diff);
    for (size_t row = 0; row < std::min(kH, (size_t)2); ++row) {
      printf("  row %zu: ", row);
      for (size_t j = 0; j < std::min(expected_w, (size_t)8); ++j) {
        printf("%.2f ", h_output[row * expected_w + j]);
      }
      printf("...\n");
    }
    return 1;
  }
  printf("[Rank %d] PASS AllGather dim1 (max_diff=%e)\n", ctx.rank, max_diff);
  return 0;
}

// ─── AllReduce SUM non-uniform test (fp32, 8-rank) ───

static int TestAllReduceSumNonUniformFp32(const TestContext& ctx) {
  printf("[Rank %d] Test 4: AllReduce SUM (non-uniform 8-rank)\n", ctx.rank);
  constexpr size_t kH = 4, kW = 256;
  constexpr size_t kElems = kH * kW;

  Tensor input(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH, kW});
  Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH, kW});

  std::vector<float> h_input(kElems);
  for (size_t i = 0; i < kElems; ++i) {
    h_input[i] = static_cast<float>(ctx.rank) * 1000.0f + static_cast<float>(i) * 0.001f;
  }
  MemcpyAsync(input.GetPtr<void>(), h_input.data(), kElems * sizeof(float), MEMCPY_HOST_TO_DEVICE, ctx.compute_stream);
  StreamSynchronize(ctx.compute_stream);

  TestableBkclAllReduceSumLayer layer;
  layer.SetupForTest(ctx.context, TYPE_FP32);

  std::vector<Tensor> inputs = {input};
  std::vector<Tensor> outputs = {output};
  Status status = layer.Forward(inputs, outputs);
  StreamSynchronize(ctx.compute_stream);

  if (!status.OK()) {
    printf("[Rank %d] FAIL AllReduce non-uniform: status=%s\n", ctx.rank, status.ToString().c_str());
    return 1;
  }

  float rank_sum = static_cast<float>(ctx.world_size * (ctx.world_size - 1)) / 2.0f * 1000.0f;
  std::vector<float> h_output(kElems);
  Memcpy(h_output.data(), outputs[0].GetPtr<void>(), kElems * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < kElems; ++i) {
    float expected = rank_sum + static_cast<float>(ctx.world_size) * static_cast<float>(i) * 0.001f;
    max_diff = std::max(max_diff, std::fabs(h_output[i] - expected));
  }
  if (max_diff > 1e-2f) {
    printf("[Rank %d] FAIL AllReduce non-uniform: max_diff=%e, expected[0]=%f, got[0]=%f\n", ctx.rank, max_diff,
           rank_sum, h_output[0]);
    return 1;
  }
  printf("[Rank %d] PASS AllReduce non-uniform (max_diff=%e)\n", ctx.rank, max_diff);
  return 0;
}

// ─── AllGather dim 0 boundary test (fp32, 8-rank) ───

static int TestAllGatherDim0BoundaryFp32(const TestContext& ctx) {
  printf("[Rank %d] Test 5: AllGather dim 0 (boundary)\n", ctx.rank);
  constexpr size_t kH = 1, kW = 512;
  constexpr size_t kElems = kH * kW;

  Tensor input(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH, kW});
  Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH * static_cast<size_t>(ctx.world_size), kW});

  std::vector<float> h_input(kElems);
  for (size_t col = 0; col < kW; ++col) {
    h_input[col] = static_cast<float>(ctx.rank) * 1000.0f + static_cast<float>(col) * 0.01f;
  }
  MemcpyAsync(input.GetPtr<void>(), h_input.data(), kElems * sizeof(float), MEMCPY_HOST_TO_DEVICE, ctx.compute_stream);
  StreamSynchronize(ctx.compute_stream);

  TestableBkclAllGatherLayer layer;
  layer.SetupForTest(ctx.context, TYPE_FP32);

  std::vector<Tensor> inputs = {input};
  std::vector<Tensor> outputs = {output};
  Status status = layer.Forward(inputs, outputs);
  StreamSynchronize(ctx.compute_stream);

  if (!status.OK()) {
    printf("[Rank %d] FAIL AllGather dim0 boundary: status=%s\n", ctx.rank, status.ToString().c_str());
    return 1;
  }

  if (outputs[0].shape[0] != kH * ctx.world_size || outputs[0].shape[1] != kW) {
    printf("[Rank %d] FAIL AllGather dim0 boundary: unexpected shape [%zu, %zu]\n", ctx.rank, outputs[0].shape[0],
           outputs[0].shape[1]);
    return 1;
  }

  std::vector<float> h_output(kH * ctx.world_size * kW);
  Memcpy(h_output.data(), outputs[0].GetPtr<void>(), h_output.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (int r = 0; r < ctx.world_size; ++r) {
    float* chunk = &h_output[r * kW];
    for (size_t col = 0; col < kW; ++col) {
      float expected = static_cast<float>(r) * 1000.0f + static_cast<float>(col) * 0.01f;
      max_diff = std::max(max_diff, std::fabs(chunk[col] - expected));
    }
    // Boundary check: last col of rank r vs first col of rank r+1
    if (r < ctx.world_size - 1) {
      float expected_r_last = static_cast<float>(r) * 1000.0f + static_cast<float>(kW - 1) * 0.01f;
      float expected_r1_first = static_cast<float>(r + 1) * 1000.0f;
      float actual_r_last = chunk[kW - 1];
      float* next_chunk = &h_output[(r + 1) * kW];
      float actual_r1_first = next_chunk[0];
      max_diff = std::max(max_diff, std::fabs(actual_r_last - expected_r_last));
      max_diff = std::max(max_diff, std::fabs(actual_r1_first - expected_r1_first));
    }
  }
  if (max_diff > 1e-2f) {
    printf("[Rank %d] FAIL AllGather dim0 boundary: max_diff=%e\n", ctx.rank, max_diff);
    return 1;
  }
  printf("[Rank %d] PASS AllGather dim0 boundary (max_diff=%e)\n", ctx.rank, max_diff);
  return 0;
}

// ─── AllGather dim 1 edge-case test (fp32, 8-rank, h=1 and h=32) ───

static int TestAllGatherDim1EdgeCaseFp32(const TestContext& ctx) {
  for (auto h_case : {1ul, 32ul}) {
    constexpr size_t kWPer = 64;
    size_t kH = h_case;
    size_t kElems = kH * kWPer;

    printf("[Rank %d] Test 6: AllGather dim 1 (rearrange, h=%zu)\n", ctx.rank, kH);

    Tensor input(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH, kWPer});
    Tensor temp(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH * static_cast<size_t>(ctx.world_size), kWPer});
    Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {kH, kWPer * static_cast<size_t>(ctx.world_size)});

    std::vector<float> h_input(kElems);
    for (size_t row = 0; row < kH; ++row) {
      for (size_t col = 0; col < kWPer; ++col) {
        h_input[row * kWPer + col] =
            static_cast<float>(ctx.rank) * 1000.0f + static_cast<float>(row) * 10.0f + static_cast<float>(col) * 0.01f;
      }
    }
    MemcpyAsync(input.GetPtr<void>(), h_input.data(), kElems * sizeof(float), MEMCPY_HOST_TO_DEVICE,
                ctx.compute_stream);
    StreamSynchronize(ctx.compute_stream);

    TestableBkclAllGatherLayer layer;
    layer.SetupForTest(ctx.context, TYPE_FP32);

    std::vector<Tensor> inputs;
    inputs.push_back(input);
    inputs.push_back(temp);
    std::vector<Tensor> outputs;
    outputs.push_back(output);
    Status status = layer.Forward(inputs, outputs);
    StreamSynchronize(ctx.compute_stream);

    if (!status.OK()) {
      printf("[Rank %d] FAIL AllGather dim1 h=%zu: status=%s\n", ctx.rank, kH, status.ToString().c_str());
      return 1;
    }

    size_t expected_w = kWPer * ctx.world_size;
    if (outputs[0].shape[0] != kH || outputs[0].shape[1] != expected_w) {
      printf("[Rank %d] FAIL AllGather dim1 h=%zu: unexpected shape [%zu, %zu]\n", ctx.rank, kH, outputs[0].shape[0],
             outputs[0].shape[1]);
      return 1;
    }

    std::vector<float> h_output(kH * expected_w);
    Memcpy(h_output.data(), outputs[0].GetPtr<void>(), h_output.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

    float max_diff = 0.0f;
    for (int r = 0; r < ctx.world_size; ++r) {
      for (size_t row = 0; row < kH; ++row) {
        for (size_t col = 0; col < kWPer; ++col) {
          float expected =
              static_cast<float>(r) * 1000.0f + static_cast<float>(row) * 10.0f + static_cast<float>(col) * 0.01f;
          float actual = h_output[row * expected_w + r * kWPer + col];
          max_diff = std::max(max_diff, std::fabs(actual - expected));
        }
      }
    }

    // Extra boundary check: verify rank 0 and rank 7 specifically
    size_t row_to_check = (kH == 1) ? 0ul : (kH - 1);
    for (size_t col = 0; col < std::min(kWPer, (size_t)4); ++col) {
      float exp_r0 = 0.0f + static_cast<float>(row_to_check) * 10.0f + static_cast<float>(col) * 0.01f;
      float act_r0 = h_output[row_to_check * expected_w + 0 * kWPer + col];
      max_diff = std::max(max_diff, std::fabs(act_r0 - exp_r0));
      float exp_r7 = 7.0f * 1000.0f + static_cast<float>(row_to_check) * 10.0f + static_cast<float>(col) * 0.01f;
      float act_r7 = h_output[row_to_check * expected_w + 7 * kWPer + col];
      max_diff = std::max(max_diff, std::fabs(act_r7 - exp_r7));
    }

    if (max_diff > 1e-2f) {
      printf("[Rank %d] FAIL AllGather dim1 h=%zu: max_diff=%e\n", ctx.rank, kH, max_diff);
      for (size_t row = 0; row < std::min(kH, (size_t)2); ++row) {
        printf("  row %zu: ", row);
        for (size_t j = 0; j < std::min(expected_w, (size_t)16); ++j) {
          printf("%.2f ", h_output[row * expected_w + j]);
        }
        printf("...\n");
      }
      return 1;
    }
    printf("[Rank %d] PASS AllGather dim1 h=%zu (max_diff=%e)\n", ctx.rank, kH, max_diff);
  }
  return 0;
}

// ─── AllReduce SUM non-uniform test (low-precision, 8-rank) ───
// Generic for fp16 and bf16

static int TestAllReduceSumNonUniformLowPrec(const TestContext& ctx, LowPrecDtype dt, int test_num, const char* dt_name,
                                             float tolerance) {
  printf("[Rank %d] Test %d: AllReduce SUM (non-uniform, %s)\n", ctx.rank, test_num, dt_name);
  constexpr size_t kH = 4, kW = 256;
  constexpr size_t kElems = kH * kW;
  constexpr size_t kBytes = kElems * sizeof(uint16_t);

  DataType dtype = LowPrecToDataType(dt);
  Tensor input(MemoryLocation::LOCATION_DEVICE, dtype, {kH, kW});
  Tensor output(MemoryLocation::LOCATION_DEVICE, dtype, {kH, kW});

  std::vector<uint16_t> h_input(kElems);
  for (size_t i = 0; i < kElems; ++i) {
    float val = static_cast<float>(ctx.rank) * 1000.0f + static_cast<float>(i) * 0.001f;
    h_input[i] = F32ToLowPrec(val, dt);
  }
  MemcpyAsync(input.GetPtr<void>(), h_input.data(), kBytes, MEMCPY_HOST_TO_DEVICE, ctx.compute_stream);
  StreamSynchronize(ctx.compute_stream);

  TestableBkclAllReduceSumLayer layer;
  layer.SetupForTest(ctx.context, dtype);
  std::vector<Tensor> inputs = {input};
  std::vector<Tensor> outputs = {output};
  Status status = layer.Forward(inputs, outputs);
  StreamSynchronize(ctx.compute_stream);

  if (!status.OK()) {
    printf("[Rank %d] FAIL AllReduce %s: status=%s\n", ctx.rank, dt_name, status.ToString().c_str());
    return 1;
  }

  std::vector<uint16_t> h_output(kElems);
  Memcpy(h_output.data(), outputs[0].GetPtr<void>(), kBytes, MEMCPY_DEVICE_TO_HOST);

  float rank_sum = static_cast<float>(ctx.world_size * (ctx.world_size - 1)) / 2.0f * 1000.0f;
  float max_diff = 0.0f;
  for (size_t i = 0; i < kElems; ++i) {
    float expected_fp32 = rank_sum + static_cast<float>(ctx.world_size) * static_cast<float>(i) * 0.001f;
    float expected = LowPrecToF32(F32ToLowPrec(expected_fp32, dt), dt);
    float actual = LowPrecToF32(h_output[i], dt);
    max_diff = std::max(max_diff, std::fabs(actual - expected));
  }
  if (max_diff > tolerance) {
    printf("[Rank %d] FAIL AllReduce %s: max_diff=%e, expected[0]=%f, got[0]=%f\n", ctx.rank, dt_name, max_diff,
           rank_sum, LowPrecToF32(h_output[0], dt));
    return 1;
  }
  printf("[Rank %d] PASS AllReduce %s (max_diff=%e)\n", ctx.rank, dt_name, max_diff);
  return 0;
}

// ─── AllGather dim 0 boundary test (low-precision, 8-rank) ───
// Generic for fp16 and bf16

static int TestAllGatherDim0BoundaryLowPrec(const TestContext& ctx, LowPrecDtype dt, int test_num, const char* dt_name,
                                            float tolerance) {
  printf("[Rank %d] Test %d: AllGather dim 0 (boundary, %s)\n", ctx.rank, test_num, dt_name);
  constexpr size_t kH = 1, kW = 512;
  constexpr size_t kElems = kH * kW;
  constexpr size_t kBytes = kElems * sizeof(uint16_t);

  DataType dtype = LowPrecToDataType(dt);
  Tensor input(MemoryLocation::LOCATION_DEVICE, dtype, {kH, kW});
  Tensor output(MemoryLocation::LOCATION_DEVICE, dtype, {kH * static_cast<size_t>(ctx.world_size), kW});

  std::vector<uint16_t> h_input(kElems);
  for (size_t col = 0; col < kW; ++col) {
    h_input[col] = F32ToLowPrec(static_cast<float>(ctx.rank) * 1000.0f + static_cast<float>(col) * 0.01f, dt);
  }
  MemcpyAsync(input.GetPtr<void>(), h_input.data(), kBytes, MEMCPY_HOST_TO_DEVICE, ctx.compute_stream);
  StreamSynchronize(ctx.compute_stream);

  TestableBkclAllGatherLayer layer;
  layer.SetupForTest(ctx.context, dtype);
  std::vector<Tensor> inputs = {input};
  std::vector<Tensor> outputs = {output};
  Status status = layer.Forward(inputs, outputs);
  StreamSynchronize(ctx.compute_stream);

  if (!status.OK()) {
    printf("[Rank %d] FAIL AllGather dim0 %s: status=%s\n", ctx.rank, dt_name, status.ToString().c_str());
    return 1;
  }

  if (outputs[0].shape[0] != kH * ctx.world_size || outputs[0].shape[1] != kW) {
    printf("[Rank %d] FAIL AllGather dim0 %s: unexpected shape [%zu, %zu]\n", ctx.rank, dt_name, outputs[0].shape[0],
           outputs[0].shape[1]);
    return 1;
  }

  std::vector<uint16_t> h_output(kH * ctx.world_size * kW);
  Memcpy(h_output.data(), outputs[0].GetPtr<void>(), h_output.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (int r = 0; r < ctx.world_size; ++r) {
    for (size_t col = 0; col < kW; ++col) {
      float expected_fp32 = static_cast<float>(r) * 1000.0f + static_cast<float>(col) * 0.01f;
      float expected = LowPrecToF32(F32ToLowPrec(expected_fp32, dt), dt);
      float actual = LowPrecToF32(h_output[r * kW + col], dt);
      max_diff = std::max(max_diff, std::fabs(actual - expected));
    }
  }
  if (max_diff > tolerance) {
    printf("[Rank %d] FAIL AllGather dim0 %s: max_diff=%e\n", ctx.rank, dt_name, max_diff);
    return 1;
  }
  printf("[Rank %d] PASS AllGather dim0 %s (max_diff=%e)\n", ctx.rank, dt_name, max_diff);
  return 0;
}

// ─── AllGather dim 1 edge-case test (low-precision, 8-rank, h=1 and h=32) ───
// Generic for fp16 and bf16

static int TestAllGatherDim1EdgeCaseLowPrec(const TestContext& ctx, LowPrecDtype dt, int test_num, const char* dt_name,
                                            float tolerance) {
  for (auto h_case : {1ul, 32ul}) {
    constexpr size_t kWPer = 64;
    size_t kH = h_case;
    size_t kElems = kH * kWPer;

    printf("[Rank %d] Test %d: AllGather dim 1 (rearrange, %s, h=%zu)\n", ctx.rank, test_num, dt_name, kH);

    DataType dtype = LowPrecToDataType(dt);
    Tensor input(MemoryLocation::LOCATION_DEVICE, dtype, {kH, kWPer});
    Tensor temp(MemoryLocation::LOCATION_DEVICE, dtype, {kH * static_cast<size_t>(ctx.world_size), kWPer});
    Tensor output(MemoryLocation::LOCATION_DEVICE, dtype, {kH, kWPer * static_cast<size_t>(ctx.world_size)});

    std::vector<uint16_t> h_input(kElems);
    for (size_t row = 0; row < kH; ++row) {
      for (size_t col = 0; col < kWPer; ++col) {
        float val =
            static_cast<float>(ctx.rank) * 1000.0f + static_cast<float>(row) * 10.0f + static_cast<float>(col) * 0.01f;
        h_input[row * kWPer + col] = F32ToLowPrec(val, dt);
      }
    }
    MemcpyAsync(input.GetPtr<void>(), h_input.data(), kElems * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                ctx.compute_stream);
    StreamSynchronize(ctx.compute_stream);

    TestableBkclAllGatherLayer layer;
    layer.SetupForTest(ctx.context, dtype);
    std::vector<Tensor> inputs;
    inputs.push_back(input);
    inputs.push_back(temp);
    std::vector<Tensor> outputs;
    outputs.push_back(output);
    Status status = layer.Forward(inputs, outputs);
    StreamSynchronize(ctx.compute_stream);

    if (!status.OK()) {
      printf("[Rank %d] FAIL AllGather dim1 %s h=%zu: status=%s\n", ctx.rank, dt_name, kH, status.ToString().c_str());
      return 1;
    }

    size_t expected_w = kWPer * ctx.world_size;
    if (outputs[0].shape[0] != kH || outputs[0].shape[1] != expected_w) {
      printf("[Rank %d] FAIL AllGather dim1 %s h=%zu: unexpected shape [%zu, %zu]\n", ctx.rank, dt_name, kH,
             outputs[0].shape[0], outputs[0].shape[1]);
      return 1;
    }

    std::vector<uint16_t> h_output(kH * expected_w);
    Memcpy(h_output.data(), outputs[0].GetPtr<void>(), h_output.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);

    float max_diff = 0.0f;
    for (int r = 0; r < ctx.world_size; ++r) {
      for (size_t row = 0; row < kH; ++row) {
        for (size_t col = 0; col < kWPer; ++col) {
          float expected_fp32 =
              static_cast<float>(r) * 1000.0f + static_cast<float>(row) * 10.0f + static_cast<float>(col) * 0.01f;
          float expected = LowPrecToF32(F32ToLowPrec(expected_fp32, dt), dt);
          float actual = LowPrecToF32(h_output[row * expected_w + r * kWPer + col], dt);
          max_diff = std::max(max_diff, std::fabs(actual - expected));
        }
      }
    }
    if (max_diff > tolerance) {
      printf("[Rank %d] FAIL AllGather dim1 %s h=%zu: max_diff=%e\n", ctx.rank, dt_name, kH, max_diff);
      return 1;
    }
    printf("[Rank %d] PASS AllGather dim1 %s h=%zu (max_diff=%e)\n", ctx.rank, dt_name, kH, max_diff);
  }
  return 0;
}

// ─── Single-rank test orchestrator ───

static int RunRank(int rank, int world_size, SharedRegion* shared) {
  TestContext ctx;
  if (!InitTestEnvironment(rank, world_size, shared, ctx)) {
    return 1;
  }

  int fail_count = 0;

  // Basic tests (2+ ranks)
  fail_count += TestAllReduceSumFp32(ctx);
  fail_count += TestAllGatherDim0Fp32(ctx);
  fail_count += TestAllGatherDim1RearrangeFp32(ctx);

  // 8-rank stress tests
  if (world_size >= 8) {
    fail_count += TestAllReduceSumNonUniformFp32(ctx);
    fail_count += TestAllGatherDim0BoundaryFp32(ctx);
    fail_count += TestAllGatherDim1EdgeCaseFp32(ctx);

    // fp16 tests
    fail_count += TestAllReduceSumNonUniformLowPrec(ctx, LowPrecDtype::kFp16, 7, "fp16", 1e2f);
    fail_count += TestAllGatherDim0BoundaryLowPrec(ctx, LowPrecDtype::kFp16, 8, "fp16", 1e1f);
    fail_count += TestAllGatherDim1EdgeCaseLowPrec(ctx, LowPrecDtype::kFp16, 9, "fp16", 1e1f);

    // bf16 tests
    fail_count += TestAllReduceSumNonUniformLowPrec(ctx, LowPrecDtype::kBf16, 10, "bf16", 5e2f);
    fail_count += TestAllGatherDim0BoundaryLowPrec(ctx, LowPrecDtype::kBf16, 11, "bf16", 1e2f);
    fail_count += TestAllGatherDim1EdgeCaseLowPrec(ctx, LowPrecDtype::kBf16, 12, "bf16", 1e2f);
  }

  // Cleanup
  if (ctx.context->device->ext->GetBKCLParam().bkcl_comm) {
    DestroyBKCLParam(ctx.context->device->ext->GetBKCLParam());
  }

  printf("[Rank %d] %s (%d failures)\n", rank, fail_count == 0 ? "ALL PASS" : "SOME FAILED", fail_count);
  return fail_count;
}

}  // namespace ksana_llm

// ─── main: Multi-threaded orchestrator ───
// Production code uses multi-threading (one thread per device). BKCL's P2P
// initialization uses System V shared memory and xpu_memcpy between devices,
// which only works within a single process (shared address space). Multi-process
// mode (fork+exec) fails because BKCL's internal P2P setup passes device pointers
// across process boundaries, which are invalid (XPUERR_INVALID_PARAM = -707).
//
// This test now mirrors production: single process, one thread per rank.

static int CountXpuDevices() {
  // Respect CUDA_VISIBLE_DEVICES if set (XPU uses CUDA compat layer)
  const char* visible = getenv("CUDA_VISIBLE_DEVICES");
  if (visible && strlen(visible) > 0) {
    int count = 1;
    for (const char* p = visible; *p; ++p) {
      if (*p == ',') ++count;
    }
    return count;
  }
  // Fallback: count /dev/xpu* device files
  int count = 0;
  DIR* dir = opendir("/dev");
  if (!dir) return 0;
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (strncmp(entry->d_name, "xpu", 3) == 0 && entry->d_name[3] >= '0' && entry->d_name[3] <= '9') {
      ++count;
    }
  }
  closedir(dir);
  return count;
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  // Initialize Python + torch_xmlir (required for XPU CUDA compat layer)
  Py_Initialize();
  PyRun_SimpleString("import torch");

  printf("=== BKCL Communication Layer Test ===\n");

  int count = CountXpuDevices();
  if (count < 2) {
    printf("SKIP: Need >= 2 XPU devices, found %d\n", count);
    Py_Finalize();
    return 0;
  }
  int world_size = std::min(count, 8);
  printf("Detected %d XPU devices (via /dev), testing with world_size=%d", count, world_size);
  if (world_size == 8) {
    printf(" [8-card mode]");
  }
  printf("\n");

  // Shared region for BKCL UID exchange (heap-allocated, shared by all threads)
  ksana_llm::SharedRegion shared;
  memset(&shared, 0, sizeof(shared));

  // Launch one thread per rank
  std::vector<std::thread> threads;
  std::vector<int> results(world_size, -1);
  for (int r = 0; r < world_size; ++r) {
    threads.emplace_back(
        [r, world_size, &shared, &results]() { results[r] = ksana_llm::RunRank(r, world_size, &shared); });
  }

  // Wait for all threads
  for (auto& t : threads) {
    t.join();
  }

  bool all_passed = true;
  for (int i = 0; i < world_size; ++i) {
    if (results[i] != 0) {
      printf("[Main] Rank %d failed (%d failures)\n", i, results[i]);
      all_passed = false;
    }
  }

  printf("=== BKCL Communication Layer Test: %s ===\n", all_passed ? "ALL PASS" : "FAILED");
  Py_Finalize();
  return all_passed ? 0 : 1;
}
