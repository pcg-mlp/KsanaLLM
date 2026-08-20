/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <xpu/runtime.h>
#include <xpu/xpuml.h>

#include <cstdio>

#include "test.h"

#include "device/kunlun/kunlun_device.h"
#include "device/kunlun/xpu_utils.h"
#include "utils/logger.h"

namespace ksana_llm {

class KunlunXpumlMemInfoTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { xpumlInit(); }
  static void TearDownTestSuite() { xpumlShutdown(); }
};

TEST_F(KunlunXpumlMemInfoTest, InitReturnsSuccess) {
  xpumlReturn_t ret = xpumlInit();
  EXPECT_EQ(ret, XPUML_SUCCESS);
}

TEST_F(KunlunXpumlMemInfoTest, InitIsIdempotent) {
  for (int i = 0; i < 5; i++) {
    xpumlReturn_t ret = xpumlInit();
    EXPECT_EQ(ret, XPUML_SUCCESS);
  }
}

TEST_F(KunlunXpumlMemInfoTest, GetDeviceCount) {
  unsigned int count = 0;
  xpumlReturn_t ret = xpumlDeviceGetCount(&count);
  EXPECT_EQ(ret, XPUML_SUCCESS);
  EXPECT_GT(count, 0u);
  KLLM_LOG_INFO << "xpumlDeviceGetCount: " << count;
}

TEST_F(KunlunXpumlMemInfoTest, GetMemoryInfoReturnsValidData) {
  // 通过共享 helper 解析当前设备对应的 xpuml 句柄，避免逻辑 ID 直接当 xpuml
  // 序号用（XPU_VISIBLE_DEVICES=7 时会误读物理卡0）。
  xpumlDevice_t device = GetXpumlHandleForCurrentDevice();

  xpumlMemory_t memory;
  xpumlReturn_t ret = xpumlDeviceGetMemoryInfo(device, &memory);
  ASSERT_EQ(ret, XPUML_SUCCESS);

  EXPECT_GT(memory.totalGlobalMemory, 0ULL);
  EXPECT_LE(memory.freeGlobalMemory, memory.totalGlobalMemory);
  EXPECT_LE(memory.usedGlobalMemory, memory.totalGlobalMemory);

  KLLM_LOG_INFO << "Device: total=" << (memory.totalGlobalMemory / (1024 * 1024)) << " MB"
                << ", free=" << (memory.freeGlobalMemory / (1024 * 1024)) << " MB"
                << ", used=" << (memory.usedGlobalMemory / (1024 * 1024)) << " MB";
}

TEST_F(KunlunXpumlMemInfoTest, FreeDecreaseAfterAlloc) {
  // 内存增量必须在"分配实际发生的那张卡"上测量：xpu_malloc 用的是当前设备，
  // 因此这里也必须用 GetXpumlHandleForCurrentDevice() 拿到同一张卡的句柄，
  // 否则（如 XPU_VISIBLE_DEVICES=7）会在物理卡0 上读到 0 增量而误判失败。
  xpumlDevice_t device = GetXpumlHandleForCurrentDevice();

  // XPU allocator rounds up to large internal pages (~32 MB granularity observed).
  const size_t kOverheadTolerance = 32ULL * 1024 * 1024;

  xpumlMemory_t before;
  ASSERT_EQ(xpumlDeviceGetMemoryInfo(device, &before), XPUML_SUCCESS);

  const size_t alloc_size = 64ULL * 1024 * 1024;  // 64 MB
  void* ptr = nullptr;
  XPU_CHECK(xpu_malloc(&ptr, alloc_size, XPU_MEM_MAIN));

  xpumlMemory_t after;
  ASSERT_EQ(xpumlDeviceGetMemoryInfo(device, &after), XPUML_SUCCESS);

  size_t consumed = before.freeGlobalMemory - after.freeGlobalMemory;
  EXPECT_GE(consumed, alloc_size);
  EXPECT_LE(consumed, alloc_size + kOverheadTolerance);

  XPU_CHECK(xpu_free(ptr));

  xpumlMemory_t restored;
  ASSERT_EQ(xpumlDeviceGetMemoryInfo(device, &restored), XPUML_SUCCESS);

  size_t recovered = restored.freeGlobalMemory - after.freeGlobalMemory;
  EXPECT_GE(recovered, alloc_size);
  EXPECT_LE(recovered, alloc_size + kOverheadTolerance);

  // Large allocation: ~20% of total memory
  const size_t large_alloc_size = static_cast<size_t>(before.totalGlobalMemory / 5);
  void* large_ptr = nullptr;
  XPU_CHECK(xpu_malloc(&large_ptr, large_alloc_size, XPU_MEM_MAIN));

  xpumlMemory_t after_large;
  ASSERT_EQ(xpumlDeviceGetMemoryInfo(device, &after_large), XPUML_SUCCESS);

  size_t large_consumed = restored.freeGlobalMemory - after_large.freeGlobalMemory;

  KLLM_LOG_INFO << "Large alloc: size=" << (large_alloc_size / (1024 * 1024)) << " MB"
                << ", free_before=" << (restored.freeGlobalMemory / (1024 * 1024)) << " MB"
                << ", free_after=" << (after_large.freeGlobalMemory / (1024 * 1024)) << " MB"
                << ", consumed=" << (large_consumed / (1024 * 1024)) << " MB";

  EXPECT_GE(large_consumed, large_alloc_size);
  EXPECT_LE(large_consumed, large_alloc_size + kOverheadTolerance);

  XPU_CHECK(xpu_free(large_ptr));

  xpumlMemory_t restored_large;
  ASSERT_EQ(xpumlDeviceGetMemoryInfo(device, &restored_large), XPUML_SUCCESS);

  size_t large_recovered = restored_large.freeGlobalMemory - after_large.freeGlobalMemory;
  EXPECT_GE(large_recovered, large_alloc_size);
  EXPECT_LE(large_recovered, large_alloc_size + kOverheadTolerance);
}

TEST_F(KunlunXpumlMemInfoTest, MemGetInfoTMatchesXpuml) {
  size_t free_val = 0, total_val = 0;
  MemGetInfoT<DEVICE_TYPE_KUNLUN>(&free_val, &total_val);

  EXPECT_GT(total_val, 0u);
  EXPECT_GT(free_val, 0u);
  EXPECT_LE(free_val, total_val);

  KLLM_LOG_INFO << "MemGetInfoT: total=" << (total_val / (1024 * 1024)) << " MB"
                << ", free=" << (free_val / (1024 * 1024)) << " MB";
}

// 验证 MemGetInfoT 的二分探测能反映碎片化：
// 昆仑芯是纯栈式分配且无 VMM。先后分配 A、B 两块（地址上 B 高于 A），再释放底部的
// A——B 仍浮在高地址，空闲被切成 [A 的洞] 与 [B 之上] 两段互不相连。此时 xpuml 报告的
// freeGlobalMemory 是两段之和，但真正能一次性连续分配的只有较大的一段，必然更小。
// MemGetInfoT 返回的应是后者（ProbeMaxContiguousMalloc 探测值），故 < 空闲总量。
TEST_F(KunlunXpumlMemInfoTest, MemGetInfoTReflectsFragmentation) {
  xpumlDevice_t device = GetXpumlHandleForCurrentDevice();

  xpumlMemory_t mem0;
  ASSERT_EQ(xpumlDeviceGetMemoryInfo(device, &mem0), XPUML_SUCCESS);
  const size_t free0 = static_cast<size_t>(mem0.freeGlobalMemory);

  // 空闲显存低于 5GB 视为环境不满足测试前提，直接失败而非跳过——
  // 这通常意味着设备未正确初始化或被异常占用，应暴露问题而非静默放过。
  const size_t kMinFree = 5ULL * 1024 * 1024 * 1024;  // 5 GB
  ASSERT_GE(free0, kMinFree) << "free memory too small: " << (free0 / (1024 * 1024)) << " MB";

  // 每块取空闲量的 ~10%：两块共 ~20%，留 ~80% 在顶部。释放 A 后洞(~10%)与顶部(~80%)
  // 分属两段，最大连续段约 80%，与空闲总量(~90%)的差(~10%)仍远大于探测精度
  // (256 MB 二分阈值)，断言稳健。取较小的 block 降低对共享环境的冲击。
  const size_t block = free0 / 10;
  ASSERT_GT(block, 0u);

  // xpu_malloc 失败时跳过而非 throw：共享 CI 上其它进程可能在此期间抢占显存，
  // 这是环境竞争而非代码缺陷，应 GTEST_SKIP 而非 FAIL。
  void* a = nullptr;
  if (xpu_malloc(&a, block, XPU_MEM_MAIN) != 0) {
    GTEST_SKIP() << "alloc A failed (device contended), block=" << (block / (1024 * 1024)) << " MB";
  }
  void* b = nullptr;
  if (xpu_malloc(&b, block, XPU_MEM_MAIN) != 0) {
    xpu_free(a);
    GTEST_SKIP() << "alloc B failed (device contended), block=" << (block / (1024 * 1024)) << " MB";
  }

  // 释放底部的 A；B 保持分配，制造出被打断的空闲空间。
  XPU_CHECK(xpu_free(a));

  xpumlMemory_t mem1;
  ASSERT_EQ(xpumlDeviceGetMemoryInfo(device, &mem1), XPUML_SUCCESS);
  const size_t raw_free = static_cast<size_t>(mem1.freeGlobalMemory);

  size_t probed_free = 0, total_val = 0;
  MemGetInfoT<DEVICE_TYPE_KUNLUN>(&probed_free, &total_val);

  XPU_CHECK(xpu_free(b));

  KLLM_LOG_INFO << "Fragmentation: block=" << (block / (1024 * 1024)) << " MB"
                << ", free0=" << (free0 / (1024 * 1024)) << " MB"
                << ", raw_free=" << (raw_free / (1024 * 1024)) << " MB"
                << ", probed_free=" << (probed_free / (1024 * 1024)) << " MB";

  // 碎片化后最大连续可分配量 < xpuml 报告的空闲总量，且差值显著（非探测精度噪声）。
  // 差值下界取 block/4：block ≈ free0/10 ≈ 数 GB，block/4 远大于 256MB 探测阈值，
  // 足以区分"真实碎片化"与"二分精度噪声"。
  EXPECT_GT(probed_free, 0u);
  EXPECT_LT(probed_free, raw_free);
  EXPECT_GT(raw_free - probed_free, block / 4);
}

// 验证 GetXpumlHandleForCurrentDevice 的 out_logical_id 出参与当前设备一致。
// 该出参用于让调用方（如 MemGetInfoT）避免自行再调一次 xpu_current_device，
// 防止多线程 xpu_set_device 切换下两次读取落到不同设备（TOCTOU 竞态）。
TEST_F(KunlunXpumlMemInfoTest, GetXpumlHandleReturnsConsistentLogicalId) {
  int logical_id = -1;
  xpumlDevice_t device = GetXpumlHandleForCurrentDevice(&logical_id);

  // 出参必须被赋值（不再是初始的 -1）。
  EXPECT_GE(logical_id, 0);

  // 与独立查询的当前设备 ID 一致。
  int current = -1;
  XPU_CHECK(xpu_current_device(&current));
  EXPECT_EQ(logical_id, current);

  // 句柄本身也应有效：能成功查询 physical_id。
  int physical_id = -1;
  ASSERT_EQ(xpumlDeviceGetId(device, &physical_id), XPUML_SUCCESS);
  EXPECT_GE(physical_id, 0);

  KLLM_LOG_INFO << "GetXpumlHandle: logical_id=" << logical_id << " physical_id=" << physical_id;
}

}  // namespace ksana_llm

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
