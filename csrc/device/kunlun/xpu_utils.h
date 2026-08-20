/* Copyright 2026 Tencent Inc.  All rights reserved.

   XPU runtime error checking utilities (Baidu Kunlun KL3 / KL4).
==============================================================================*/
#pragma once

#include <xpu/runtime.h>  // 间接 include <xpu/defs.h>，带入 XPU_SUCCESS + xpu_strerror
#include <xpu/xpuml.h>    // 设备管理 API（xpumlDevice_t / xpumlDeviceGet*）

#include <cstdint>
#include <stdexcept>
#include <string>

#include "device/device_types.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

namespace ksana_llm {

// XPU runtime API 全部返回 `int`，0 == XPU_SUCCESS。
// 失败时：① 写错误日志（带符号化字符串）② throw std::runtime_error。
// 上层 catch 不到时由进程默认 std::terminate 兜底，行为等价 abort。
inline void CheckXpuError(int result, const char* call, const char* file, const int line) {
  if (result != XPU_SUCCESS) {
    const char* err_str = xpu_strerror(result);
    KLLM_LOG_ERROR << fmt::format("XPU runtime error {} ({}): {} at {}:{}", result, err_str, call, file, line);
    throw std::runtime_error(std::string("XPU error ") + std::to_string(result) + " (" + err_str + ") at " + call +
                             " " + file + ":" + std::to_string(line));
  }
}

// 用法：XPU_CHECK(xpu_stream_create(&s));
#define XPU_CHECK(val) ::ksana_llm::CheckXpuError((val), #val, __FILE__, __LINE__)

// 解析"当前逻辑设备"对应的 xpuml 句柄。
//
// 背景：xpu_current_device 返回的是逻辑 ID（受 XPU_VISIBLE_DEVICES 影响，
// 例如 XPU_VISIBLE_DEVICES=7 时物理卡7 的逻辑 ID 为 0），而
// xpumlDeviceGetHandleByIndex 使用的是 xpuml 自己的枚举序号（物理序号的某种排列）。
// 直接用逻辑 ID 去 xpumlDeviceGetHandleByIndex 会查到错误的物理卡。
//
// 因此这里先用 XPUATTR_DEVID 拿到物理 devid，再遍历 xpuml 设备用
// xpumlDeviceGetId 匹配，返回真正对应当前设备的句柄。
//
// 调用方需保证已经 xpumlInit()。
//
// `out_logical_id`（可选）回传本次解析所用的逻辑设备 ID，供调用方日志使用——
// 避免调用方自行再调一次 xpu_current_device，以防多线程 xpu_set_device 切换
// 导致日志里的 logical_id 与实际被查询的设备不一致（TOCTOU 竞态）。
inline xpumlDevice_t GetXpumlHandleForCurrentDevice(int* out_logical_id = nullptr) {
  int device_id = 0;
  XPU_CHECK(xpu_current_device(&device_id));
  if (out_logical_id) *out_logical_id = device_id;

  uint64_t physical_devid = 0;
  XPU_CHECK(xpu_device_get_attr(&physical_devid, XPUATTR_DEVID, device_id));

  unsigned int xpuml_count = 0;
  xpumlReturn_t ret = xpumlDeviceGetCount(&xpuml_count);
  if (ret != XPUML_SUCCESS) {
    throw std::runtime_error("xpumlDeviceGetCount failed: " + std::to_string(ret));
  }

  for (unsigned int i = 0; i < xpuml_count; i++) {
    xpumlDevice_t candidate;
    if (xpumlDeviceGetHandleByIndex(i, &candidate) != XPUML_SUCCESS) continue;
    int phys_id = -1;
    if (xpumlDeviceGetId(candidate, &phys_id) != XPUML_SUCCESS) continue;
    if (static_cast<uint64_t>(phys_id) == physical_devid) return candidate;
  }
  throw std::runtime_error("xpuml: no device found for physical_devid=" + std::to_string(physical_devid));
}

}  // namespace ksana_llm
