# Kunlun split_fuse + build-ahead 兼容适配

## 背景

Phase 1 build-ahead（`enable_executor_build_ahead`）通过 XPU 两阶段 Forward（kContext → kDecode）
在执行线程内联采样、XpuReplaceLastToken / XpuWriteRing，实现 decode 步 host-device 延迟隐藏，
目标将 Avg ITL 从 0.109 s 降至 ~0.083 s。

原实现在 `FastPathController::InitializeAtStartup` 中直接拦截了 `split_fuse_token_num != 0`，
导致 split_fuse（chunked prefill）与 build-ahead 不能同时开启。

本次适配参考同事 [zakwang/NumerousLLM feat/build-ahead-split-fuse](https://git.woa.com/zakwang/NumerousLLM/tree/feat/build-ahead-split-fuse)
分支，将 split_fuse 中间 chunk 标记为 `ForwardStepKind::kPrefillChunkMid`，
在不改动 XPU 两阶段 Forward 核心路径的前提下，让 WriteRing 跳过这些 chunk，
从而消除两者冲突。

---

## 变更文件一览

| 文件 | 变更性质 |
|------|---------|
| `csrc/runtime/infer_stage.h` | 新增 `ForwardStepKind` 枚举 + helper |
| `csrc/runtime/infer_request.h` | 新增 `forward_step_kind` 字段 |
| `csrc/runtime/infer_request.cpp` | 在 `LaunchPlanningTask` 中设置 `forward_step_kind` |
| `csrc/runtime/executor_infer_request.h` | 新增 `forward_step_kind` 至 `ExecutorInferRequest` 和 `InferRequestPatch` |
| `csrc/runtime/schedule_output.cpp` | 新增 `forward_step_kind` 序列化 / 反序列化（全量 + patch 两路） |
| `csrc/executor/executor_request_manager.cpp` | `UpdateInferRequest` 同步 `forward_step_kind` |
| `csrc/runtime/fast_path_controller.cpp` | 删除 `split_fuse_token_num != 0` 拦截 |
| `csrc/executor/model_sampler.cpp` | WriteRing 前过滤 `kPrefillChunkMid` 请求 |
| `csrc/kernels/kunlun/custom_kernels.xpu` | `write_ring_kernel` upsert 步跳过已 evict 的请求 |
| `csrc/batch_scheduler/CMakeLists.txt` | CUPTI 链接目录修复（所有测试目标） |
| `csrc/executor/CMakeLists.txt` | CUPTI 链接目录修复（`global_cache_put_worker_test`） |
| `csrc/ipc/CMakeLists.txt` | CUPTI 链接目录修复（`ipc_test` / `ipc_wrapper_test`） |

---

## 核心设计

### 1. ForwardStepKind 枚举（`infer_stage.h`）

```cpp
enum class ForwardStepKind : uint8_t {
  kDecode = 0,
  kPrefill,
  kPrefillChunkMid,  // split-fuse 中间 chunk：仅写 KV，不采样，不写 ring
};
inline bool IsPrefillChunkMidStep(ForwardStepKind kind) {
  return kind == ForwardStepKind::kPrefillChunkMid;
}
```

- `kPrefill`：prefill 最后一个 chunk（或无 split_fuse 时的完整 prefill）—— 正常采样。
- `kPrefillChunkMid`：split_fuse 中间 chunk —— 只写 KV cache，不进采样，不写 ring。
- `kDecode`：decode 步。

### 2. LaunchPlanningTask 中设置 forward_step_kind（`infer_request.cpp`）

```cpp
// 在 ++inflight_count_ 之前
if (new_inflight.workload.prefill_token_num > 0) {
  const bool is_last_chunk =
      (new_inflight.workload.prefill_token_num +
       new_inflight.workload.prefill_start_offset) >= prefilling_tokens_.size();
  forward_step_kind = is_last_chunk
      ? ForwardStepKind::kPrefill
      : ForwardStepKind::kPrefillChunkMid;
} else {
  forward_step_kind = ForwardStepKind::kDecode;
}
```

> **为何不用 `IsSplitPrefillStep()`**：该 helper 读 `inflight_tasks_[0]`（最老的 inflight task），
> 而此处关心的是刚推入的 `new_inflight`，必须直接检查 `new_inflight.workload`。

### 3. schedule_output 序列化（`schedule_output.cpp`）

在全量序列化 / 反序列化、patch 序列化 / 反序列化、patch 构建共 5 处
紧跟 `infer_stage` 新增 `forward_step_kind`，保证 runtime → executor 跨进程传递正确。

### 4. WriteRing 过滤（`model_sampler.cpp`）

```cpp
std::vector<int64_t> cur_req_ids_host;
cur_req_ids_host.reserve(ba_reqs.size());
for (const auto& req : ba_reqs) {
  if (!IsPrefillChunkMidStep(req->forward_step_kind)) {
    cur_req_ids_host.push_back(req->req_id);
  }
}
```

XPU 两阶段 Forward（kContext → kDecode）天然将 `cur_batch_pairs_cpu_` 覆写为 decode-only 的 pair，
`device_output_tokens_` 只含 kDecode 结果。`kPrefillChunkMid` 请求没有对应的 sampled token，
必须在写 ring 前过滤掉，否则 ring entry 偏移错乱，下一步 `XpuReplaceLastToken` 会写错 token（XPU 报 -714）。

### 5. FastPathController 解除拦截（`fast_path_controller.cpp`）

删除：
```cpp
if (sched_cfg.split_fuse_token_num != 0) {
    MarkDisabled(..., "split_fuse_token_num must be 0, got ...");
    return;
}
```
替换为注释：
```cpp
// split_fuse_token_num!=0 已通过 ForwardStepKind::kPrefillChunkMid 支持，不再拦截。
```

---

## CUPTI 链接目录修复

`libtorch_cpu.so`（经 `KUNLUN_PYTHON_LIBS` 传递）含版本化 `DT_NEEDED libcupti.so.11.8`；
CUDA 路径下的同名文件 SONAME 为 `11.7`，不匹配。正确文件在 nvidia conda 包路径下。

在 `csrc/CMakeLists.txt` 中 `batch_scheduler`（31）、`executor`（35）、`ipc`（36）
均早于 `kernels`（37）处理，`_CUPTI_LIB_DIR` 在它们运行时尚未设置，
导致相关测试目标链接失败。

**修复模式**（三个文件各自包含完整 `find_file` 块 + 链接目录声明）：

```cmake
if(WITH_KUNLUN AND NOT DEFINED _CUPTI_LIB_DIR)
  find_file(_CUPTI_LIB_FILE libcupti.so.11.8
    HINTS $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_cupti/lib
          /data/miniconda3/envs/env-3.10/lib/python3.10/site-packages/nvidia/cuda_cupti/lib
    NO_DEFAULT_PATH)
  if(_CUPTI_LIB_FILE)
    get_filename_component(_CUPTI_LIB_DIR "${_CUPTI_LIB_FILE}" DIRECTORY)
  else()
    set(_CUPTI_LIB_DIR ".../nvidia/cuda_cupti/lib")
  endif()
endif()
# batch_scheduler 用 link_directories（覆盖文件内所有测试）
# executor / ipc 用 target_link_directories（精准到具体目标）
```

受影响的测试目标：
- `global_cache_query_hook_test`（及 batch_scheduler 内所有其他测试）
- `global_cache_put_worker_test`
- `ipc_test` / `ipc_wrapper_test`

---

## Ring stale-entry 溢出修复（`write_ring_kernel`）

### 问题描述

`write_ring_kernel` 的 upsert 步没有跳过已被 evict 的请求，导致每次请求结束时：

1. carry-forward 步将完成的请求从 ring 中移除（dst 减 1）
2. upsert 步又把同一请求重新追加回 ring（dst 加 1）

净效果：ring 条目数**不减反平**。随着请求顺序增加，ring 中累积的过期条目（已完成请求的 stale entry）持续增长。当总请求数超过 `ring_entry_capacity = 2 * max_batch_size = 64` 后，新请求无法写入 ring → 第一个 decode 步的 `-1` placeholder 无法被 `XpuReplaceLastToken` 回填 → 模型以错误 token 作为输入 → 输出陷入重复 token 循环（如 `，，，，`、`的的的的`、`-正偶-` 等）。

### 现象

以 concurrency=1 顺序运行 220 个请求的 benchmark 时：
- 约 3–4% 的请求输出为单一 token 的死循环（text_len ≈ 514 字节，约等于 max_new_tokens × 1 byte/token）
- 两次 run 受影响的具体请求不同（随机分布在 req #64 之后）

### 根因

```
                  carry-forward          upsert
完成请求 N:  ring 中删除 N  →  upsert 重新追加 N
                ↑ 应当删除        ↑ 不应追加（N 已完成，无后续 decode 步）
```

### 修复

在 `write_ring_kernel` upsert 循环开头增加 evict 检查，跳过 `evict_ids` 中的请求：

```cpp
// 对每个 cur_pairs 中的 req_id，先检查是否在 evict_ids 中
int64_t evicted = 0;
for (int64_t k = 0; k < evict_count; ++k) {
    if (evict_ids[k] == rid) { evicted = 1; break; }
}
if (evicted) continue;  // 已完成的请求不写回 ring
```

修复后：请求完成时 ring 条目数净减 1（carry-forward 删除，upsert 跳过），ring 不再累积过期条目，永不溢出。

---

## 测试配置

同时开启 split_fuse 与 build-ahead，在 serving 配置中设置：

```yaml
batch_scheduler_config:
  split_fuse_token_num: 256
  enable_executor_build_ahead: true
  enable_async: true
```

预期效果：split_fuse 中间 chunk 正常转发 KV，最后一个 chunk 正常采样并写 ring，
decode 步通过 `XpuReplaceLastToken` 无 D2H 反填 `-1` placeholder，
Avg ITL 相较纯 build-ahead（无 split_fuse）进一步下降。
