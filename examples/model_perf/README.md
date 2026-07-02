# ModelPerf Backend 使用说明

`ModelPerfBackend` 是 event-driven scheduler 下的**纯性能测试模式**：不接收外部推理请求，按 `batch_config.json` 依次构造固定 batch、执行多轮 warmup/stat，并采集调度与算子耗时，结果写入 JSON 文件，供后续模拟器拟合使用。

## 前置条件

1. **调度器类型**：runtime yaml 中需启用 event-driven scheduler：

   ```yaml
   setting:
     batch_scheduler:
       scheduler_type: 1   # 1 = event_driven_scheduler
       max_step_tokens: 4096
   ```

2. **环境变量**（必须同时设置，否则不会激活）：

   | 变量 | 说明 |
   |------|------|
   | `ENABLE_MODEL_PERF=1` | 开启 ModelPerf 模式 |
   | `MODEL_PERF_CONFIG` | 指向 `model_perf_config.json` 的**绝对路径**（或确保进程工作目录下能解析其中的相对路径） |

3. **平台限制**：当前在 `ENABLE_ACL` 编译路径下不会执行 `StartPerf()`（与正常 warmup 相同被跳过），请在 GPU 环境使用。

4. **多 DP**：若 batch 中配置了 `dp_id > 0`，需将 yaml 中 `setting.global.attn_data_para_size` 设置为对应 DP 数。

## 快速开始

```bash
# 1. 进入 build 目录（编译产物与默认工作目录）
cd ${GIT_PROJECT_REPO_ROOT}/build

# 2. 设置环境变量（请替换为实际 runtime yaml 与模型路径）
export ENABLE_MODEL_PERF=1
export MODEL_PERF_CONFIG=${GIT_PROJECT_REPO_ROOT}/examples/model_perf/model_perf_config.json

# 3. 启动服务
# ./bin/ksana_llm_server --config ${GIT_PROJECT_REPO_ROOT}/examples/your_runtime.yaml
```

服务在 `InferenceEngine::Start()` 阶段会**替代** `DoWarmupRun()`，阻塞执行完所有合法 batch 后写出 `model_perf_result.json`（路径由 `model_perf_config.json` 的 `output_file` 指定），随后**自动退出进程**，不会启动 RPC/HTTP 推理服务。

## 配置文件

### model_perf_config.json

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `batch_config` | string | 是 | batch 规格文件路径（相对路径相对于 `build` 工作目录） |
| `output_file` | string | 否 | 结果 JSON 输出路径，默认 `model_perf_result.json` |
| `output_level` | string | 否 | 输出粒度：`summary`（仅多轮汇总）或 `detail`（汇总 + 每轮数据），默认 `detail` |
| `token_seed` | uint | 否 | input token 生成的全局种子，默认 `42`；各请求在此基础上叠加偏移，保证 MoE 路由等结果可复现 |
| `metrics_granularity` | string | 否 | 预留字段，当前仅支持 `"batch"` |

本目录示例：

```json
{
  "output_file": "model_perf_result.json",
  "output_level": "detail",
  "metrics_granularity": "batch",
  "token_seed": 42,
  "batch_config": "../examples/model_perf/batch_config.json"
}
```

### batch_config.json

顶层为 `batches` 数组，按顺序逐个执行。每个 batch 结构如下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `batch_id` | string | batch 标识，写入结果 JSON |
| `warmup_rounds` | int | warmup 轮数（不记录指标，用于预热） |
| `stat_rounds` | int | 正式统计轮数（每轮记录一条 `ProfileMetrics`） |
| `dp_batches` | array | 各 DP 分组的请求构成 |

`dp_batches[].requests[]` 中每个请求项：

| 字段 | 类型 | 说明 |
|------|------|------|
| `input_token_len` | int | prefix 长度（已缓存上下文 token 数） |
| `query_token_num` | int | 本轮 prefill 的 query token 数 |
| `req_num` | int | 相同规格的请求副本数，默认 `1` |

**token 语义**

- 每个请求的 `input_tokens` 总长度为 `input_token_len + query_token_num`。
- token 内容由 `GenerateTokens` 按 `token_seed` 确定性生成，取值范围为 `[0, ModelConfig.vocab_size)`。
- KV 分配分两步：先为 prefix（`input_token_len`）分配 block，再为总长（`input_token_len + query_token_num`）分配 query 部分。

**max_step_tokens 约束**

单个 batch 内所有请求的 `query_token_num × req_num` 之和不得超过 runtime yaml 中的 `max_step_tokens`（对应 `BatchSchedulerConfig.max_step_token_num`）。超限的 batch 会在解析时打印告警并**剔除**，不会执行。

示例（本目录 `batch_config.json`）：

- `prefill_2req_dp0`：DP0 上 2 个请求，各 512 query，合计 1024 query token。
- `prefill_mixed_dp0`：DP0 上两种不同 input/query 组合。
- `prefill_multi_dp`：DP0/DP1 各 1 个请求（需 `attn_data_para_size >= 2`）。

## 执行流程

```
InferenceEngine::Start()
  └─ ModelPerfBackend::StartPerf()          # 阻塞直到全部 batch 完成
       for each batch in batches:
         1. 构造 InferRequest，AddInferRequest 一次性提交
         2. ModelPerfBackend::Schedule()     # 多轮（warmup + stat）调度同一批请求
              - 第 0 轮：分配 KV、设置 workload、加入 running_reqs
              - 后续轮：恢复快照后再次加入 running_reqs
         3. UpdateWithGenerationResult()     # 每轮结束采集指标；全部轮次完成后 kStopRequest
         4. batch_waiter 返回，进入下一 batch
       写出 model_perf_result.json
  └─ InferenceEngine::Stop() → 进程退出（不启动 RPC/HTTP 服务）
```

此模式下**不处理**前端/API 推理请求，仅执行配置文件中的 batch。

## 输出结果

`output_file` 为 JSON，结构示例：

```json
{
  "batches": [
    {
      "batch_id": "prefill_2req_dp0",
      "stat_summary": {
        "round_count": 3,
        "schedule_forward_interval_ns": {"mean": 50000, "stddev": 1000},
        "forward_metrics": {
          "main": {
            "forward_time_ns": {"mean": 1200000, "stddev": 100},
            "ops": {
              "attn": {
                "cumulative_time_ns": {"mean": 1000000, "stddev": 0},
                "count": {"mean": 32, "stddev": 0}
              },
              "ffn": {
                "cumulative_time_ns": {"mean": 2000000, "stddev": 0},
                "count": {"mean": 32, "stddev": 0}
              }
            }
          }
        }
      },
      "stat_rounds": [
        {
          "schedule_forward_interval_ns": 50000,
          "forward_metrics": {
            "main": {
              "forward_time_ns": 1200000,
              "ops": {
                "attn": {"cumulative_time_ns": 1000000, "count": 32},
                "ffn": {"cumulative_time_ns": 2000000, "count": 32}
              }
            }
          }
        }
      ]
    }
  ]
}
```

`output_level=summary` 时仅输出 `stat_summary`，不包含 `stat_rounds`。汇总中的 `stddev` 为总体标准差（`round_count <= 1` 时为 0）。

| 指标 | 含义 |
|------|------|
| `stat_summary.*.mean` / `stddev` | 多轮 stat 指标的均值与标准差 |
| `stat_rounds` | 每轮原始指标（仅 `output_level=detail`） |
| `schedule_forward_interval_ns` | 调度 dispatch 到首次 forward 开始的间隔（纳秒） |
| `forward_metrics.main` | 正常 forward（`RunMode::kMain`）的子指标 |
| `forward_metrics.speculative` | 投机解码 forward（`RunMode::kNextN`）的子指标，未启用时不出现 |
| `forward_time_ns` | 单次 forward 总耗时（纳秒） |
| `ops.<name>.cumulative_time_ns` | 某采集项在所有层上的累积耗时（纳秒） |
| `ops.<name>.count` | 该采集项在本轮 forward 中的执行次数（通常等于层数） |

内置采集项 `attn`、`ffn` 由框架预注册；模型可通过 `ForwardProfileOpRegistry::Register` 扩展更多采集项，详见下一节。

## 自定义 Profile 采集项

ModelPerf 在 forward 路径上通过 `ProfileMetrics` 采集算子耗时。除内置的 `attn` / `ffn` 外，可在模型代码中注册并埋点任意采集项，结果会自动出现在 JSON 的 `forward_metrics.*.ops` 字典中。

### 机制概览

```
ModelPerfBackend 设置 enable_profile_metrics=true
  → ModelExecutor::Forward 设置 ForwardingContext::current_forward_metrics
  → 模型 LayerForward 中 MODEL_PERF_OP_* 宏写入 ForwardPerfMetrics::ops
  → SamplingOutput → GenerationOutputGroup → model_perf_result.json
```

- **关闭 profiling 时**（正常推理）：`current_forward_metrics == nullptr`，宏完全空展开，零运行时开销。
- **开启 profiling 时**（ModelPerf 模式）：按采集项累积 `cumulative_time_ns` 与 `count`。

相关代码：`csrc/runtime/profile_metrics.h`。

### 1. 注册采集项

每个采集项名称在进程内**只能注册一次**。重复注册同名项会返回非法 id `ForwardProfileOp::kInvalid`（`-1`）并打印 ERROR 日志。

```cpp
#include "runtime/profile_metrics.h"

namespace {
// 文件级 static，保证只 Register 一次
const ForwardProfileOpId kMhcPre =
    ForwardProfileOpRegistry::Register("mhc_pre");
const ForwardProfileOpId kSharedExpert =
    ForwardProfileOpRegistry::Register("shared_expert");
}
```

内置项无需 Register，直接使用常量：

| 常量 | 名称 | id |
|------|------|-----|
| `ForwardProfileOp::kAttn` | `"attn"` | 0 |
| `ForwardProfileOp::kFfn` | `"ffn"` | 1 |

### 2. 在 LayerForward 中埋点

在目标代码段前后插入宏，`metrics_ptr` 统一使用 `forwarding_context.current_forward_metrics`：

```cpp
// 单 stream / 跨 stream 场景：全设备同步
MODEL_PERF_OP_BEGIN(forwarding_context.current_forward_metrics, mhc_t);
pre_attention_mhc_->PreForwardWithNorm(residual_buffer, hidden_buffer_tensors_0);
MODEL_PERF_OP_END(forwarding_context.current_forward_metrics, kMhcPre, mhc_t);

// 内置 attn 示例（simple_decoder_layer 等）
MODEL_PERF_OP_BEGIN(forwarding_context.current_forward_metrics, attn_t);
mha_->Forward(...);
MODEL_PERF_OP_END(forwarding_context.current_forward_metrics, ForwardProfileOp::kAttn, attn_t);
```

**多 stream 场景**（算子提交在已知 stream 上，如 `GetSecondaryComputeStream()`）：

```cpp
auto& secondary_stream = forwarding_context.GetContext()->device->GetSecondaryComputeStream();
MODEL_PERF_OP_BEGIN(forwarding_context.current_forward_metrics, se_t);
shared_expert_->Forward(..., LayerExecContext(secondary_stream));
MODEL_PERF_OP_END_STREAM(forwarding_context.current_forward_metrics, kSharedExpert, se_t, secondary_stream);
```

**算子后已有 sync / event wait** 时，可跳过同步：

```cpp
MODEL_PERF_OP_BEGIN(forwarding_context.current_forward_metrics, op_t);
// ... 算子，末尾已 StreamSynchronize ...
MODEL_PERF_OP_END_NOSYNC(forwarding_context.current_forward_metrics, kMyOp, op_t);
```

### 3. 宏选用指南

| 宏 | 同步方式 | 适用场景 |
|----|----------|----------|
| `MODEL_PERF_OP_END` | `DeviceSynchronize()` | 单 stream；或算子跨多个 stream 需全部完成 |
| `MODEL_PERF_OP_END_STREAM` | `StreamSynchronize(stream)` | 多 stream，算子仅在指定 stream 上 |
| `MODEL_PERF_OP_END_NOSYNC` | 无 | END 前已手动 sync |

> profiling 模式下同步会显著拉长 forward 耗时，结果反映的是「可测量的算子时间」，不代表生产吞吐。

### 4. 输出 JSON 中的体现

注册 `"mhc_pre"` 后，`stat_rounds` 中会出现：

```json
"forward_metrics": {
  "main": {
    "forward_time_ns": 1200000,
    "ops": {
      "attn": {"cumulative_time_ns": 1000000, "count": 32},
      "ffn": {"cumulative_time_ns": 2000000, "count": 32},
      "mhc_pre": {"cumulative_time_ns": 50000, "count": 32}
    }
  }
}
```

`stat_summary.forward_metrics.main.ops.mhc_pre` 会给出多轮 `mean` / `stddev`。

### 5. 注意事项

1. **Register 只调用一次**：用文件级 `static const ForwardProfileOpId` 保存返回值；不要对 `"attn"` / `"ffn"` 再次 Register。
2. **检查返回值**：若 `Register` 返回 `ForwardProfileOp::kInvalid`，说明名称冲突，后续宏不会写入指标。
3. **命名规范**：建议使用小写 + 下划线（如 `mhc_pre`、`shared_expert`），与 JSON key 一致。
4. **MTP / 投机解码**：`RunMode::kNextN` 的 forward 写入 `forward_metrics.speculative`，与 main 分开统计。
5. **单元测试**：`csrc/runtime/profile_metrics_test.cpp` 覆盖注册、编解码与宏行为，可作为参考。

## 自定义 batch 时注意

1. 先估算 `Σ(query_token_num × req_num)`，确保不超过 yaml 中的 `max_step_tokens`。
2. 多 DP batch 的 `dp_id` 从 0 开始，且小于 `attn_data_para_size`。
3. `warmup_rounds` 建议 ≥ 1，避免首次执行的冷启动影响 stat 数据。
4. 示例以 `build` 为工作目录：`batch_config` 使用 `../examples/model_perf/batch_config.json`，`output_file` 默认写入 `build/model_perf_result.json`；若需自定义路径，建议使用绝对路径。

## 相关代码

- `csrc/batch_scheduler/event_driven_scheduler/model_perf/model_perf_backend.{h,cpp}`
- `csrc/batch_scheduler/event_driven_scheduler/model_perf/model_perf_workload_balancer.{h,cpp}`
- `csrc/batch_scheduler/event_driven_scheduler/model_perf/model_perf_token_utils.h`
- `csrc/runtime/profile_metrics.{h,cpp}` — Profile 采集项注册、宏与序列化
- `csrc/runtime/profile_metrics_test.cpp` — 注册 / 编解码 / 宏单元测试
- `csrc/engine/inference_engine.cpp`（`InitModelPerfBackend` / `Start`）
