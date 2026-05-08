# 统一 RequestID 设计文档

## 1. 背景与目标

**核心目标**：打通从 Python 接入层到 C++ 请求结束的完整 access log，让每一条请求都能通过同一个 Request ID 唯一定位、串联整个调用链。

重构前存在两个问题：

1. **Python ↔ C++ ID 断链**：C++ 端用 `IdGenerator` 自己生成 `req_id`，Python 入口无法拿到这个值，导致 Python access log 与 C++ 调度日志无法通过 ID 对齐，排障必须靠人工比对时间戳与 prompt。

2. **PD 分离场景 ID 双轨**：C++ 内部同时存在 `req_id`（调度用）和 `kv_comm_request_id`（KV Cache 传输用），代码里充斥着"用哪一个"的分支判断，来源是两个不同的 HTTP header（`request-id` 和 `kv-comm-request-id`）。

重构后：

- Python 接入层统一使用 `python/ksana_llm/request_id.py` 的 `generate_request_id()` 生成 ID，通过 `request-id` HTTP header 下发给 C++。
- C++ `Request` 构造时优先采用 header 值，Python 与 C++ 看到同一个 ID。
- 合并 `req_id` 与 `kv_comm_request_id` 为单一的 `req_id`（`int64_t`），HTTP header 统一为 `request-id`。

---

## 2. ID 的产生方式

### 2.1 Python 侧：统一生成器

所有部署形态**完全使用同一个函数** `generate_request_id()`，定义于 `python/ksana_llm/request_id.py`：

```python
SUB_ID_BITS: int = 8  # 必须与 C++ IdGenerator::kSubIdBits 保持一致
_MAX_BASE_COUNTER: int = (1 << (63 - SUB_ID_BITS)) - 1

def generate_request_id() -> int:
    base = _CUSTOM_ALLOCATOR() if _CUSTOM_ALLOCATOR else _get_default_counter().next()
    return (base & _MAX_BASE_COUNTER) << SUB_ID_BITS
```

**默认 counter 是 `_SharedMemoryCounter`**（mmap + fcntl，纯 stdlib），存储于 `/dev/shm/ksana_request_id_default.bin`。`fcntl.lockf` 保证多个 uvicorn worker 进程之间的原子性。文件在每次服务启动时**强制删除重建**，确保 counter 从 0 重新开始，不会继承上次运行的残留状态。

| 部署形态 | 调用入口 | 底层 counter | 额外配置 |
| --- | --- | --- | --- |
| **非 PD**（serving_server.py） | `ksana_service.py:get_trace_context` | 默认 shared-memory counter | 无 |
| **PD single-machine** | `simple_router/generate.py` | 默认 shared-memory counter | 无 |
| **PD multi-machine**（跨节点 Redis） | `simple_router/generate.py` | Redis allocator（注入） | `simple_router/main.py` 启动时调用 `set_request_id_allocator(allocate_comm_id)` |

> **调用方代码完全一致**，底层 counter 的差异对上层透明。

### 2.2 注入点

**非 PD**：`ksana_service.py:get_trace_context` 在每个请求进入时检查，若客户端未携带 `request-id` header，则调用 `generate_request_id()` 生成并注入 `req_ctx`：

```python
if "request-id" not in req_ctx:
    req_ctx["request-id"] = str(generate_request_id())
```

**PD**：`simple_router/generate.py` 在路由每个请求时调用 `generate_request_id()`，将 ID 写入转发给 prefill 和 decode 节点的 HTTP header：

```python
req_id = generate_request_id()
headers = {
    "kv-comm-group-key": f"{prefill_name}__{decode_name}",
    "request-id": str(req_id),
    ...
}
```

**PD multi-machine（Redis 模式）**：`simple_router/main.py` 的 `lifespan` 在 `init_allocator()` 之后注入 Redis allocator：

```python
if settings.comm_id_allocator_type == "redis":
    set_request_id_allocator(allocate_comm_id)
```

### 2.3 C++ 侧：header 优先，兜底自动生成

`csrc/utils/request.cpp`：

```cpp
int64_t base_id = id_generator_.Gen();  // 兜底（warmup / 单测等不经 Python 入口的场景）
if (req_ctx && req_ctx->count("request-id")) {
    try {
        base_id = std::stoll((*req_ctx)["request-id"]);
    } catch (...) {
        // 解析失败则沿用自动生成值，并打印 WARNING
    }
}
req_id = base_id;
```

线上请求两条链路（PD / 非 PD）都携带 `request-id` header，C++ 走 header 分支；`IdGenerator::Gen()` 仅覆盖 warmup、单测等不经 Python 入口的内部场景。

---

## 3. ID 的位布局

```
 bit 63        bit 8                  bit 0
   |             |                      |
   v             v                      v
   +---+----------------------+--------+
   | 0 |    base counter      | branch |
   +---+----------------------+--------+
                              ^
                              |
                  kSubIdBits = 8（预留 8 位）
```

### 3.1 为什么保留低 8 位

一条 `Request` 在 beam search 或 `num_return_sequences > 1` 时会派生出多条 `InferRequest`（每个分支一个），设计目标：

1. **可推导**：从任意分支 ID 反推得到所属 `Request` 的 base ID（`branch_id & ~0xFF`）。
2. **零成本**：单一 `int64_t` 直接可用，无需复合键 `<base_id, branch_index>`。

因此预留低 8 位作为分支号：

- `req_ids[i] = base_id | i`，其中 `i` 是分支下标（0-indexed）。
- `InferRequest(request, i)::req_id == request->req_ids[i]`。

C++ 实现（`csrc/utils/id_generator.h`）：

```cpp
class IdGenerator {
 public:
  static constexpr int kSubIdBits = 8;  // 预留低 8 位

  int64_t Gen() {
    return static_cast<int64_t>(
        (id_.fetch_add(1, std::memory_order_relaxed) & ((1ul << (63 - kSubIdBits)) - 1))
        << kSubIdBits);
  }
};
```

Python 端 `SUB_ID_BITS = 8` **必须**与 C++ `kSubIdBits` 保持一致；如果未来调整，两处需同步修改。

### 3.2 分支数上限

`kSubIdBits = 8` 意味着每条 `Request` 最多支持 `2^8 - 1 = 255` 个 `InferRequest` 分支。`Request` 构造函数显式校验，超出则抛异常拒绝请求：

```cpp
static constexpr size_t kMaxOutputGroup = (1u << IdGenerator::kSubIdBits) - 1;  // 255
if (output_group.size() > kMaxOutputGroup) {
    KLLM_THROW(fmt::format("output_group size {} exceeds max {}", ...));
}
```

---

## 4. 类型统一：为什么用 `int64_t`

历史代码里 `TaskKey::req_id`、`TransferTask::req_id`、`ForwardRequest::req_id` 等使用 `int` / `size_t`，跨模块传递时存在隐式截断与符号转换隐患。重构后统一收敛到 `int64_t`：

- `Request` / `InferRequest` / `ExecutorInferRequest::req_id`
- `ForwardRequest::req_id`（原 `size_t`）
- `TaskKey::req_id` / `TransferTask::req_id`（原 `int`）
- `TransferEngine` 全部接口形参与 `meta_map_` key（原 `int`）
- `TaskManager::GetShardIndex` / `GetShard` 形参（原 `int`）
- `BatchRequestSchedInfo::RequestSchedInfo::req_id`、`streaming_iterator::dying_requests` map key 等

**为什么不用 `size_t`**：`InferenceEngine::DoWarmupRun` 在 PREFILL 角色下会把 warmup 请求的 ID 置为 `-1` 作为 sentinel，让 `TransferEngine` 识别并跳过 KV 传输：

```cpp
if (connector_config.group_role == GroupRole::PREFILL) {
    req->req_id = -1;
    req->req_ids[0] = -1;  // sentinel：跳过 KV 传输
}
```

`size_t` 是无符号类型，`-1` 会被解释为极大正整数，破坏 sentinel 语义；`int64_t` 既能承载 64 位正常 ID，又能保留负值作为特殊标记。

---

## 5. Access Log 串联

Python access log 格式（`ksana_engine.py`）：

```
ACCESS| streaming=<bool> input_tokens=<n> output_tokens=<n> status=<OK|FAILED> elapsed=<ms>ms x_remote_ip=<ip> kv_comm_group_key=<key> request_id=<id> traceparent=<trace>
```

C++ 调度日志（`InferRequest::ScheduleStateToStr`）也打印 `request_id=<id>`。

两端均来自同一个 `request-id` header 值，可以直接 `grep request_id=<id>` 串联 Python 入口到 C++ 生成结束的全链路耗时。
