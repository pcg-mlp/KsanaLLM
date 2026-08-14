# KsanaLLM 功能模块划分（子域）

> 本文件是 `/ai-clarify` 等研发工作流做**子域匹配**的唯一来源。
> 目录名 = `spec/<子域>/` 下的子域名；关键词用于需求标题/描述的粗匹配。
> 当前为骨架：路径与职责可按实际演进再补全。

---

## scheduler

- **职责**：请求/批次调度、continuous batching、PD（Prefill/Decode）分离下的多 scheduler 协同与调度策略
- **关键词**：scheduler、batch_scheduler、scheduling、prefill、decode、PD 分离、multi-scheduler、协同调度、路由
- **主要代码路径（待细化）**：
  - `csrc/batch_scheduler/`
  - `csrc/engine/`（与调度相关的编排部分，待确认边界）
  - `python/ksana_llm/`（服务侧调度/入口相关，待确认）

---

## cache_manager

- **职责**：KV Cache / 显存块管理、PagedAttention 相关的 cache 分配与回收、prefix cache 等
- **关键词**：cache、cache_manager、KV、PagedAttention、block、prefix cache、显存管理
- **主要代码路径（待细化）**：
  - `csrc/cache_manager/`
  - `csrc/global_cache_connector/`（跨节点/全局 cache 连接，边界待确认）

---

## models

- **职责**：模型结构、权重加载、layer/算子编排、forward 路径与模型适配
- **关键词**：model、models、model_loader、weight、layer、forward、HuggingFace、算子
- **主要代码路径（待细化）**：
  - `csrc/models/`
  - `csrc/model_loader/`
  - `csrc/layers/`
  - `csrc/kernels/`（算子实现边界待确认）

---

## distributed

- **职责**：多卡/多机并行（如 tensor parallel）、进程间通信、connector/transfer、分布式运行时
- **关键词**：distributed、tensor parallel、TP、PP、IPC、connector、transfer、多卡、多机、NCCL
- **主要代码路径（待细化）**：
  - `csrc/distributed/`
  - `csrc/connector/`
  - `csrc/transfer/`
  - `csrc/ipc/`

---

## 兜底

- 无法归入以上四类时，工作流仍可将 `main` 作为默认兜底子域（需单独存在 `spec/main/` 时方可落盘）。
- 跨多子域需求：主子域写 `meta.yaml.subdomain`，其余写入 `cross_subdomains`，并考虑复杂度升为 `L`/`XL`。
