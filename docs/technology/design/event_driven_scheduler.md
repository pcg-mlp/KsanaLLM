# 事件驱动调度器设计文档

## 1. 背景与动机

### 1.1 当前问题

现有调度器存在以下问题，需要通过事件驱动架构进行改进：

1. **耦合过高**：大量业务相关逻辑被直接写入调度器内部，难以扩展和维护。
2. **策略不清晰**：调度策略与框架逻辑混杂，策略层不够干净、可替换性差。
3. **调度时机滞后**：在异步调度场景下，调度结果不会在 launch 前根据请求最新状态做调整，导致执行延迟或无效调度。
4. **负载均衡缺失**：多个 DP（Data Parallel）之间缺乏统一的负载管理与协调机制。

### 1.2 设计目标

- 通过事件驱动与责任链，将「事件处理」与「调度决策」解耦。
- 统一由调度主线程处理事件并更新状态，在 launch 前有机会根据最新状态做调度调整。
- 引入清晰的调度状态与规划数据结构，支持多 DP 负载协调与可扩展的 Hook 机制。

---

## 2. 核心设计

### 2.1 Hook 机制：责任链模式

采用**责任链模式（Chain of Responsibility）**，使多个对象都有机会处理请求，从而解耦发送者与接收者。请求沿链传递，由链上的对象同步或异步处理。

**行为约定：**

- **对象管理器**：
  - 若收到**异步请求**：将请求放入队列，等待异步处理器完成任务后的通知。
  - 若收到**同步请求**：在当前对象内直接处理。
- 请求在某节点处理完成后，继续沿责任链向后传递。
- 对象管理器可持有 **Context**，其中包含异步线程池；异步处理器将任务提交到线程池，完成后将结果回传给对象管理器。
- 线程池可在多个对象管理器之间**共享**。
- **InferRequest** 需记录：其所属对象管理器对应的责任链是否已处理完成。

### 2.2 调度器线程模型

- **单调度主线程**：从事件队列中拉取事件并顺序处理，所有会触发调度行为的入口都转化为事件入队。
- **事件驱动循环**：队列中的调度事件处理完后，会尝试更新**尚未 launch 的 batch**，使 launch 前能根据最新状态做调整。

**与 Hook 的协作：**

- 请求预处理等环节存在 **Hook Manager**，可能执行耗时操作，因此使用**异步线程池**执行。
- 调度器向这些处理程序注册 **task finish callback**。
- 该 callback 内部执行**事件入队**，从而唤醒/驱动调度主线程。

Task 可能包括：预处理、异步调度相关操作等。

**典型流程：**

1. Hook Manager 发起异步 task。
2. 异步 task 完成后调用 callback。
3. Callback 向调度器推送事件（event push），触发调度主线程。
4. 调度主线程调用 Hook Manager 更新 task 状态，并根据状态变化决定是否进行下一步调度调整。
5. 若可继续，则以当前 **PlanningBatch** 为上下文进行调度调整。
6. 若有 PlanningBatch 可以 launch 且已有接收方，则放入 **launch 队列**。（接收方是否也通过事件驱动主线程，可在实现时按需设计。）

---

## 3. 核心状态与数据结构

### 3.1 DpScheduleState

表示**单个 DP 分组**内的请求与运行情况。

| 字段/概念 | 说明 |
|-----------|------|
| `preprocessing_reqs` | 责任链中处于异步等待的请求，使用 `map<req_id, InferRequestPtr>` 便于快速查找；异步完成后先按 hook_idx 调用 `UpdateAsyncResult`，对可继续的请求再从指定 hook_idx 继续责任链，链执行完的移入 prefilling_reqs |
| `prefilling_reqs` | 责任链已执行完、可进入 prefill 的请求 |
| `decoding_reqs` | 正在 decode 的请求 |
| `stopped_reqs` | 已停止的请求 ID 列表（需传递给其他节点） |
| `recomputed_reqs` | 触发过 recompute 的请求 ID 列表（需传递给其他节点） |

AddRequest 的请求经 `Process` 后**一定**进入 `prefilling_reqs`（kContinue）或 `preprocessing_reqs`（kWaitAsyncResult），不再经中间队列。

### 3.2 ScheduleState

调度器**全局状态**。

| 字段 | 说明 |
|------|------|
| `schedule_states` | `vector<DpScheduleState>`，按 `dp_idx` 直接索引各 DP 调度状态 |
| `req_id_to_req_ptr` | 全局请求索引 `unordered_map<req_id, InferRequestPtr>`，用于仅有 `req_id`（如异步回调）时快速定位请求对象 |

### 3.3 DpBatch

表示**当前在一个 DP 内已规划、即将执行**的 Batch。

| 字段 | 说明 |
|------|------|
| `running_reqs` | 即将执行的请求列表；为便于查找，可用 map 管理 |
| `running_step_token_num / running_step_logits_num` | 当前已规划请求在该 DP 上的 step 级工作量统计；负载均衡策略以此作为主要观测量之一 |

### 3.4 PlanningBatch

表示**当前已规划的一个 Batch**（可能跨多 DP）。

| 字段 | 说明 |
|------|------|
| `batch_idx` | 在流水线并行中，多个 batch 可能同时存在；同一 `batch_idx` 同时只能有一个 batch 在执行 |
| `dp_batches` | `vector<DpBatch>`，按 `dp_idx` 直接索引各 DP 上的子 batch |

### 3.5 ScheduleOutput

与现有 **ScheduleOutput** 保持一致，由 `LaunchBatch()` 根据 PlanningBatch 生成。

### 3.6 调度后端：ScheduleBackendInterface

调度器**不直接依赖 BatchSchedulerInterface**，仅依赖本接口。主要接口如下：

```cpp
class ScheduleBackendInterface {
 public:
  explicit ScheduleBackendInterface(const BatchSchedulerConfig& batch_scheduler_config);
  virtual ~ScheduleBackendInterface() = default;
  virtual void PreSchedule(const ScheduleContext& context,
                           const std::vector<std::shared_ptr<PlanningBatch>>& planning_batches,
                           std::vector<ScheduleBackendEvent>& events);
  virtual void Schedule(const ScheduleContext& context,
                        std::vector<std::shared_ptr<PlanningBatch>>& planning_batches,
                        std::vector<ScheduleBackendEvent>& events) = 0;
  virtual void UpdateWithGenerationResult(size_t multi_batch_id,
                                          const GenerationOutputGroup& generation_output,
                                          const ScheduleContext& context,
                                          std::vector<ScheduleBackendEvent>& events) = 0;
  virtual void Stop() = 0;
};

struct ScheduleBackendEvent {
  ScheduleBackendEventType type;
  size_t dp_idx;
  std::shared_ptr<InferRequest> req_ptr;  // 供延迟请求级操作使用
  Status status;  // 事件携带状态，如 timeout / alloc failed 等
};
```

- **Schedule(context, planning_batches, events)**：后端只读 `ScheduleState`，输出本轮可执行的 `planning_batches` 与 `events`，不直接修改 `ScheduleState`。
- `ScheduleBackendEvent` 采用**一事件一请求**语义（每个 event 只包含一个 req_ptr）。
- **PreSchedule(context, planning_batches, events)**：基类 `ScheduleBackendInterface` 提供通用实现，在正式 `Schedule` 前执行轻量检查（如 timeout 扫描）；若发现超时请求，输出 stop 事件供调度器先摘除状态后再调度。timeout 判定使用本轮调度开始时间 `context.schedule_start_time_in_ms` 与 `req->timestamp_in_us / 1000 + batch_scheduler_config_.waiting_timeout_in_ms` 比较。子类可复用或按需重载。
- **UpdateWithGenerationResult(multi_batch_id, generation_output, context, events)**：处理生成结果并输出 `events`。对于生成过程中已结束的请求，后端通过 stop 事件通知调度器；`ScheduleState` 的实际变更由调度器主线程执行。
- **Stop()**：请求后端停止。
- 默认后端实现为 **ContinuousBatchingBackend**（参考 ContinuousBatchingStrategy）：
  - `ProcessDecodingQueue/ProcessPrefillingQueue` 在轮询队列时根据 block 分配结果直接决策并产出 events；请求能被规划的判定基于 `HasPlanningTask()==false`。
  - 请求加入 `DpBatch.running_reqs` 前必须调用 `SetPlanningTask()`，固定本轮 planning 状态。
  - `ProcessDecodingQueue` 的 OOM 兜底：若当前 DP 的 `running_reqs` 为空且 decoding 队列本轮候选请求全部分配失败，则优先从 `prefilling_reqs` 中选择“已持有 kv blocks”的请求产出 `kRecomputeRequest`；若不存在此类请求，再从 decoding 候选中选择一个请求产出 `kRecomputeRequest`。
  - 支持 `kRecomputeRequest / kStopRequest / kMoveToDecode` 等事件；其中 `kMoveToDecode` 用于将请求从 `prefilling_reqs` 迁移到 `decoding_reqs`。
  - `UpdateWithGenerationResult` 中仅在“完成判定”时产出 `kStopRequest`；`stop/abort` 仅用于决定是否继续执行 cache 状态更新，超时等其余终止状态由后续 `PreSchedule` 统一处理。
  - 调度器只消费 backend events，不再在外层额外扫描状态生成同类事件。

### 3.7 SchedEvent（调度事件）

用于驱动调度主线程的**事件**。

| 字段 | 说明 |
|------|------|
| `event_info` | `std::shared_ptr<SchedEventInfoBase>`，具体事件 payload；事件类型由 `event_info->GetType()` 提供 |

**事件类型**（当前设计）：

- `EventAddRequest`
- `EventUpdateAfterGeneration`
- `EventAsyncTaskFinished`
- `EventGetScheduleResult`：消费者请求下一个结果；主线程**只有收到此事件后**才将 PlanningBatch 组织成 ScheduleResult 放入 result_queue_。

---

## 4. 事件数据结构

各事件类型对应的 payload 结构如下。

### 4.1 EventAddRequest

直接调用 `GetHookManager()->Process(requests, context)`，根据返回的 **ReqIdToNextStep** 将请求放入对应队列：`kContinue` → prefilling_reqs，`kWaitAsyncResult` → preprocessing_reqs；`kFailAndEnd` 或未出现在 map 中的请求不进入任一队列。AddRequest 的请求一定会进入 prefilling_reqs 或 preprocessing_reqs（由责任链结果决定）。

```cpp
struct EventAddRequestInfo {
  std::vector<std::shared_ptr<InferRequest>> reqs;
};
```

### 4.2 EventUpdateAfterGeneration

```cpp
struct EventUpdateAfterGenerationInfo {
  GenerationOutputGroup gen_results;
};
```

### 4.3 EventAsyncTaskFinished

单条异步任务结果与事件 payload 定义如下：

```cpp
// 单条异步任务完成结果（原 EventAsyncTaskFinishedInfo 重命名）
struct AsyncTaskResult {
  int hook_idx;  // 标识从哪个 Hook 开始调用，调用顺序为 hook_idx 从小到大
  int task_id;
  int task_ret_state;
};

// 异步任务完成事件：包含多条 AsyncTaskResult
struct EventAsyncTaskFinishedInfo {
  std::vector<AsyncTaskResult> task_results;
};
```

### 4.4 EventGetScheduleResult

`GetNextScheduleResult()` 会先向事件队列发送本事件，再阻塞等待 `result_queue_`。主线程处理本事件时**仅设置标志位** `pending_get_schedule_result_`。

**调度时机**：`AsyncSchedulerLoop` 在每轮先 `Get()` 一个事件并 `DispatchEvent`，再用 `NonBlockingGet()` 尽量排空当前队列并逐一 `DispatchEvent`；随后统一执行 `RunScheduleForBatches()`（内部先 `PreSchedule` 再 `Schedule`，并应用 backend events）。若 `pending_get_schedule_result_` 为 true，主线程会尝试 `TryBuildAndLaunchScheduleResult()`；成功时会立刻再跑一轮 `RunScheduleForBatches()` 预热下一次结果。`ApplyBackendEventsToState()` 对 Stop/Recompute 事件采用”两阶段处理”：先把请求从 `ScheduleState` 与 `DpBatch.running_reqs` 摘除；若 `HasInflightTask()==true` 则写入延迟队列，待后续 `ProcessDelayedBackendEvents()` 条件满足再执行请求级操作。

`pending_get_schedule_result_` 的语义可视为“二阶段提交”：

1. **阶段一（意图登记）**：收到 `EventGetScheduleResult` 时，若当前不可 launch，仅将 `pending_get_schedule_result_` 置为 true，不立即返回空结果。
2. **阶段二（提交发布）**：当后续事件驱动使 batch 变为 launchable，`TryBuildAndLaunchScheduleResult()` 在 `result_queue_.Put(result)` 前先清除该标志位，再投递结果，避免并发观察到“已投递但标志仍为 true”的重复构建窗口。

---

## 5. 接口设计

### 5.1 Hook 对象接口：ScheduleHookInterface

单个 Hook 的抽象，用于在责任链上对请求做预处理、校验或异步任务协调等。

#### 5.1.1 NextStep 与 ReqIdToNextStep

**NextStep** 表示责任链上对单个请求的**下一步指引**。**ReqIdToNextStep** 为独立类型（不定义在 ScheduleHookInterface 下），表示「req_id → NextStep」的映射，供 Manager 与调度器根据处理结果做请求移动。

```cpp
enum class NextStepAction {
  kContinue,           // 继续交给后续 Hook 处理；整条链执行完后表示可进入 prefilling_reqs
  kWaitAsyncResult,    // 等待异步结果，当前不继续向后传递；调度器放入 preprocessing_reqs
  kFailAndEnd,         // 失败并结束处理
};

struct NextStep {
  NextStepAction action;
  int task_id = 0;  // 可携带与 action 相关的附加信息
};

// 独立类型：req_id -> NextStep，描述每个请求经责任链处理后的下一步
using ReqIdToNextStep = std::unordered_map<int64_t, NextStep>;
```

#### 5.1.2 主要接口

| 接口 | 说明 |
|------|------|
| `Process(requests, context)` | 对当前批次的请求做同步处理，返回 **ReqIdToNextStep**，由调用方根据结果将请求移入 prefilling_reqs / preprocessing_reqs 等。 |
| `UpdateAsyncResult(finished_tasks, context)` | 根据异步任务完成情况更新状态，返回 **ReqIdToNextStep**，由调用方根据结果做请求移动。入参为 `std::vector<AsyncTaskResult>`。 |

```cpp
class ScheduleHookInterface {
 public:
  using InferRequestPtr = std::shared_ptr<InferRequest>;

  virtual ~ScheduleHookInterface() = default;

  virtual ReqIdToNextStep Process(
      const std::vector<InferRequestPtr>& requests,
      ScheduleContext& context) = 0;

  virtual ReqIdToNextStep UpdateAsyncResult(
      const std::vector<AsyncTaskResult>& finished_tasks,
      ScheduleContext& context) = 0;

  virtual void SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback callback) = 0;
};
```

---

### 5.2 对象管理器接口：ScheduleHookManagerInterface

管理 Hook 责任链，按顺序调用各 Hook，并根据 NextStep 决定是继续传递、等待异步还是结束。

#### 5.2.1 主要接口

| 接口 | 说明 |
|------|------|
| `AddHook(hook)` | 向链尾添加一个 Hook（`std::shared_ptr<ScheduleHookInterface>`）。 |
| `Process(requests, context)` | 按 Hook 链顺序（hook_idx 从小到大）调用各 Hook 的 `Process`。Manager 内部维护活跃请求列表：每轮先并入 `start_hook_idx == cur_hook_idx` 的请求，再调用当前 Hook；`kContinue` 保留到下一轮，其余写入结果并移出。**返回 ReqIdToNextStep**。 |
| `UpdateAsyncResult(finished_tasks, context)` | 先按 hook_idx 从小到大调用各 Hook 的 `UpdateAsyncResult`（每个 Hook 只接收本 hook_idx 的 task 子集），将返回为 `kContinue` 的请求汇总成 `req_id -> start_hook_idx(hook_idx+1)`；之后统一调用一次内部责任链续跑逻辑。若返回的 req_id 不在 `preprocessing_reqs`，记录 error 并忽略该条结果。**返回 ReqIdToNextStep**。 |
| `SetAsyncTaskFinishedCallback(callback)` | 设置 Hook 异步任务完成时的回调；通常由调度器设为 `UpdateAfterAsyncTaskFinished`，HookManager 将该回调下发给各 Hook，Hook 在异步执行完成时直接调用该回调。 |

```cpp
using AsyncTaskFinishedCallback = std::function<void(const EventAsyncTaskFinishedInfo&)>;

class ScheduleHookManagerInterface {
 public:
  virtual ~ScheduleHookManagerInterface() = default;

  virtual void AddHook(std::shared_ptr<ScheduleHookInterface> hook) = 0;

  virtual ReqIdToNextStep Process(
      const std::vector<InferRequestPtr>& requests,
      ScheduleContext& context) = 0;

  virtual ReqIdToNextStep UpdateAsyncResult(
      const std::vector<AsyncTaskResult>& finished_tasks,
      ScheduleContext& context) = 0;

  virtual void SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback callback) = 0;
};
```

**行为约定：**

- `Process`：内部执行按 hook_idx 递增的责任链。支持 `req_id -> start_hook_idx` 的起始位置控制（对外 `Process` 默认全部从 0 开始）。每轮先将 `start_hook_idx == cur_hook_idx` 的请求加入活跃列表，再调用当前 Hook；返回 `kContinue` 的请求继续留在活跃列表用于下一个 Hook，非 `kContinue` 立即写入结果并从活跃列表移除；所有 Hook 结束后仍活跃的请求记为 `kContinue`。
- `UpdateAsyncResult`：先按 hook_idx 从小到大依次调用 `hook->UpdateAsyncResult`，每个 Hook 只处理本 hook_idx 的任务子集；对返回 `kContinue` 的请求构造 `req_id_to_start_hook_idx[req_id] = hook_idx + 1`，然后统一调用一次内部续链流程。若 `UpdateAsyncResult` 返回的 req_id 不在 `preprocessing_reqs`，视为状态异常，打印 error 并忽略该条返回。最终返回 **ReqIdToNextStep** 供调度器移动请求。

---

### 5.3 调度器接口

调度器实现**继承自 ScheduleProcessorInterface**，并增加异步任务完成后的入口。

#### 5.3.1 与 ScheduleProcessorInterface 的关系

- 调度器对外表现为「事件驱动的调度处理器」，实现现有 `ScheduleProcessorInterface` 中定义的接口（如事件入队、主循环、与 PlanningBatch/ScheduleState 的交互等）。
- 调度逻辑仅依赖 **ScheduleBackendInterface**：backend 只读 `ScheduleState`，输出 `PlanningBatch + events`；`ScheduleState` 的实际改动（如 stop/recompute）统一在调度主线程执行，不直接依赖 BatchSchedulerInterface。

#### 5.3.2 扩展接口

**SetScheduleHookManager(ScheduleHookManagerInterface\* hook_manager)**  
设置外部 Hook 管理器；若为 `nullptr` 则使用内置默认管理器。设置时会自动将 `UpdateAfterAsyncTaskFinished` 通过 `hook_manager->SetAsyncTaskFinishedCallback(...)` 注册为该管理器的异步完成回调，使 Hook 异步执行完成后可通过该回调驱动调度器。

**UpdateAfterAsyncTaskFinished**

异步 task 完成时，由调度主线程统一回调，用于更新 Hook 状态并驱动后续调度。

```cpp
// 在调度器类上增加的接口（或在 ScheduleProcessorInterface 中扩展）
void UpdateAfterAsyncTaskFinished(
    const EventAsyncTaskFinishedInfo& event_info);
```

| 参数 | 说明 |
|------|------|
| `event_info` | 本次完成的异步任务事件；类型为 `EventAsyncTaskFinishedInfo`，其内 `task_results` 为 `AsyncTaskResult` 数组，每条 `AsyncTaskResult` 包含 `hook_idx`、`task_id`、`task_ret_state`。 |

**行为约定：**

- 调度主线程在处理 `EventAsyncTaskFinished` 事件时，将事件 payload 传入并调用 `UpdateAfterAsyncTaskFinished`。调度器将 `event_info.task_results` 传给 ScheduleHookManager 的 **UpdateAsyncResult**。Manager 先按 hook_idx 顺序调用各 Hook 的 `UpdateAsyncResult`，再将可继续请求按 `req_id -> hook_idx+1` 统一续跑责任链；若返回 req_id 不在 `preprocessing_reqs`，打印 error 并忽略。
- 调度器根据 **UpdateAsyncResult** 返回的 **ReqIdToNextStep** 做请求移动：kContinue 的请求从 **preprocessing_reqs** 移入 **prefilling_reqs**，其余按 action 保持等待或结束。
- `RunScheduleForBatches` 阶段，调度器先调用 backend 的 `PreSchedule(const ScheduleContext&, planning_batches, events)` 处理 timeout 等预检查事件，再调用 `Schedule(const ScheduleContext&, planning_batches, events)` 获取本轮可执行 batch；两类 events 都按主线程两阶段处理：先从 `ScheduleState` 队列和 `DpBatch.running_reqs` 移除请求；请求级 Stop/Recompute 若遇到 `HasInflightTask()==true` 则写入延迟队列，后续轮次在 `HasInflightTask()==false` 时再执行。

---

### 5.4 负载均衡接口设计

为支持多 DP 场景下的动态负载协调，调度器引入独立的负载均衡接口。接口统一读取 `ScheduleContext` 中的负载均衡配置/观测信息，并结合当前 `ScheduleState` 与 `PlanningBatch` 进行决策。

#### 5.4.1 主要接口

| 接口 | 说明 |
|------|------|
| `AddInferRequests(std::vector<InferRequestPtr>& requests, const ScheduleContext& context)` | 根据 `ScheduleContext` 为新增请求指定 `attn_dp_group_id`，并将请求路由到目标 DP。该函数在 `OnEventAddRequest` 中被调用，先于 Hook 责任链处理。 |
| `GetDpSchedulingOrderAndQuota(const PlanningBatch& batch, const ScheduleContext& context, InferStage stage)` | 在 backend 调度阶段返回本轮 DP 调度序与每轮配额 `vector<WorkloadQuota>`（`WorkloadQuota={dp_idx, step_token_limit}`）。backend 逐项消费该序列，以控制“先调度谁、该 DP 本轮最多可吃多少 step token”。 |
| `RecordInflightBatch(const PlanningBatch& batch, const ScheduleContext& context)` | 在 batch 成功 launch 后记录每个 DP 的在飞负载（inflight workload），用于跨轮反馈。当前 `BasicWorkloadBalancer` 预留该接口，尚未启用 inflight 反馈。 |

```cpp
class EventDrivenScheduleProcessor : public ScheduleProcessorInterface {
 public:
  void AddInferRequests(std::vector<InferRequestPtr>& requests,
                        const ScheduleContext& context);

  std::vector<WorkloadQuota> GetDpSchedulingOrderAndQuota(
      const PlanningBatch& batch,
      const ScheduleContext& context,
      InferStage stage);

  void RecordInflightBatch(const PlanningBatch& batch,
                           const ScheduleContext& context);
};
```

#### 5.4.2 行为约定与调用时序

1. **请求接入阶段（OnEventAddRequest）**
   - 调用 `AddInferRequests(...)`，完成请求到 `attn_dp_group_id` 的映射；
   - `attn_dp_group_id` 写回请求对象，后续 Hook/Backend/状态迁移均沿该 DP 处理；
   - 当 `ScheduleContext` 未提供有效负载均衡信息时，使用默认策略（如 round-robin 或固定组）保证可用性。

2. **调度前阶段（Schedule 之前）**
   - backend 在 `ProcessPrefillingQueue/ProcessDecodingQueue` 开始时，调用 `GetDpSchedulingOrderAndQuota(...)`；
   - 返回值是 `vector<WorkloadQuota>`，每个元素给出一个 `dp_idx` 及该轮 `step_token_limit`；
   - backend 按返回顺序遍历各 DP 并尝试放入请求，`IsRequestAcceptable(...)` 同时受 `quota.step_token_limit` 与全局 `max_step_token_num_` 双重约束。

3. **调度发布后阶段（TryBuildAndLaunchScheduleResult）**
   - 当 `result_queue_.Put(result);` 成功后，调用 `RecordInflightBatch(...)`；
   - 将本次已 launch 的各 DP 负载写入调度器内部 inflight 统计（策略可扩展项）；
   - 当前 `BasicWorkloadBalancer` 中该接口为 no-op，后续可用于引入跨轮历史反馈。

#### 5.4.3 BasicWorkloadBalancer 当前策略（与代码对齐）

- **请求入口分配（`AddInferRequests`）**
  - 单 DP：所有请求固定写入 `attn_dp_group_id = 0`；
  - 多 DP：按队列负载均衡选择目标 DP
    `preprocessing_reqs + prefilling_reqs + decoding_reqs` 最小优先；
  - 若负载相同，按 round-robin 距离 `next_dp_idx_` 做 tie-break，避免长期偏置。

- **调度序与配额（`GetDpSchedulingOrderAndQuota`）**
  - 单 DP：直接返回 `{dp_idx=0, step_token_limit=max_step_token_num_}`；
  - 多 DP：
    1. 先按 `batch.dp_batches[dp_idx].running_step_token_num` 升序排序（轻载优先）；
    2. 生成两轮序列（同一 DP 顺序重复两次）；
    3. 第一轮使用小配额 `small_quota = max(1, max_step_token_num_ / (2 * dp_num))`，避免前序 DP 吃掉全局预算；
    4. 第二轮使用大配额 `large_quota = max(1, max_step_token_num_ / dp_num)`，提升整体资源利用率。

#### 5.4.4 与现有状态的关系

- `AddInferRequests` 负责“请求入场归属”，与 `DpScheduleState` 的队列落点一致。
- `GetDpSchedulingOrderAndQuota` 负责“本轮顺序 + 额度计算”，输出由 backend 直接消费。
- `RecordInflightBatch` 负责“历史负载反馈”扩展点，当前基础实现尚未启用。

三者组成完整负载均衡框架：**请求分配（入口） -> 顺序与额度控制（调度前） -> 在飞反馈（调度后，可扩展）**。

---

### 5.5 错误恢复与关闭流程

本节补充正常调度路径之外的关键状态机行为，便于排障与维护。

#### 5.5.1 DelayedBackendEvents 设计意图与触发时机

- **设计意图**：将”状态摘除”和”请求级副作用执行（stop/recompute）”解耦，避免对仍有 in-flight task 的请求提前执行破坏性动作。
- **触发时机**：
  - 在 `ApplyBackendEventsToState()` 中，Stop/Recompute 事件会先把请求从队列与 `running_reqs` 摘除；
  - 若发现 `HasInflightTask()==true`，则事件进入 `delayed_backend_events_`；
  - 后续在 `OnEventUpdateAfterGeneration` 后调用 `ProcessDelayedBackendEvents()`，当请求变为无 in-flight task 时再真正执行请求级操作。
- **收益**：保证”后续调度不可再选中该请求”，同时避免”请求级操作与正在执行的任务并发冲突”。

#### 5.5.2 `pending_get_schedule_result_` 的二阶段语义

- 它不是“是否需要调度”的标记，而是“是否有消费者在等待下一个可发布结果”的意图标记。
- 第一阶段由 `EventGetScheduleResult` 设置等待意图；第二阶段由 `TryBuildAndLaunchScheduleResult()` 成功构建后清标志并投递结果。
- 主线程在成功发布一个结果后会立即再跑一轮 `RunScheduleForBatches()`，用于预热下一次结果，降低下一次 `GetNextScheduleResult()` 的等待时延。

#### 5.5.3 Stop() 有序关闭顺序

当前实现中，`EventDrivenScheduleProcessor::Stop()` 的关闭顺序为：

1. 设置 `terminated_ = true`，阻止调度循环继续正常处理；
2. 调用 `backend_->Stop()`，通知后端停止继续产生活跃调度行为；
3. 向 `event_queue_` 注入空事件，唤醒可能阻塞在 `Get()` 的调度主线程；
4. `join` 调度主线程，确保主循环退出完成；
5. 向 `result_queue_` 投递 `nullptr`，唤醒可能阻塞的消费者。

说明：
- Hook 线程池生命周期不由调度器内核直接管理；应由 Hook/HookManager 持有方在系统关闭路径中执行线程池停机与任务收敛。

---

## 6. 小结

- 调度器改为**单主线程 + 事件队列**，所有调度相关触发统一为事件，便于在 launch 前统一更新状态并做调整。
- 通过**责任链 + Hook Manager + 异步 callback** 将耗时逻辑移出主线程，同时保证结果仍通过事件回填到调度状态。
- **ScheduleState / DpScheduleState / PlanningBatch / DpBatch** 形成清晰的分层状态，为多 DP 负载协调和策略扩展提供基础。
- 事件类型与 payload 结构可随需求扩展，保持接口稳定、实现可迭代。
