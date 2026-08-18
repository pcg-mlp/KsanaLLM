---
spec_version: "1"
spec_type: change
id: "1070131059135905223"
status: draft
subdomain: scheduler
cross_subdomains: []
complexity: L
work_type: feature
---

### A. 工作类型（work_type）【必填，先于目标】
`feature` — 功能实现（新增能力/配置/对外行为）

### B. 目标（goal）
现状：`csrc/batch_scheduler/event_driven_scheduler/` 下的调度器测试（如 `event_driven_scheduler_test.cpp`）目前只覆盖单机场景，通过 `BatchSchedulerEnvironmentSimulator`/`FakedBlockAllocatorGroup` 模拟 block manager 操作与 token 生成。而 PD v2（`csrc/pd_v2/`）引入了跨节点的 Prefill/Decode 调度协同（Decode 预分配 KV block 并携带目标地址发起请求，Prefill 计算后写回 KV，Decode 收到完成通知后继续解码），这部分多节点调度协同逻辑目前没有对应的自动化测试覆盖，出错风险高且不易排查。

目标：为 PD v2 event driven scheduler 编写一个可参数化的 M（Prefill）/N（Decode）节点调度协同模拟测试框架：跨节点通信通过测试专用 `FakePdV2Connector`（直接实现 `PdV2ConnectorInterface`）+ 进程内 `PdV2FakeConnectorRegistry` 地址簿与共享 `ThreadPool` 异步派发完成；复用 `batch_scheduler_test` 中对 block manager 操作与 token 生成的模拟方式；Prefill 侧用 `PdV2PrefillIntakeAdapter` 接入、`SimulatedKvWriteBridge` 模拟 KV 写回；并基于该框架落地一批针对 PD 分离场景下调度协同正确性的单测程序。

**为 PD v2 event driven scheduler 补齐多节点调度协同的单测能力，验证 Decode/Prefill 在固定/轮询路由下协同工作的正确性与无死锁性。**

> 修订说明（r2，与 design 二次迭代对齐）：不再复用 `csrc/pd_v2/mooncake/fake_tent` / 真实 `MooncakeEngineConnector`；改判依据见 `1-change-plan.md §A` 第 2/5/6 点。

### C. 非目标（non_goals）
- 不做真实 RDMA/网络传输性能测试；不接入真实 Mooncake/RDMA，也不复用 `fake_tent` mock 传输层（跨节点走测试专用 `FakePdV2Connector`）
- 不接入真实 `MooncakeEngineConnector`（及其内嵌的 `IpcFuncWrapper`/`HostStagePool`/`PeerCredits`、Redis 发现/心跳后台线程）
- 不做真实 Redis 服务发现/心跳机制的集成测试
- 不做真实多进程/多机部署端到端测试（含 python `serving_server` 层）；全部在单进程内以多个 `EventDrivenScheduleProcessor` 实例 + fake connector 模拟
- 不验证 `PeerSelector`/`LinUCBRouter` 真实路由算法本身的正确性（已有独立单测 `peer_selector_test.cpp`/`linucb_router_test.cpp` 覆盖）；本框架路由决策使用简化固定/轮询策略，聚焦调度协同流程本身
- 不接入真实 `PdV2KvDispatcher`/`PdV2KvBarrier` 按层 RDMA 写入路径（Phase B 后续迭代）
- 不覆盖节点故障/掉线场景（Redis TTL 过期后的资源清理等），留待后续迭代补充

### D. 影响面（impact_surface）
- `csrc/batch_scheduler/event_driven_scheduler/pd_v2_multi_scheduler_test.cpp`、`pd_v2_multi_scheduler_test_helper.h/.cpp`（新增：多节点协同模拟测试与 `FakePdV2Connector`/`PdV2FakeConnectorRegistry`/`PdV2PrefillIntakeAdapter`/`SimulatedKvWriteBridge`/`MultiNodeTopology` 等测试基础设施）
- `csrc/batch_scheduler/batch_scheduler_test_helper.h`、`csrc/batch_scheduler/batch_scheduler_test_client.h`、`csrc/batch_scheduler/batch_scheduler_test.h`（复用现有 block manager 与 token 生成 mock 能力；原则上不改产品逻辑，测试工程可做必要的 ODR/可见性修正）
- `csrc/pd_v2/decode/pd_v2_decode_hook.h/.cpp`（被测：真实 Decode hook，只读不改）
- `csrc/pd_v2/public/pd_v2_connector.h`、`pd_v2_types.h`、`pd_v2_config.h`（`FakePdV2Connector` 实现的接口/类型来源，只读不改）
- `csrc/runtime/threadpool.h`（复用：registry 异步派发）
- `csrc/batch_scheduler/CMakeLists.txt`（新增 `pd_v2_multi_scheduler_test` 目标；不依赖 `PD_V2_MOONCAKE_*`/`fake_tent`/`libhiredis.a`）

本次**不涉及**：`csrc/pd_v2/mooncake/fake_tent/`、`mooncake_engine_connector.*`、`mooncake_executor_connector.*`、`PdV2KvDispatcher`/`PdV2KvBarrier` 生产实现。

### E. 行为不变量（invariants）
- 每个请求最终产生的 token 序列与单机 baseline（同一 seed/prompt 直接跑单机 event driven scheduler）一致
- Decode 预分配的 KV block 与 Prefill 写入的目标地址严格匹配，不发生越界、不占用其它请求的 block
- 固定/轮询路由下，各 Prefill/Decode 节点调度器协同过程中不出现死锁或请求悬挂——所有请求最终都能完成或明确失败
- 并发多请求跨节点调度时，请求间的 KV/token 结果互不串扰

### F. 功能分支（functional_branches）
- Prefill 节点资源不足/繁忙时，Decode 侧的等待或退避行为
- 多个 Decode 并发向同一个 Prefill 发起请求（并发写入不同请求的 KV block）
- prefix cache / global cache 命中场景下直接本地完成，不发往 Prefill
- Prefill 计算完成到 Decode 收到完成通知之间的时序竞争（KV 写入未完成前 Decode 不应误读）
- 多 P 对 1 D 或 1 P 对多 D 的一对多路由分配正确性（不丢失、不重复分配）
- SWA（滑窗注意力）场景下跨节点 KV 传输/裁剪的正确性（Decode 侧 SWA 窗口范围与 Prefill 写入的 KV 范围保持一致）
- prefix cache 与跨节点调度的组合场景（请求部分命中 prefix cache、剩余部分需发往 Prefill 补齐计算的分支路径，与 T05 纯命中场景区分）

### G. 建议测试点（suggested_tests）
- id: T01
  scenario: 1P1D 基础协同流程（Decode 发起请求 → Prefill 计算 → 经 `FakePdV2Connector`/`SimulatedKvWriteBridge` 写回 KV → Decode 完成解码）
  expect: 请求正常完成，最终 token 序列与单机 baseline 一致
  covers: token_consistency 不变量
  check_type: unit
- id: T02
  scenario: KV block 地址匹配校验
  expect: Prefill 写入位置与 Decode 预分配 block 精确对应，不误写其它请求的 block
  covers: kv_addr_match 不变量
  check_type: unit
- id: T03
  scenario: 2P2D 等多节点固定/轮询路由下并发调度
  expect: 所有请求最终完成或明确失败，不出现死锁/悬挂
  covers: no_deadlock 不变量、一对多路由分配正确性分支
  check_type: unit
- id: T04
  scenario: 并发多请求跨节点调度
  expect: 各请求 KV/token 结果互不串扰
  covers: no_cross_talk 不变量、多个 Decode 并发向同一 Prefill 发起请求分支
  check_type: unit
- id: T05
  scenario: prefix/global cache 命中场景
  expect: 命中缓存的请求直接本地完成，不产生发往 Prefill 的请求
  covers: cache 命中跳过 Prefill 功能分支
  check_type: unit
- id: T06
  scenario: Prefill 资源不足/繁忙
  expect: Decode 侧表现出预期的等待/退避行为，请求不丢失，最终能完成或明确超时失败
  covers: Prefill 繁忙功能分支
  check_type: unit
- id: T07
  scenario: Prefill 完成通知与 KV 写入完成之间的时序竞争
  expect: Decode 不会在 KV 完全落地前提前读取，结果始终正确
  covers: 完成通知时序竞争功能分支
  check_type: unit
- id: T08
  scenario: SWA（滑窗注意力）场景下的跨节点协同（参考单机 `SwaBasicTokenGenerationTest` 的验证方式，扩展到跨节点）
  expect: Decode 侧 SWA 窗口范围内的 KV 内容与单机 SWA baseline 一致，token 生成正确
  covers: SWA 与多节点协同交互功能分支
  check_type: unit
- id: T09
  scenario: prefix cache 部分命中场景（请求部分内容命中本地 cache，剩余部分需发往 Prefill 计算）
  expect: 命中部分复用本地 cache 不重复计算，未命中部分正确通过跨节点协同补齐，最终 token 序列正确
  covers: prefix cache 与跨节点调度组合功能分支
  check_type: unit

### H. 资源/硬件约束（resource_constraints）
无（纯单进程内模拟，不涉及真实 GPU/显存/网络）

### I. 回退方案（rollback）
新增测试代码，不影响现有产品逻辑；如需回退，直接删除/不启用新增测试目标（CMake target）即可，无需数据迁移或配置开关。

### J. 风险备注（risk_notes）
`FakePdV2Connector` + `SimulatedKvWriteBridge` 与真实 `MooncakeEngineConnector`/RDMA 行为存在差异（例如真实暂存写协议、异步完成时序、网络分区/丢包、Mooncake VA 换算等），本测试框架通过后不能完全代表线上真实环境下的正确性，仍需结合线上灰度/集成测试验证；真实按层 RDMA 路径覆盖属于 Phase B，不在本次范围。

### K. 需要人确认的点（待确认清单）
- [x] 跨节点通信是否仍复用 `fake_tent` / 真实 `MooncakeEngineConnector` → **否**：改判为测试专用 `FakePdV2Connector` + `PdV2FakeConnectorRegistry`（design 二次迭代已落地，本轮 clarify 同步回写）
- [x] ~~`fake_tent` 是否支持 M/N≥2 拓扑~~ → 已作废（不再依赖 `fake_tent`）；M/N 拓扑改为构造时 `SetPeerAddrs` 显式注册
- [ ] [?] T08（SWA 跨节点）当前实现未实质覆盖（`sliding_window_size=0`；评审 FIX-6 major SPEC_DRIFT）——是否在本需求内补齐，还是显式降级为后续迭代？
