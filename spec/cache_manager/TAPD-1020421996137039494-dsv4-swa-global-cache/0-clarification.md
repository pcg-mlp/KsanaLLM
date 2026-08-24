---
spec_version: "1"
spec_type: change
id: "1020421996137039494"
status: draft
subdomain: cache_manager
cross_subdomains: [scheduler, distributed]
complexity: XL
work_type: feature
---

> TAPD：[【[feat] DSv4 SWA 全局 cache（Mooncake Put/Get）方案设计与实现】](https://tapd.woa.com/tapd_fe/20421996/story/detail/1020421996137039494)  
> 反写说明：本澄清由已合入提交 `6a1fe8d724e5a35c405dbb4d6cf84aa2875906b2` 回溯，描述**已落地行为**，不是待开发需求。按「影响面 ≥6 模块」定为 XL。`work_type=feature` 按提交 `feat(global-cache)` 与 TAPD 标题落地。

### A. 工作类型（work_type）【必填，先于目标】
`feature` — 功能实现（新增能力/配置/对外行为）

次要动机：SWA 稀疏存取降低全局 cache 占用（perf）、IPC 并行度可配（refactor 成分）。主类型仍是 feature。

### B. 目标（goal）

问题 1：全局 KV cache（Mooncake）原先按主 KV（FA）连续前缀存取；DSv4 等 SWA 模型的 SWA 块未压缩（约 7.6× 主 KV），全量 Put 会把全局 cache 占用翻倍，且 Get 若不按滑窗对齐会把「窗口外无 SWA」的 FA 命中当成可用前缀。

问题 2：启用全局 cache 需要额外部署 `mooncake_master`，serving 与 master 端口易不一致，本地/单机联调成本高。

问题 3：IPC heavy long-task 出队并行度写死，与 `fetch_thread_num` 不对齐时，Query H2D stream 无法 1:1 绑定。

**在已有 Mooncake 全局 cache 上补齐 SWA 稀疏 Put/Get、serving 内按需拉起 master，并把 IPC long-task 并行度做成可配且与 fetch 线程数启动对齐。**

### C. 非目标（non_goals）

- 不改 PD v2 RDMA KV 下发（`pd_v2` Mooncake connector / KvDispatcher）语义
- 不改 LinUCB / 路由决策
- 不把 SWA 块改为压缩存储，也不做自适应选点（固定「尾部 N 密存 + 前段等分中位点稀存」）
- 不在本提交引入增量 prefix hash 复用优化（后续性能项）
- 不保证未开 `ENABLE_GLOBAL_CACHE_HOOK` 或未配 `service_endpoint` 时有任何全局 cache 行为

### D. 影响面（impact_surface）

主路径：`csrc/global_cache_connector/`（Mooncake 真/mock、SWA key、`AlignHitBoundaryBySwa`、`known_exist`）、`csrc/executor/global_cache_put_worker.*`（SWA 选点 + 快照）、`csrc/cache_manager/`（`DropUnbackedSwaBlocks` / `BeginSwaPutHold` / `EndSwaPutHold`）、`csrc/batch_scheduler/`（fetcher 聚合 SWA 对齐命中、put tracker 跨卡累加 FA/SWA 块数）、`csrc/ipc/`（query/put done 携带 SWA 字段；`long_task_thread_num` 注入）、`python/ksana_llm/global_cache_bootstrap.py`（serving 内 master）、`csrc/configure/schedule_config_parser.*`、`examples/ksana_llm.yaml` / `README.md`。

推理路径：event-driven scheduler（`scheduler_type: 1`）+ `ENABLE_GLOBAL_CACHE_HOOK=1` + 非空 `service_endpoint`。训练路径不涉及。

### E. 行为不变量（invariants）

- I1：未配 `service_endpoint` 或未开 hook 时，不构造 connector、不拉 master、请求路径与本功能合入前一致
- I2：SWA Put 只上传满块；选点 = 尾部 `[B-N, B)` ∪ 前段 `parts` 等分后去掉首尾的中间份中位点，再与提交时刻可见的 `swa_kv_cache_block_idx_offsets` 求交
- I3：SWA Get 的可用前缀上界不超过 FA 连续命中，且该上界的末 `swa_window_block_num` 个位置必须「全局 SWA 存在 ∧ 本地有可写 SWA slot」；否则回退到更短边界
- I4：非 SWA 路径下，`known_exist` 覆盖本轮全部候选 FA 块，连续前缀中断后仍可拉取洞后命中块；SWA 路径先裁边界，边界外不再拉
- I5：异步 Put 提交时必须快照 `forwarding_tokens` 与 SWA block 列表；`PrefixCacheManager` 在 put 未完成前对滑出窗口的 SWA 做 hold，全部 device put done 或请求销毁时 `EndSwaPutHold`
- I6：全局命中拉长 FA 前缀后、forward 前必须 `DropUnbackedSwaBlocks`，避免把未写入的 SWA 标成 valid
- I7：FA/SWA put 成功块数按卡累加（与主 KV 分片上传一致），不是只记卡 0
- I8：启动时 `long_task_thread_num` 与 `fetch_thread_num` 若不一致，两侧都抬到 `max`；yaml 值 `<1` 钳到 1
- I9：hook 开启且本机 RPC 端口未监听时拉起 `mooncake_master`；已监听则跳过；找不到二进制则启动失败（抛错），不静默降级

### F. 功能分支（functional_branches）

- F1：SWA 未启用（无 SWA allocator / `swa_block_byte_size=0`）→ 只走主 KV Put/Get
- F2：满块数 `< N` → 尾部密存覆盖全部满块，前段中位点为空
- F3：SWA 边界上缺全局内容或本地 slot → 命中长度截断，不允许「FA 很长但窗口 SWA 空洞」
- F4：mock `known_exist` 与真 connector 同语义：SWA 裁边界 vs 非 SWA 洞命中
- F5：split-fuse 中间 chunk（`covered < input_tokens`）跳过 Put；仅覆盖完整 input 的最后一段 prefill 上传
- F6：`put_hotkey_window_sec>0` 时同 key 窗口内第二次才上传；`0` 关闭
- F7：IPC query 带 SWA context；SWA GetBlockPtrs 失败则回退只查主 KV
- F8：serving bootstrap：hook=0 直接返回；端口占用 skip；二进制缺失失败

### G. 建议测试点（suggested_tests）

- id: T01
  scenario: yaml 未配 `swa_tail_block_num`，只配 `service_endpoint`
  expect: parser 得到 `swa_tail_block_num=2`；`long_task_thread_num` 默认 4，与 fetch 对齐
  covers: I8、F1
  check_type: unit
  check_id: ScheduleConfigParserTest（`swa_tail_block_num==2`）

- id: T02
  scenario: `long_task_thread_num` 与 `fetch_thread_num` 不一致
  expect: 两侧都升到 max
  covers: I8
  check_type: unit
  check_id: LongTaskAndFetchThreadNumAlignToMax

- id: T03
  scenario: SWA Put 选点（尾部密存 + 前段等分中位）
  expect: 选中集合与公式一致；不满块不上传；无 SWA 配置不 put SWA
  covers: I2、F2
  check_type: unit
  check_id: PutWorkerSwaTest

- id: T04
  scenario: SWA Get 对齐窗口；缺 SWA 或本地 slot 截断
  expect: 边界回退；无 SWA 内容则无命中
  covers: I3、F3
  check_type: unit
  check_id: MockConnectorSwaTest

- id: T05
  scenario: 非 SWA 部分块 miss
  expect: 产生非连续 matched ranges（洞后仍可命中）
  covers: I4、F4
  check_type: unit
  check_id: PartialBlockMissProducesNonContiguousRanges

- id: T06
  scenario: 多卡 FA/SWA put done
  expect: tracker 跨卡累加 `fa_put_blocks` / `swa_put_blocks`
  covers: I7
  check_type: unit
  check_id: FaAndSwaPutBlocksSumAcrossDevices

- id: T07
  scenario: 全局命中后 DropUnbackedSwaBlocks
  expect: keep_start 之前的未 backing SWA 被释放；未知 req 失败
  covers: I6
  check_type: unit
  check_id: PrefixCacheManagerSWATest / DirectCacheManagerTest

- id: T08
  scenario: IPC query 携带 SWA 字段；GetBlockPtrs 失败
  expect: 序列化往返正确；失败回退主 KV
  covers: F7
  check_type: unit
  check_id: QueryCallableTest.Swa*

- id: T09
  scenario: hook=1 时 serving 拉起 / 跳过 master
  expect: 端口空闲拉起；已监听 skip；无二进制失败
  covers: I9、F8
  check_type: manual
  check_id: README Enable Global Cache at runtime

### H. 资源/硬件约束（resource_constraints）

- SWA 块远大于主 KV：必须稀疏 Put；`global_segment_size` yaml 默认 100GiB，小内存节点必须显式下调，否则 Mooncake `NO_AVAILABLE_HANDLE`
- 真后端依赖本机/集群 Mooncake Store；单测走 `mooncake-mock`，不要求 RDMA
- Query/Put 走独立 copy stream，不阻塞 CPU Forward，但会与 compute 争用拷贝引擎/显存带宽
- `ENABLE_GLOBAL_CACHE_HOOK=1` 时 serving 需要 `mooncake_master` 在 PATH 或 `MOONCAKE_MASTER_BIN`

### I. 回退方案（rollback）

1. 运行时关闭：去掉 `ENABLE_GLOBAL_CACHE_HOOK` 或把 yaml `service_endpoint` 置空（connector 不启用）
2. 配置回退：删除/注释 `setting.global_cache_connector` 与 `setting.ipc.long_task_thread_num`
3. 代码回退：revert `6a1fe8d724e5a35c405dbb4d6cf84aa2875906b2`（范围大，优先用开关）

### J. 风险备注（risk_notes）

- `GlobalCacheConnectorConfig.swa_tail_block_num` 结构体默认是 4，yaml 解析默认是 2；serving 走 parser，直接构造 C++ config 会得到不同 N
- `BeginSwaPutHold` 漏调或 `EndSwaPutHold` 未在销毁路径调用，会导致 SWA 块泄漏或 Put 读到已回收显存（错 key 静默污染全局 cache）
- serving 自动拉起的 master 与 yaml 端口不一致时，connector 会连错/连不上，表现像全 miss
- DirectCacheManager 的 put hold 为空操作：无跨请求共享时成立；若该路径也会异步 Put + 立刻滑窗回收，存在与 Prefix 路径不对齐的风险

### K. 需要人确认的点（待确认清单）

- [x] [BLOCK] 主类型是否 feature — 按提交 `feat(...)` 与会话「按规则补产物」落地为 feature
- [x] [BLOCK] 主子域是否 cache_manager — 按 `ai_docs/modules.md`（global_cache_connector 归 cache_manager）落地；交叉 scheduler / distributed
- [x] [BLOCK] SWA 选点公式是否以代码为准 — 以 `GlobalCachePutWorker::PutRequestSwaCache` 与 yaml 注释为准（尾部 N + 前段等分中位，首尾两段不传）
- [x] [?] TAPD 单 — [【[feat] DSv4 SWA 全局 cache（Mooncake Put/Get）方案设计与实现】](https://tapd.woa.com/tapd_fe/20421996/story/detail/1020421996137039494)（workspace `20421996` / story `1020421996137039494`）
- [ ] [?] `swa_tail_block_num` 结构体默认 4 vs yaml 默认 2，是否要改成一致
- [ ] [AUTO?] PD 分离 role=prefill/decode 的 first-token 存取不在本次主验收 — (N/A) → AI默认：沿用原全局 cache 角色语义，本提交不扩展
