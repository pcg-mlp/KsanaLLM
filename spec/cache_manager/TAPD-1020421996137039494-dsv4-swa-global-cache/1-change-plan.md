### A. 问题分析与方案总述【必填】

- 问题分析：提交前全局 cache 只覆盖主 KV（FA）连续前缀。DSv4 SWA 块未压缩（约 7.6×），全量上传会翻倍占用；Get 若只用 FA 连续命中，会把窗口外无 SWA 的前缀当成可复用。Put 异步与 engine `UpdateRequestTokens` 同步滑窗回收竞态：滑出块可能被别的请求拿走，静默写错全局 key。serving 侧 master 需手工拉起，端口易与 yaml 漂移。IPC long-task 出队并行度硬编码，与 `fetch_thread_num` 不对齐时 H2D stream 无法 1:1。
- 方案总体思路：以已合入 `6a1fe8d72`（TAPD 1020421996137039494）为施工结果。connector 增加 SWA 独立 key/staging 与 `AlignHitBoundaryBySwa`；PutWorker 按「尾部 N 密存 + 前段等分中位稀存」选点并快照 SWA 列表；PrefixCacheManager 用 `BeginSwaPutHold`/`EndSwaPutHold` 与 `DropUnbackedSwaBlocks` 守住生命周期；fetcher/tracker/IPC 贯通 SWA 字段与跨卡指标；python bootstrap 在 hook=1 且 RPC 空闲时拉起 master；`long_task_thread_num` 从 yaml 注入并与 fetch 取 max。不改 `pd_v2` 传输路径。

### B. 代码定位【必填】

- `csrc/global_cache_connector/global_cache_connector_types.h` — SWA 配置字段与默认值 — `swa_tail_block_num` / `swa_front_equal_parts` / `swa_block_byte_size`
- `csrc/global_cache_connector/mooncake_connector.cpp` / `mooncake_mock_connector.cpp` — Get：`AlignHitBoundaryBySwa` + `known_exist` 覆盖候选；Put：SWA 独立 staging — `QueryAndFetch` / `PutSwaBlocks`
- `csrc/executor/global_cache_put_worker.cpp` — 仅最后一段完整 prefill put；快照 tokens/SWA；选点公式 — `SubmitBatchPut` / `PutRequestSwaCache`
- `csrc/cache_manager/cache_manager_interface.h` 及 Prefix/Direct 实现 — hold 与丢未 backing SWA — `BeginSwaPutHold` / `EndSwaPutHold` / `DropUnbackedSwaBlocks`
- `csrc/batch_scheduler/global_cache_fetcher.cpp` / `global_cache_put_tracker.cpp` — 多卡聚合 SWA 对齐命中；FA/SWA 块数按卡累加 — `FinalizeSwa` / `OnDevicePutDone`
- `csrc/ipc/functions/global_cache_query.cpp` / `global_cache_put_done.cpp` — SWA context 序列化；put done 带 fa/swa 块数
- `csrc/ipc/core/ipc_func_proxy.cpp` — long-task 线程数由构造参数注入，替代硬编码
- `csrc/configure/schedule_config_parser.cpp` — yaml 解析 + `AlignLongTaskAndFetchThreadNum`
- `python/ksana_llm/global_cache_bootstrap.py` — `ensure_mooncake_master`
- `examples/ksana_llm.yaml` / `README.md` — 运行时启用步骤与注释块

### C. 调用链影响【必填】

- serving 启动 — 调用方：`python/serving_server.py` / `ksana_engine.py` → `ensure_mooncake_master`
- Query — `EventDrivenScheduleProcessor` → `GlobalCacheQueryHook` → IPC `GLOBAL_CACHE_QUERY`（long-task 线程）→ executor `QueryAndFetch` → fetcher 聚合 → 命中后 `DropUnbackedSwaBlocks`
- Put — executor Forward 后 `GlobalCachePutWorker::SubmitBatchPut` → `BeginSwaPutHold`（engine 侧登记）→ 异步 D2H/Put → IPC put done → `OnDevicePutDone` 累加 → 全卡完成 `EndSwaPutHold` + `DestroyFinishedRequest`
- 配置 — `ScheduleConfigParser::InitGlobalCacheConnectorConfig` / `InitIpcConfig` → supervisor 创建 `IpcFuncProxy(long_task_thread_num)`

### E. 测试要点【必填】

- T01 → `ScheduleConfigParserTest`：未配 `swa_tail_block_num` 时期望 2（yaml 默认）
- T02 → `LongTaskAndFetchThreadNumAlignToMax`
- T03 → `PutWorkerSwaTest::{SelectsTailDensePlusFrontEqualMids,TailCoversAllWhenBlockNumSmallerThanTail,SkipsPartialTailBlock}`
- T04 → `MockConnectorSwaTest::{SwaAlignedHitFetchesBoundaryWindow,MissingSwaTruncatesBoundary,NoSwaContentYieldsNoHit}`
- T05 → `MockConnectorTest.PartialBlockMissProducesNonContiguousRanges`
- T06 → `PutTrackerTest.FaAndSwaPutBlocksSumAcrossDevices`
- T07 → `PrefixCacheManagerSWATest.DropUnbackedSwaBlocks*` / Direct 同名
- T08 → `QueryCallableTest.Swa*`
- T09 → 手工：README 三条件启用路径（本会话未跑 E2E）
- 缺口：`PrefixCacheManager::{Begin,End}SwaPutHold` 无直接单测（仅 mock 断言被调用）— 不延期关闭，记入 review

### F. 灰度回滚

- 开关：`ENABLE_GLOBAL_CACHE_HOOK` 去掉或 `service_endpoint` 置空即关闭 connector 与 bootstrap
- 不改存储格式兼容；Mooncake 内已有 key 残留可接受，关闭后不再读写
- 代码级回退：revert 该提交（89 文件，优先用开关）

### G. 风险待办【必填】

- `GlobalCacheConnectorConfig.swa_tail_block_num` 结构体默认 4 vs parser 默认 2：serving 走 parser；单测直接构造 config 会用 4
- Prefix put hold 无单测：回归依赖 E2E/日志，错 hold 会导致错 key 污染
- Direct 路径 hold 为空：若 Direct + 异步 Put 同时滑窗回收，可能读已释放块

### Env 执行环境提示【必填】

- 本次改动能否在编码环境内做语法/类型级自检？能（已合入，单测可在 `dpv3_mn_test` 编跑；本会话未重跑）
- 是否涉及需要容器内才能验证的行为？是（真 Mooncake、RDMA、serving 拉起 master）
- 建议的验证范围（增量单测目标）：`global_cache_connector_test`、`global_cache_put_worker_test`、`global_cache_put_tracker_test`、`global_cache_fetcher_test`、`global_cache_query_callable_test`、`schedule_config_parser_test`、`prefix_cache_manager_test`、`direct_cache_manager_test`
- 是否预判需要集成测试：是；自动化程度：需人工触发（`scripts/e2e/mooncake_cache_stages/`，依赖本机 master）
