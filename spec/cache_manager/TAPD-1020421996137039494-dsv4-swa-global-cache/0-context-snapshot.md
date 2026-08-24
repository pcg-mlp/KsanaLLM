## 0. Summary Card【必读】
- 检索是否命中：是；命中文档数：5（另有若干低相关未列）
- 关键结论数：4；待确认缺口数：2
- 最近一次更新阶段：design

## A. 检索到的相关文档【clarify 起草，design 可追加】
- `ai_docs/modules.md` — 相关原因：子域划分真源 — 结论摘要：`global_cache_connector` 归 `cache_manager`；IPC 归 `distributed`；`batch_scheduler` 归 `scheduler`
- `ai_docs/code_wiki/9.2. 全局缓存与LinUCB路由.md` — 相关原因：全局 KV 接口与 Mooncake 实现 — 结论摘要：Query/Fetch/Put/first-token 四类能力；key=`kv_{prefix_hash}_tp{tp_rank}_l{layer}_b{idx}`；真/mock 后端
- `ai_docs/code_wiki/9. PD分离与全局缓存.md` — 相关原因：PD 与全局 cache 分层 — 结论摘要：connector 是存储级，与 PD v2 传输级正交；本需求不改后者
- `ai_docs/code_wiki/6.2. 块分配与缓存布局.md` — 相关原因：SWA 块池与主 KV 解耦 — 结论摘要：DSv4 SWA 独立 layout；滑窗回收走 LFU；全局 Put 必须 hold 滑出块
- `ai_docs/code_wiki/6. KVCache缓存管理.md` — 相关原因：Prefix/Direct/SWA 管理器关系 — 结论摘要：Prefix 有跨请求共享，Direct 无；hold 语义只在 Prefix 非空操作 `[design 补充]`

## B. 关键结论（供 go 编码遵守）
- 全局 cache 走 `GlobalCacheConnectorInterface`，禁止把 SWA 块写进主 KV key 空间；SWA 用 `device_id` 分 key（各卡内容不同）
- Put 在 Forward 之后异步提交；提交瞬间必须快照 token 与 SWA block 列表，禁止读 live `InferRequest`
- Prefix 路径：`BeginSwaPutHold` → 异步 Put → 全卡 done/`Destroy` 时 `EndSwaPutHold`；漏 hold 会把他人 KV 写进本前缀 hash
- 运行时三条件同时满足才启用：`ENABLE_GLOBAL_CACHE_HOOK=1`、非空 `service_endpoint`、`scheduler_type=1`

## C. 待确认缺口（ai_docs 未覆盖）
- SWA 稀疏选点公式（尾部 N + 前段等分中位）— 假设：以 `GlobalCachePutWorker` 与 yaml 注释为准 — 影响：N/parts 默认不一致会让稀疏度与文档分叉
- serving 内拉起 `mooncake_master` — 假设：ai_docs 未写运行时步骤，以 README 新增节为准 — 影响：wiki 仍只讲「先编译 connector」会被当成未启用

## D. 供 review 核对的约束清单
- 不改 `csrc/pd_v2/` 生产传输路径
- SWA Put 只上传满块；选点与 yaml `swa_tail_block_num` / `swa_front_equal_parts` 一致
- mock 与真 connector 的 `known_exist`：非 SWA 可洞命中；SWA 先 `AlignHitBoundaryBySwa`
- FA/SWA put 指标按卡累加
- `long_task_thread_num` 与 `fetch_thread_num` 启动取 max
- DirectCacheManager 的 put hold 为空操作（无共享）；Prefix 必须真正 hold
