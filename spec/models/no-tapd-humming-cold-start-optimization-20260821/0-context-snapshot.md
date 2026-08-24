# 0-context-snapshot — humming 冷启动优化（搜索缓存 + JIT 进程模型）

## 0. Summary Card【必读】
- 检索是否命中：是；命中文档数：3
- 关键结论数：5；待确认缺口数：2
- 最近一次更新阶段：clarify

## A. 检索到的相关文档【clarify 起草，design 可追加】
- `ai_docs/code_wiki/8. 模型与算子框架.md` — 相关原因：记录 MoE 层工厂五元组与后端映射（FP4 W4A8 → `HummingFusedWfp4Afp8MoeLayer`）及 `csrc/layers/nvidia/` 目录职责 — 结论摘要：被改的 W4A8 MoE 层由 `MoeLayerFactory` 按五元组创建，`Preprocess` 属于该层生命周期，本次不改工厂与创建路径
- `ai_docs/code_wiki/8.2. 算子封装与硬件后端.md` — 相关原因：记录 humming 作为 FP4 MoE 计算后端与 kernel 多硬件后端结构 — 结论摘要：humming 是 FP4（W4A16/W4A8）专用后端，JIT 编译链路不在该文档覆盖范围内
- `ai_docs/code_wiki/12. 部署配置与使用.md` — 相关原因：提及 `humming_tool.py` 作为附加内核包部署依赖 — 结论摘要：humming 包与工具脚本是部署依赖的一部分，进程模型变更不影响部署形态

## B. 关键结论（供 go 编码遵守）【必填】
- 缓存层任何失败只降级为 miss / 不保存（warn + return false），绝不抛出或影响主流程；这是 `HummingMoeSearchCache` 的设计契约，后续维护不得破坏
- 全链路禁止引入 `std::filesystem`（libtorch 符号插入导致 `remove_all` SIGSEGV，见 `humming_jit_compiler.cpp` 相关注释）；目录/文件操作一律用 syscall（mkdir/unlink/rename）
- 缓存 key 必须包含候选 kernel 集（kernel_name 排序后哈希，顺序无关）+ gpu_name/sm_count；任何影响候选集或硬件身份的变更必须改变 key（或 bump `kSearchCacheFormatVersion`）
- 落盘必须原子写：C++ 侧 tmp 文件带 pid 后缀 + rename；Python 侧 `.tmp` + `os.replace`；并发读者要么看到旧完整文件要么看到新完整文件
- 子进程执行必须走 `process_utils::ExecuteCommand`（无 shell、stdout/stderr 分离、独立进程组 + 超时组杀），禁止回退到 `popen`

## C. 待确认缺口（ai_docs 未覆盖，凭经验假设的点）【必填】
- humming JIT 编译流程细节（Phase 1~4、flock 跨进程去重、cubin 加载注册）ai_docs 未覆盖 — 假设：以 `humming_jit_compiler.cpp` 代码注释与 plan 文档描述为准 — 影响：若流程理解偏差，design 阶段 change-plan 的调用链映射可能失真
- 同名不同 SM 数 GPU 已入 key，但同 gpu_name 同 sm_count 的不同显存/频率变体（如 H20 不同版本）未区分 — 假设：该粒度下搜索结果差异可忽略 — 影响：若假设错，极端情况可能出现跨变体缓存假命中（可被 `HUMMING_SEARCH_REGENERATE=1` 兜底）

## D. 供 review 核对的约束清单【必填】
- W4A16 层（`humming_fused_wfp4a16_moe_layer.cpp`）零逻辑改动（仅 include 路径迁移）
- humming JIT 链路无 `popen` 残留；无 `std::filesystem` 引入
- 缓存 JSON 校验链完整：format_version / cache_key / payload_sha256 / kernel 存在性 / size_ms 严格递增，任一不过即 miss
- env 开关行为：`HUMMING_SEARCH_CACHE=0` 不读不写；`HUMMING_SEARCH_REGENERATE=1` 跳过加载、覆盖保存
- 子进程超时取值：query-configs 300s / generate-batch 1800s / nvcc 1800s / rm 60s
- 冷启动验收：python(humming_tool) 进程峰值 ≤ 2 且 /bin/sh 包装 = 0；热缓存二次运行无 `batch generate`、出现 `search cache hit` ≥ 2
