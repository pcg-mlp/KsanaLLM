---
spec_version: "1"
spec_type: change
id: "no-tapd-humming-cold-start-optimization-20260821"
status: draft
subdomain: models
cross_subdomains: []
complexity: XL
work_type: perf
---

> 复杂度声明：跨 KsanaLLM 主仓 + `3rdparty/LLM_kernels` 独立 git 仓库提交，核心影响模块 ≥6（搜索缓存 / MoE 层 / jit_cache / device_context / process_utils / jit_compiler / humming_tool.py），按「影响面模块数 + 协作范围」维度就高定为 XL。
>
> 本文档为**回溯补充**：代码已实现并提交（KsanaLLM 仓 `7b8bf9976`、LLM_kernels 仓 JIT 进程优化提交），澄清素材来自两份实现 plan（`docs/superpowers/plans/2026-08-20-humming-moe-search-cache.md`、`2026-08-21-humming-jit-process-optimization.md`）与实际提交 diff；与 plan 有出入之处一律以实际实现为准。

### A. 工作类型（work_type）【必填，先于目标】
`perf` — 性能优化（吞吐/时延/资源，默认不改对外语义）
（交叉动机：搜索缓存部分引入了文件级缓存这一新机制，属「为性能而加能力」，主类型仍定 perf，缓存机制细节写入 §B；对外推理语义与数值结果不变）

### B. 目标（goal）

现状问题（两个互相独立的问题）：

1. **MoE 算子搜索结果跨进程不复用**：`HummingFusedWfp4Afp8MoeLayer::Preprocess()` 每次进程启动都要对 GEMM1/GEMM2 做全量 bucket 搜索（`ProfileAllBuckets`），同形状问题在不同进程/重启后重复 profile，启动耗时长。
2. **JIT 冷启动进程风暴与挂死风险**：`humming_jit_compiler.cpp` 的 `ExecuteCommand` 用 `popen(cmd + " 2>&1")` 执行子进程——① 每个 python 外再包一层 `/bin/sh`，进程数翻倍（每 rank 2 次 `query-configs` python（`import torch` ~10s、~600MB RSS）+ ~91 个并发 `generate` python + 等量 sh 包装 + ~1500 次 `mkdir -p` sh）；② stdout/stderr 合流，`query-configs` 的 stdout JSON 解析可被 stderr warning 污染，失败后静默回退到 2 个默认 config；③ `pclose` 无超时，python 挂死则该线程永久持有 config 目录 flock，拖死全部 rank。

目标（两部分，均已完成实现）：

1. **搜索缓存（KsanaLLM 仓 `csrc/layers/nvidia/`）**：新增纯逻辑类 `HummingMoeSearchCache`（`humming_search_cache.{h,cpp}`），文件级跨进程缓存；缓存 key = 问题形状（n/k/e/dt/gt/top_k/size_m_multiplier/ep_size/using_deepep）+ GPU 硬件身份（gpu_name/sm_count，堵死同候选集不同 SM 数 GPU 的假命中）+ 候选 kernel 集（kernel_name 排序后哈希，顺序无关）；`Preprocess()` 改为三级查找：内存单例 → 文件缓存 → 真实搜索（搜完写文件）。JSON 带完整校验链（format_version / cache_key / payload_sha256 / kernel 存在性 / size_ms 严格递增），原子写（tmp + pid 后缀 + rename）。配套收敛重构：候选过滤统一 `GetHummingKernelCandidates`，缓存根目录解析统一 `GetKernelCacheRootDir`，设备信息由 device context ext 缓存提供。
2. **JIT 进程模型优化（LLM_kernels 仓 `csrc/kernels/nvidia/others/humming/jit/`）**：新增 `process_utils.{h,cpp}`（`posix_spawnp` 直执行无 sh 中间层、stdout/stderr 分离捕获（poll 并发排空防死锁）、子进程独立进程组 + 超时组杀、`mkdir(2)` 递归建目录）；`humming_jit_compiler` 5 个子进程调用点全部迁移（超时取值 query-configs 300s / generate-batch 1800s / nvcc 1800s / rm 60s）；`humming_tool.py` 新增 `generate-batch` 子命令（单进程批量渲染全部 .cu，torch 只 import 一次；tmp + `os.replace` 原子落盘 + pid 后缀防滚动升级撕裂，逐项失败隔离）；`CompileKernels` 拆分 Phase 3a（主线程批量渲染，无 flock，靠原子 rename 保证多 rank 并发安全）/ 3b（每 config 一线程 nvcc，保留 flock 跨进程去重）。

基线与目标指标：

| 指标 | 改动前 | 改动后 |
|---|---|---|
| 同环境二次启动 MoE 搜索 | 每次全量 profile | 文件缓存命中（"search cache hit" ≥ 2，GEMM1/GEMM2 各一） |
| 冷启动单 rank python 进程峰值 | 2 + ~91 | 2 + 1（generate-batch） |
| python 外的 /bin/sh 包装进程 | 等量 | 0 |
| MkdirP 开销 | ~1500 个 sh | 0（syscall） |
| 挂死 python 持有 flock | 永久 | 上限超时值（组杀） |
| query-configs JSON 解析 | 可被 stderr 污染 → 静默回退 2 config | stdout-only 解析，污染免疫 |

实现后实测（服务启动端到端耗时，四卡 DeepSeek V4 Flash）：

| 场景 | 优化前 | 优化后 |
|---|---|---|
| 冷启动 | 520-560s | 350-360s |
| 热启动 | 370-400s | 120-140s |

（热启动指 JIT 编译缓存与 MoE 搜索文件缓存均命中的二次启动；冷启动收益主要来自 JIT 进程模型优化，热启动收益来自搜索缓存命中 + JIT 进程优化叠加。）

范围边界：

- 不改 W4A16 层（`humming_fused_wfp4a16_moe_layer.cpp` 零逻辑改动，仅 include 路径迁移）；不改 GEMM1/GEMM2 串行结构；不加编译线程池上限；不做 query-configs 结果落盘缓存
- 缓存层任何失败只降级为 miss / 不保存，绝不抛出影响主流程；环境变量开关 `HUMMING_SEARCH_CACHE=0`（禁读写）/ `HUMMING_SEARCH_REGENERATE=1`（跳读强写）
- 不引入 `std::filesystem`（libtorch 符号插入导致 `remove_all` SIGSEGV，一律用 syscall）
- JIT 侧改动在 `3rdparty/LLM_kernels` 独立 git 仓库内提交，不在 KsanaLLM 主仓

**消除 humming 冷启动的重复算子搜索与 python 进程风暴：二次启动搜索结果直接命中文件缓存，冷启动 .cu 渲染收敛为单 python 进程，子进程全部有超时兜底，任何缓存/子进程失败只降级不阻断主流程。**
