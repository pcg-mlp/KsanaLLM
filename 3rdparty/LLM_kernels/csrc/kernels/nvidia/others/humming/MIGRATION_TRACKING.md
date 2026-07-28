# Humming MoE Operator Migration Tracking

## 算子出处

| 字段 | 值 |
|------|------|
| 原始项目 | [inclusionAI/humming](https://github.com/inclusionAI/humming) |
| pip 包名 | `humming-kernels` |
| 迁移时版本 | `0.1.0` → `0.1.3` → **`0.1.5`** |
| 源码 commit | `d596b9f` (0.1.5), migrated from `a373046` (0.1.3) |
| 安装来源 | `pip install git+https://github.com/inclusionAI/humming.git@d596b9f` |
| 迁移日期 | 2026-05-16 (初始迁移) |
| 目标分支 | `integrate-humming` (迁移) → `opt-moe-7-humming-2` (性能优化) |
| License | 见原始仓库 LICENSE |

## 迁移范围

### .cuh Header 文件

52 个 `.cuh` header 从 `humming-kernels` 0.1.5 pip 包 (`d596b9f`) 的 `include/humming/` verbatim 拷贝到本地 `include/humming/`，支持项目内直接修改进行二次开发。

**Include 路径解析**: JIT 编译时 `nvcc -I` 按序解析: (1) `$HUMMING_INCLUDE_DIR` 环境变量（显式覆盖）；(2) 从 `__FILE__` 推导 `.../humming/include`（默认，哨兵检查 `humming/kernel/humming.cuh` 存在）。CMake 构建 (preprocess) 直接使用本地 `${CMAKE_CURRENT_SOURCE_DIR}/../include`。

> `$KSANA_HUMMING_BASE_DIR` 指向 `jit/` 目录（用于 `humming_tool.py` 路径解析），**不用于** include 路径解析——复用会导致 `.../jit/include` 错误路径。

修改本地 header 后需清空 JIT 缓存 (`rm -rf ~/.cache/KsanaLLM/humming_kernel_files/`) 重新触发编译。升级 pip 包后需重新同步本地副本（见 [更新操作指南](#更新操作指南)）。

### 自研代码

**算子目录内** (`3rdparty/LLM_kernels/csrc/kernels/nvidia/others/humming/`):

| 文件 | 职责 |
|------|------|
| `jit/humming_tool.py` | Python 工具: `generate` (渲染 .cu 源码) + `query-configs` (候选配置生成: heuristic 归一化 + GEMM2 注入 + variant 生成 + dedup + 校验 + cache_key) |
| `jit/humming_jit_compiler.h` / `.cpp` | JIT 编译器 (ELF 解析、nlohmann/json 消费 query-configs、条件编译、非致命编译失败处理) |
| `launcher/tma.h` / `tma.cu` | TMA 描述符构建 (KsanaLLM 版) |
| `launcher/launcher.h` / `launcher.cu` | Kernel 启动 (KsanaLLM 版，17 个参数) |
| `launcher/kernel_meta.h` | 内核元数据 `KernelMeta` (含 `min_shape_m`/`max_shape_m` 范围字段) + `FormatKernelMeta` 可读格式化辅助函数 (输出 `"8x128x256 S3 C1 [SK]"`，供 layer 日志与测试打印共用) |
| `ops/ops.h` / `ops.cu` | 公共 API + dispatch (`forced_meta`/`required_block_m` 参数、`get_kernel_block_m` 带 `out_meta`) |
| `ops/wrapper.h` / `wrapper.cu` | Workspace sizing |
| `preprocess/humming_preprocess.h` | 权重预处理接口 (`pack_weight_fp4`/`repack_weight_wgmma`/`process_mxfp4_w4a8_weight`/`process_scales`) |
| `preprocess/pack_weight.cu` / `repack_weight.cu` / `process_mxfp4_w4a8.cu` / `process_scales.cu` | 权重预处理实现 (实例化 humming 模板 kernel) |
| `humming_stubs.cpp` | **非 SM90 平台** stub 实现 (所有符号 no-op/零返回，运行时调用打 ERROR 安全返回) |

**算子调用层** (仓库根 `csrc/`):

| 文件 | 职责 |
|------|------|
| `csrc/layers/humming_fp4_moe_layer.h` | Humming FP4 MoE Layer 头文件 (workspace 布局、profiling 接口、18-event per-phase profiling、`GetJitKernelTable` getter、`SetDeferPerfFlush`/`FlushPerfStats` 延迟 flush 接口、`profile_m_thresh1_` 分级阈值成员、`InvokeAlignBlockSize`/`InvokeMergedAlignBlockSize` (EP/TP template branching)、`MakeDummyTopkIds` (ep_size param + EP -1 建模)) |
| `csrc/layers/nvidia/humming_fp4_moe_layer.cpp` | Layer 实现 (Forward/Preprocess、profile-based kernel 选择、GEMM2 block_m 解耦、merged align (移除 >=224 阈值，EP/TP 统一 SglMoe 路径)、fused silu+quant (EP 版跳过 -1 行)、W4A8 量化、分级 profile 迭代、延迟 perf flush、W4A16 quant 事件跳过、`FormatKernelMeta` 可读 best= 日志、`last_moe_block_size_1/2` 写入 perf_stats) |
| `csrc/layers/moe_perf_stats.h` | `MoePerfStats` (GEMM 耗时 + 子相位 + kernel meta 指针 + FLOP/TFLOPS + `last_moe_block_size_1/2` GEMM1/GEMM2 align block size，供测试 routing/TFLOPS 统计) |
| `csrc/utils/search_status.h` | `HummingMoeSearchStatus` 单例 (profiled schedule 缓存 + right-align + tile-waste-aware) |
| `csrc/layers/tests/fp4_moe_layer_test.cpp` + 5 headers (`_common`/`_config`/`_perf`/`_print`/`_ref`) | FP4 MoE 测试 (W4A16/W4A8 基准、精度验证、JIT kernel 表打印)。支持 `FP4_BACKENDS` env 选择性运行 backend；`FP4_PERF_CSV=1` 输出 CSV；expert routing 统计 (torch 复现 gating 算命中，EP 按 `[0, E_local)` 过滤 rank 0 本地专家)；有效/真实双轨 TFLOPS/带宽 (真实含 block-size padding，`ceil(hit/bs)*bs` 逐专家)；5 表格 (命中汇总/有效+真实 TFLOPS/有效+真实带宽，EP 不再 skip)；`FormatKernelMeta` 共用打印 selected kernel；deferred perf flush 接入 (`if constexpr` Humming 分支传 `FlushPerfStats` 回调，Marlin 不变)。原 `csrc/layers/fp4_moe_layer_test.cpp` 已拆分移入 `tests/` 子目录 |

> 另有 `3rdparty/LLM_kernels/csrc/kernels/nvidia/moe_utils/moe_utils.{cu,h}` 新增/修改 align 内核：`InvokeSglMoeAlignBlockSize`、`InvokeSglMoeAlignBlockSizeMerged`、`InvokeMoeAlignBlockSize` 均加 `UseExpertParallel` 模板参数 (`if constexpr` 跳过 `topk_ids==-1` 的非本地 expert 行，EP 路径专用)，`InvokeSglMoeAlignBlockSizeMerged` 支持 GEMM1/GEMM2 双 block_m 一次 align；`3rdparty/LLM_kernels/csrc/kernels/nvidia/moe/humming_triton_fp8_quant/generate_triton_silu_mul_quant.py` 为融合 SiluAndMul+FP8 量化 Triton kernel 生成器，含 TP 版 `_silu_and_mul_quant_kernel` 与 EP 版 `_silu_and_mul_quant_ep_kernel` (透传 topk_ids 跳过 -1 行)。两者路径根为 `3rdparty/LLM_kernels/csrc/`。

## 运行时依赖

### Python 包 (JIT 编译时) 与系统依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| `humming-kernels` | 0.1.5 (pin commit `d596b9f`) | 内核源码生成 (Jinja2) |
| `torch` | 2.7+ | humming 内部依赖 |
| `jinja2` / `numpy` / `pyelftools` | - | humming 内部 (模板/启发式/ELF) |
| `nvcc` | CUDA 12.x (SM90) | JIT 编译 .cu → .cubin |
| `libcuda.so` | Driver API | `cuModuleLoad`/`cuLaunchKernelEx` |
| `flock` | POSIX | 多进程缓存安全 |

> `humming-kernels` 仅在**服务启动时** (JIT 首次生成) 需要。cubin 缓存命中后不再调用 Python。

### 环境变量

| 变量 | 作用 | 默认值 |
|------|------|--------|
| `KSANA_HUMMING_KERNEL_PATH` | JIT cubin 缓存目录 | `~/.cache/KsanaLLM/humming_kernel_files/` |
| `HUMMING_INCLUDE_DIR` | 覆盖 .cuh include 路径 (JIT) | 从 `__FILE__` 推导 `.../humming/include` |
| `KSANA_HUMMING_BASE_DIR` | 覆盖 `humming_tool.py` 目录 (指向 `jit/`) | 从 `__FILE__` 推导 `.../humming/jit` |
| `HUMMING_REGENERATE` | `=1` 强制重新生成+编译所有 cubin | 未设置 (使用缓存) |
| `HUMMING_JIT_LOG` | `=1` 输出详细 JIT 日志 | 未设置 (静默) |
| `QUANT_PROFILE` | profiling 迭代次数，`0` 禁用。Layer `Init` 读一次存 `profile_iters_`，`warmup_iters_=max(1,profile_iters_/3)`，`compile_extend=(profile_iters_!=0)` 显式传入 `CompileKernels` (见 [候选配置生成 > 条件编译](#条件编译-compile_extend_configs))。`ProfileAllBuckets` 按 `rep_m` 分级使用 iter: `rep_m≤THRESH1` 用全量 `profile_iters_`，`rep_m>THRESH1` 用 `max(profile_iters_*2/3, warmup)` (见 [分级 profile 迭代](#分级-profile-迭代)) | `15` |
| `FP4_PROFILE_M_THRESH1` | 分级 profile 迭代的 `rep_m` 阈值 ( tier 0→1 边界)，Layer `Init` 读一次存 `profile_m_thresh1_` | `512` |
| `FP4_BACKENDS` | **测试专用** (仅 `fp4_moe_layer_test.cpp`)。逗号分隔的 backend 名 (大小写不敏感精确匹配): `marlin`/`humming_w4a16`(`w4a16`)/`humming_w4a8`(`w4a8`)。未设/空 → 运行全部 3 个 backend；未知名 → 警告并跳过；重复 → 去重保序 | 未设置 (全部) |
| `FP4_PERF_CSV` | **测试专用**。`=1` 时将 5 个性能表格 (专家命中汇总、有效/真实 TFLOPS、有效/真实带宽) 写入 `./fp4_moe_layers_test/` 目录下标准 CSV 文件 (复合 cell 拆分为独立列)，终端打印不受影响 | 未设置 (仅终端) |
| `DISABLE_FORCE_INSTALL_HUMMING` | CMake 跳过自动 `pip install humming + triton` | 未设置 (自动安装) |
| `HUMMING_DISABLE_PARALLEL_BUILD` | 禁止 humming 后台并行编译 launcher 扩展 | `"1"` (humming_tool.py 设置) |

## 架构差异与设计决策

| 方面 | 原生 humming | KsanaLLM | 设计原因 |
|------|-------------|----------|---------|
| 框架绑定 | PyTorch (pybind11, torch.Tensor) | 无框架依赖 (raw CUDA pointers) | 去除 PyTorch 运行时依赖 |
| Kernel 注册/编译 | Python `register_kernel()`，`__post_init__` 即触发编译 | C++ `HummingJitCompiler` 三步解耦 (见 [JIT 流水线](#jit-流水线)) | 独立控制编译时机/并行度/缓存策略 |
| ELF 解析 | Python `pyelftools` | C++ 内嵌 ELF 解析器 | cubin 缓存命中后纯 C++ 启动 |
| 候选配置生成 | Python heuristic configs 数组 | Python `query-configs` 子命令 (heuristic+variant+GEMM2 注入+dedup+cache_key)，C++ nlohmann/json 消费 | 与 heuristic 强耦合放 Python 易维护；C++ 去除重复逻辑 |
| Kernel 选择 | Python configs 数组 | C++ `SelectKernel` (heuristic) / `GetProfiledMeta` (profiling) | 支持启动时 profiling + right-align + tile-waste-aware |
| TMA 描述符 | PyTorch Tensor shape/stride | 手动 dims/strides；FP4 B/Bs 按 `make_tma_desc_b/bs` 重写 | 见 [TMA 描述符修复](#tma-描述符修复-launchercu) |
| 缓存目录 | `~/.humming/cache/` | `$KSANA_HUMMING_KERNEL_PATH` | 独立缓存隔离 |
| .cuh headers | pip 包内 include/ | 本地 include/ 副本 (0.1.5 d596b9f) | 支持项目内二次开发 |
| GEMM2 优化 | 无 | stream-K + block_n=256 + 多 block_m 注入，M 分档 | 见 [GEMM2 block_m 解耦](#gemm2-block_m-解耦) |
| 配置变体 | 仅 heuristic | heuristic + variant (stream_k/block_n/stages/block_k/num_ctas + 组合) | 扩展候选空间供 profiling 选择 |

**未修改项**: .cuh 模板 verbatim copy；17 个 kernel 参数位置一致；`extern "C" __constant__` 元数据从 cubin ELF 读取。nvcc 编译参数核心一致（从 `NVCCCompiler.get_flags()` 抄录），差异: 省略 `-fopenmp`/`-lgomp`，硬编码 `sm_90a`。

## JIT 编译器优化机制

`HummingJitCompiler` (`jit/humming_jit_compiler.cpp`) 是 KsanaLLM 自研代码，**非 humming 原生组件**。

### JIT 流水线

原始 humming 在 Python 侧编译/加载/启动耦合在一起。KsanaLLM 拆解为三步：

| 步骤 | 函数 | 说明 |
|------|------|------|
| Generate | `GenerateSource()` | 调 `humming_tool.py generate` (Jinja2) 渲染 .cu。monkey-patch 跳过 CUDA context，仅取 `kernel.code` |
| Compile | `CompileCubin()` | nvcc .cu → .cubin。`.tmp` + `rename` 原子发布 |
| Load | `LoadAndRegister()` | `cuModuleLoad`/`cuModuleGetFunction` 加载 cubin，读 `__constant__` 元数据，注册 kernel_table_ |

为加速首次启动（多 expert × 多 block_m × 多 variant 可能数十个 kernel），`CompileKernels` 采用四阶段：

| 阶段 | 线程 | 动作 |
|------|------|------|
| 1 | 主线程 | 分离 cached vs uncached |
| 2 | 主线程 | 快速路径 `LoadAndRegister` 已缓存 cubin |
| 3 | 多线程 | 并行 Generate+Compile (nvcc CPU 密集)。**不 Load**。编译失败非致命: catch + WARNING (`if verbose_`) + `--active_count` + return 跳过 |
| 4 | 主线程 | 统一 `LoadAndRegister` 新 cubin；跳过不存在的 (Phase 3 失败项) |

> **Phase 3/4 分离**: `cuModuleLoad` 要求调用线程持有 CUDA context，`std::thread` 子线程不继承。重活并行，轻活回主线程串行。

### 缓存安全与去重

| 机制 | 粒度 | 说明 |
|------|------|------|
| `flock(LOCK_EX)` | 进程级 | 防多进程同时编译同一 cache dir |
| `dir_mutexes_` (`unique_ptr<mutex>`, per-dir) | 线程级 | `flock` 不阻塞同进程兄弟线程，需额外 mutex 串行化 |
| double-check (`CacheExists`) | - | 加锁后再次检查避免重复编译 |
| `loaded_keys_` (by `cache_key`) | kernel 级 | 跳过已 `cuModuleLoad` 的 kernel |
| `compiled_shapes_` (by `(N,K,E,dt,gt)`) | shape 级 | 多 layer 共享同 shape 只查一次 heuristic |

**损坏 cubin 自愈**: `LoadAndRegister` 检测 `CUDA_ERROR_INVALID_IMAGE` (CUresult=201) → WARNING (`if verbose_`, `log_mutex_` 保护) + 删除缓存目录，下次运行自动重新生成。

**避免 `std::filesystem`**: `MkdirP`/`rm -rf` 手动实现——libtorch.so 符号插入致 `std::filesystem` 调用 SIGSEGV。

### Kernel 选择策略 (`SelectKernel`)

heuristic 兜底路径 (profiling 未启用/未命中时): (1) **范围匹配优先**: kernel 带 `[min_shape_m, max_shape_m]` 范围且 `size_m` 落入则直接返回；(2) **tile-waste 兜底**: 按 `(size_m + block_m - 1) / block_m` padding 浪费选最小。调用方可通过 `forced_meta`/`required_block_m` 绕过 (见 [Profile-based 选择](#profile-based-kernel-选择机制))。

### 候选配置生成 (Python)

全部候选生成逻辑在 `humming_tool.py query-configs` 子命令，C++ 经 nlohmann/json 消费。输出格式: `{"heuristic_configs":[...], "extend_configs":[...]}`，每项含 `cache_key` + `is_heuristic`。

| 职责 | Python 函数 | 说明 |
|------|-------------|------|
| heuristic 归一化 | `_normalize_config` | 展平 `block_shape`/`warp_shape`，补全 tma/mbarrier/wspec/f16acc 默认值 |
| override 应用 | `_apply_override` | 生成变体/注入的基础: 拷贝 base + 覆盖 stream_k/block_n/stages/block_k/ctas；**一致性修复**: `block_n` override 时 `warp_shape[1]=min(bn,base)`，`block_k` override 时 `warp_shape[2]=block_k`，防止 warp > block 的非法 config 被静默丢弃 |
| GEMM2 注入 | `_inject_gemm2_configs` | 见下表 |
| **block_k 整除兜底** | `_fix_block_k_divisibility` | 见下表，GEMM2 注入后 + variant 生成后各调用一次 |
| variant 生成 | `_generate_variants` | 见下表 |
| 去重 | `_config_identity` | 14 字段 tuple (block_m/n/k, warp, stream_k, stages, ctas, tma_a/b/c/bs, mbarrier, wspec, f16acc) |
| 校验 | `_validate_config` | monkey-patch 下构造 `HummingKernel`，异常则丢弃 |
| cache_key | `_make_cache_key` | 含全部最终字段的确定性字符串 |

`generate` 子命令接收完整配置 CLI 参数 (block_n/block_k/warp_shape/use_stream_k/num_stages/num_ctas_per_sm/use_tma_*/use_mbarrier/use_warp_spec/use_f16_accum)，不再查询 heuristic。`KernelConfig` 存最终值 + `cache_key` + `is_heuristic`，不再有 `force_*` override 字段。

**Fallback**: Python 侧 `get_configs()` 异常 → 输出 2 个 fallback config；C++ 侧 Python/JSON 失败 → 构造 block_m=64/128 的 `KernelConfig`。两层防御。

**Variant Config 生成** (`_generate_variants`): 每个 heuristic base config 生成变体。**`hard_cap` 截断已移除**——原 72 上限被小 block_m base 的变体填满，致大 block_m base 的 C2/C3 候选被静默丢弃；现仅靠 dedup (`_config_identity`) 控制候选数。变体 `min_shape_m`/`max_shape_m` 重置为 `[1, INT64_MAX]`（override 后 heuristic M 范围不再适用），由 profiling 选出赢家。

| 变体 | 覆盖字段 | 理由 |
|------|---------|------|
| stream_k | `use_stream_k=True` | 翻转 heuristic 小-K 关闭 |
| block_n=256 | `block_n=256` | 提升 large-N 吞吐 |
| block_n=128 (2b) | `block_n=128` | 从大 block_m base (heuristic bn=256) 生成 bn=128，倍增 N-parallelism (如 N=768: 3→6 N-blocks) |
| stages ∈ {2,3,4,5} | `num_stages` | 探索 pipeline 深度；S2 填/排仅 6.3% 但单 stage overlap (R4 新增) |
| block_k=64 (FP8, K>64) | `block_k=64` + `warp_shape[2]=64` | 放宽 K≤512 限制 |
| num_ctas ∈ {1,2,3,4} | `num_ctas_per_sm` | 小 M 减 tile 浪费、增寄存器 |
| stages4+bn256 / stages5+ctas2 (6a/6b) | 组合 | 交叉探索 |
| stages3+bk64 (6c) | `stages=3, block_k=64` (FP8) | small-K GEMM2 最优 pipeline |
| bk64+S4/S5 (R2) | `stages=4/5, block_k=64` (FP8) | 6c 仅覆盖 S3，补 S4/S5 深度探索 |
| bn256+bk128 (6d) | `block_n=256, block_k=128` (size_k≥128) | 生成 `64x256x128` tile；opt14 移除 `size_n>=4096` 守卫，GEMM1 也覆盖 (block_m=64 base) |
| stages3+bk128 (6e) | `stages=3, block_k=128` (FP8) | small-K pipeline 填充 |
| stages3+ctas2 (6f) / stages3+ctas3 (6g) | `stages=3, num_ctas=2/3` | S3 pipeline + C2/C3 occupancy，生成 S3 C2/C3 config |

**大 block_m 注入** (`_generate_variants` §4.4): heuristic `get_config1` 上限 `block_m=176`，大 token (16K/32K) profiling 选 block_m=112 (省 padding 但损 WGMMA 效率)。仅当 `max(heuristic block_m) >= 152` 时，从最大 base 注入 `block_m ∈ {200,224,256}` 候选 (必须 8 的倍数)，配 `block_n∈{256,128}`/`stages∈{3,4,5}`/`ctas∈{1,2}` 组合 (含 R5 C2 探测)；smem 可行性 (H200 cap 227KB) 由 `_validate_config` + JIT 编译兜底丢弃超预算项。

**缺失 block_m 补全** (`_generate_variants` §R1): heuristic 仅产生 `argmin(calc_num_block_list)` 的 block_m，留下间隙 (如 Pro_EP8 GEMM2 缺 block_m=16)。从最大 base 注入 `block_m ∈ {16,24,32,48,64,96,128}` (小于已注入最大值)，每个试 `(S3,C1)/(S3,C2)/(S4,C1)/(S5,C1)` 组合，覆盖特定 EP token 范围最优。

**GEMM2 注入** (`_inject_gemm2_configs`): 针对 `size_k<=512 && size_n>=4096` 的 small-K large-N 形状，heuristic 对小-中 M 不优。注入前移除 `max_shape_m<=3072` 的 heuristic config。**pipeline 填充修复**: 选最大 2 的幂 `block_k ∈ {256,128,64}` 使 `size_k/block_k >= 4` (S3 需 4 个 K-iteration 填充 pipeline)，原 heuristic `block_k=256` 致 K=512 仅 2 iteration、K=384 经整除兜底后 3 iteration，pipeline 饥饿使小 block_m kernel 人为偏慢:

| M 范围 (max_shape_m) | 注入 block_m | 配置 |
|---------------------|-------------|------|
| 1-384 | 8, 64 | `stream_k=True, block_n=256, stages=3, block_k=optimal_bk` |
| 385-1536 | 16, 32, 64 | 同上 |
| 1537-3072 | 48, 64 | 同上 |
| 3073+ | (保留 heuristic) | heuristic 已近优 |

### block_k 整除兜底 (`_fix_block_k_divisibility`)

**背景**: H200 设备名含 "H200"，`get_heuristics_class()` (`humming/tune/__init__.py:46`) 的 `if "H20" in name and "H200" not in name` 为 false，故走通用 `Sm90Heuristics` (非 `Sm90H20Heuristics`)。通用 `get_config1` 的 N>4096 分支 (sm90.py:55-61) **缺** `while meta.shape_k % block_shape_k != 0` 整除循环 (H20 版 sm90_h20.py:196-199 有此循环)。FP8 small-K large-N 形状 (如 Pro_TP8 GEMM2: K=384, N=7168) 会产生 `block_k=256` 但 `384 % 256 = 128 ≠ 0` 的不可编译 config，导致该 M 范围运行时无 kernel 可选 → `humming_moe_gemm` 断言 `No humming kernel with block_m=8`。

**兜底逻辑** (mirrors H20 while-loop):
- 若 `size_k % block_k == 0`: 原样返回。
- 否则: 反复 `block_k //= 2` 直至整除；若 `warp_shape[2]` 不再整除 `block_k` 则重设为 `512 // a_num_bits` (BF16=32, FP8=64)。
- 若 `block_k` 降至 `< min_warp_k` 仍无法整除: 返回 `None`，调用方丢弃此 config。

**调用点** (`_cmd_query_configs`):
1. **GEMM2 注入后** (step 2.5): 修正 heuristic configs，使后续 `_generate_variants` 继承正确 `block_k`。
2. **variant 生成后** (step 3.5): variant 的 `block_k_override=64` 可能与某些 K 不整除，同样修正。

**验证**: 修正后所有候选 `size_k % block_k == 0` 且 `block_k % warp_shape[2] == 0`；Pro_TP8 GEMM2 (K=384) FP8 下 `bm=8` 从 `bk=256` (无效) 改为 `bk=128` (有效)。

### 条件编译 (compile_extend_configs)

`CompileKernels` 接收显式 `bool compile_extend_configs`。`false` 时跳过所有 `is_heuristic==false` 的 config，节省非 profiling 场景首次编译时间。该参数由 MoE layer `Init` 时根据 `QUANT_PROFILE` 计算一次 (`compile_extend = (profile_iters_ != 0)`) 显式传入——**单一真相源**，避免 JIT 编译器与 profiler 读 env 不一致导致 missing kernel。

## Layer 调用层优化

`csrc/layers/nvidia/humming_fp4_moe_layer.cpp` 中对 humming 算子**调用侧**的优化。

### Forward 整体流程

```
hidden_states ──► TopKSoftplusSqrt gating ──► topk_ids, topk_weights
                                                       │
                                                       ▼
                    SelectGemmMeta(GEMM1) + SelectGemmMeta(GEMM2)  ← GEMM2 block_m 提前算
                                                       │
                                                       ▼
   ┌──── merged align (block_m 不同): 一次 launch 算 GEMM1+GEMM2 align (EP/TP template) ──┐
   │ 否则 (block_m 相同): InvokeAlignBlockSize(GEMM1)，GEMM2 复用 GEMM1 sorted/expert    │
   ▼                                                                                    ▼
    hidden_states ──► [W4A8: QuantizeToFp8] ──► humming_moe_gemm (GEMM1, w1)
                                                       │
                                                       ▼
     W4A16: SiluAndMul            W4A8: FusedSiluAndMul+Fp8Quant (一次 Triton kernel；EP 版 `_silu_and_mul_quant_ep_kernel` 跳过 topk_ids=-1 非本地行)
                                                       │
                                                       ▼
                                           humming_moe_gemm (GEMM2, w2, top_k=1)
                                                       │
                                                       ▼
                                           InvokeMoeFusedMulSum (×topk_weight, sum)
```

- **无 chunking**: humming 内部 stream-K 调度，一次性处理所有 token。
- **GEMM2 block_m 提前**: `SelectGemmMeta(GEMM2)` 在 align 之前完成，使能 merged align (见 [GEMM2 block_m 解耦](#gemm2-block_m-解耦))。
- **GEMM2 top_k=1**: `silu_output` 已展开为 `[tokens*topk, inter]`，top_k=1 使 kernel 直接用 `sorted_ids[i]` 作行索引。
- **locks 清零**: 每次 GEMM launch 前 `cudaMemsetAsync(ws_.kernel.locks, 0, ...)`，防残留锁值致 stream-K 死锁。

### 路由分支 (EP / DeepEP / EPLB)

| 模式 | 条件 | 行为 |
|------|------|------|
| **DeepEP** | `using_deepep_` | `InvokeExpertMapInverseInplace` 反映射 → Dispatch (all-to-all) → `ProcessTokens` → Combine |
| **EP (无 DeepEP)** | `global_expert_para_size_ > 1 && !using_deepep_` | `InvokeExpertMapInplace` 重映射 → `ProcessTokens` |
| **单节点** | 其他 | 直接 `ProcessTokens` |

EPLB: `enable_load_eplb_config_` 从 `eplb_map` 加载 per-layer expert 重映射 (gating 后执行)；`enable_dump_eplb_data_` dump topk_ids 为 .npy；`expert_map_` 当 `enable_load_eplb_config_ || !using_deepep_` 时创建 (纯 DeepEP 不创建)。

### Profile-based Kernel 选择机制

heuristic 基于通用 GPU 模型，对 H20/H200 等特定卡未必最优。本分支引入启动时 profiling。

**`HummingMoeSearchStatus`** (`csrc/utils/search_status.h`) 单例，按 `(N,K,E,compute_dtype,gemm_type)` 存储 `ProfiledSchedule { size_ms (升序), metas, block_ms }` (parallel arrays)。

**Profiling 流程** (`Preprocess` → `ProfileAllBuckets` → `ProfileSingleGemm`):
1. `Preprocess` 对 GEMM1 (N=inter*2, K=hidden, top_k=expert_topk) 和 GEMM2 (N=hidden, K=inter, top_k=1, size_m×=expert_topk) 分别 profiling。iters 由 `profile_iters_` (`QUANT_PROFILE`) 控制，`warmup_iters_=max(1,profile_iters_/3)`。
2. `ProfileAllBuckets`: 30 个 bucket `kBuckets = {1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096,6144,8192,10240,12288,14336,16384}` (中间值覆盖 decode batch size，中点减半 right-align gap，扩展 bucket 覆盖 8K-16K token 范围)。dummy 权重 (`PrepareDummyWeights`，随机 FP4+E8M0，每 shape 一次) 跨 bucket 复用。dummy topk_ids 用**随机均匀采样** (seed=42 `std::mt19937`)，匹配 benchmark 真实 expert 分布 (小 M 稀疏、大 M 收敛均匀)。**EP -1 建模**: `MakeDummyTopkIds` 的 `ep_size` 参数，EP (`ep_size>1`) 时每对以 `(ep_size-1)/ep_size` 概率置 -1 (非本地 expert)，模拟 Forward `InvokeExpertMapInplace` 重映射后的 per-rank topk_ids；配合 `InvokeAlignBlockSize(use_expert_parallel=true)` 使 profile 的 `num_tokens_post_pad` 匹配真实负载 (否则高估 ~ep_size×)。TP (`ep_size=1`) 路径 0% -1，rng 消耗不变，输出逐字节一致。
3. `ProfileSingleGemm` 对所有候选 kernel 逐个测时: M 范围过滤 (跳过 `max_shape_m < size_m`)；per distinct block_m 预跑 align；跳过 `actual_tiles < num_ctas` (防 stream-K 死锁)；launch 前清零 locks；`MeasureCudaMinExecutionTime` (`csrc/device/nvidia/cuda_utils.h`，warmup+iters，per-iteration event 取 **min** 替代 mean，`flush_l2=false`)；`at::cuda::setCurrentCUDAStream` 切计算流后恢复；try-catch 跳过 launch 失败候选 (smem 超限等)，catch 块 `cudaDeviceSynchronize()`+`cudaGetLastError()` 清除 sticky CUDA error。返回最快 kernel。

#### 分级 profile 迭代

大 `rep_m` bucket 单次 GEMM 耗时已达 ms 级，少量迭代即可稳定测时。阈值作用于 `rep_m = bucket × size_m_multiplier` (实际 GEMM M 维)，Layer `Init` 读 `FP4_PROFILE_M_THRESH1` (默认 512) 存 `profile_m_thresh1_`:

| Tier | 条件 | iter (env=15) |
|------|------|---------------|
| 0 (全量) | `rep_m ≤ THRESH1` | `profile_iters_` (15) |
| 1 (2/3) | `rep_m > THRESH1` | `max(profile_iters_*2/3, warmup)` (10) |

兜底 `iter ≥ warmup`；`warmup` 跨 bucket 固定。原 3-tier 的 `FP4_PROFILE_M_THRESH2`/`profile_m_thresh2_` 已移除 (大 M tier 由 1/3 提升至 2/3，配合 min 估计器保持条件数)。日志 `rep_m=... done best=8x128x256 S3 C1 [SK] iter=10` 打印当前 bucket iter 便于确认 tier 归属。

**运行时选择** (`SelectGemmMeta`):
1. **Profiled 优先**: `GetProfiledMeta` right-align 二分查找 (最小 profiled size_m ≥ 实际；若实际 > max profiled，返回最大条目)。**tile-waste-aware**: 若 right-align kernel padding 浪费 >25%，检查下一个更小 bucket，选浪费更小者 (需 `waste_left < waste_right - 5pp`)。
2. **Heuristic 兜底**: `get_kernel_block_m` (→ `SelectKernel`，范围匹配优先 + tile-waste 兜底) + `out_meta`。
3. **Block_m 兜底**: 返回 `required_block_m > 0`，由 `humming_moe_gemm` 内部线性搜索。

> **Right-align 原理**: 小 M 对 padding 敏感，用 ≥ 实际 M 的 profiled 点更保守 (耗时上界)；大 M 趋势一致可外推。tile-waste-aware 修正 right-align 系统性偏向大 block_m kernel 的偏差。

### GEMM2 block_m 解耦

GEMM1 (gate+up, 大 K 小 N) 和 GEMM2 (down, 小 K 大 N) 形状差异大，heuristic 最优 `block_m` 通常不同。GEMM2 的 `SelectGemmMeta` 在 align 之前独立完成，`forced_meta=sel2.meta`。

**align 路径** (`use_merged_align = (sel2.block_m != sel1.block_m)`，原 `&& expert_num_per_node_ >= 224` 阈值已移除——EP/TP 统一走 SglMoe 快速路径):

| 路径 | 条件 | 行为 |
|------|------|------|
| **Merged align** | block_m 不同 | `InvokeMergedAlignBlockSize` → `InvokeSglMoeAlignBlockSizeMerged<UseExpertParallel>` 一次 launch 算两个 block_m 的 align (共享 per-expert token 计数)；EP 模板 `if constexpr` 跳过 `topk_ids==-1` |
| **复用** | block_m 相同 | GEMM1 走 `InvokeAlignBlockSize` (SglMoe EP/TP 模板)；GEMM2 直接复用 GEMM1 的 sorted_ids/expert_ids |

> 原 "Deferred separate" (block_m 不同 && expert<224) 分支随 `>=224` 阈值移除已不可达——`use_merged_align` 恒等于 block_m 差异判断。代码中保留的 deferred `InvokeAlignBlockSize(GEMM2)` 作为防御性兜底 (当前死路径)。

`ws_.align_gemm2` 与 `ws_.align_gemm1` 独立缓冲，避免重对齐覆盖未用完数据。`InvokeFillInfo` 死代码已删除，merged align 统一 PDL launch。`InvokeAlignBlockSize` 签名精简: 移除 `top_k`/`expert_para_size`/`rank` 参数 (rank 原硬编码 0 致 EP4 多进程 profiling coredump，已改用 `rank_`)。Merged align + EP 模板显著降低 EP no_gemm 阶段 align/silu 耗时。

### Workspace 内存布局优化

所有 workspace 从单一连续缓冲 `workspace_buffer_` carve-up:

```
[gating.topk_weights] [gating.topk_ids]
[align_gemm1] [align_gemm2]
[cache.gemm1_output / gemm2_output (共享)] [cache.silu_output]
[kernel.locks] [kernel.tma_descriptors]
[fp8.activation] [fp8.scales]
← softplus_sqrt 尾部切片 (与 fp8 区域时间安全重叠) →
```

| 优化 | 机制 | 理由 |
|------|------|------|
| **cache1/cache3 共享** | `gemm1_output`/`gemm2_output` 同地址，取 `max(gemm1,gemm2)` | 生命周期不重叠 |
| **W4A8 融合 silu+quant** | fused kernel 直接写 `fp8.activation`/`scales`，不经 `cache.silu_output` | 省一次 HBM round-trip |
| **softplus_sqrt 尾部切片** | topk 子层 workspace 从 buffer 尾部切片 | gating 在 ProcessTokens 前执行，与 fp8 区域时间安全重叠 |
| **num_ctas_per_sm 扫描** | 硬编码下限 (FP8=3, BF16=2) + 扫描 kernel table 取 `std::max` | heuristic 可能产生 num_ctas_per_sm>3 |
| **block_m 扫描** (动态 sizing) | `kHummingMoeMaxBlockSize=8` 兜底 + 扫描 kernel table 取 `max(block_m)` | H200 通用 Sm90 heuristic 产生 block_m 最大 176 (H20 为 64)。原硬编码 64 致 align buffer (按 `numel + E*(block_m-1)` 算) 溢出 → 非法显存访问 → CUDA error coredump。`GetWorkspaceSizeT` 遍历 JIT kernel table (过滤 dtype/`INDEXED`) 取 `max(8, meta.block_m)` 动态定 align_gemm1+gemm2 大小，升级 heuristic 自动适应。与 num_ctas_per_sm 扫描同模式 |
| **locks 4x 过配** | `get_locks_size` 含 4x 余量 | num_ctas_per_sm 升级时安全 |

### W4A8 FP8 激活量化

- **GEMM1** (`QuantizeToFp8`): `TritonWrapper::InvokeHummingFp8Quant` 生成 per-token FP8 scale，量化 `[tokens, hidden]`。
- **GEMM2** (融合 SiluAndMul+Quant): `InvokeHummingFusedSiluAndMulFp8Quant` (TP) 读 `[M*topk, inter*2]` BF16，写 `[M*topk, inter]` FP8 + `[M*topk]` FP32 scale (per-token absmax/448.0)。省一次 kernel launch + HBM round-trip。**EP 版** `InvokeHummingFusedSiluAndMulFp8QuantEp` 透传 topk_ids 跳过 `-1` 非本地行 (EP4 75% 行为无效计算)，TP 保持原 kernel 不变 (零回归)。
- **per-expert global_scale**: 预处理时 `process_mxfp4_w4a8_weight` 计算 `global_scale = 2^(scale_min_new-128)`，从 `weight_global_scale` 字段获取。
- **W4A16**: 不量化，`a_scale`/`global_scale` 传 nullptr，`compute_dtype=BF16`，`InvokeSiluAndMul`。

### 运行时性能统计 (`MoePerfStats`)

`SetPerfStats(MoePerfStats*)` 启用后，每次 Forward 通过 **18 个 CUDA event** (`std::vector<cudaEvent_t>` + `enum class Evt`: Gate/Align1/Quant1/Gemm1/Silu/Align2/Quant2/Gemm2/MulSum 各 Start/End) 测量。事件惰性创建 (`EnsureProfilingEvents`)，析构统一 destroy，跨 Forward 复用。默认 `nullptr` → `do_profile==false`，零开销 (production 路径)。

累加子相位: `gating` / `align`(=Align1+Align2) / `quant`(=Quant1+Quant2) / `gemm1` / `silu` / `gemm2` / `mulsum` / `total` (ms)。另存 `last_gemm1_meta`/`last_gemm2_meta` (kernel 元数据指针) + `last_moe_block_size_1`/`last_moe_block_size_2` (GEMM1/GEMM2 align block size，Marlin 两值相同)。TFLOPS/带宽由**测试侧** `RoutingStats`/`ComputeCostModel` 计算 (基于 meta + block_size + routing 命中，见下)。静态 `GetJitKernelTable()` 返回 JIT `kernel_table_` 指针供测试打印。

**两种 flush 模式** (消除 per-Forward `cudaEventSynchronize` 阻塞 + 7× `ElapsedMs` 带来的恒定 `other` 开销，小 M 时占 `nogemm` 显著比例):
- **Legacy per-Forward** (默认): 每次 Forward 结束 sync `MulSumEnd` + 7× `ElapsedMs` 累加。
- **Deferred** (测试专用): `SetDeferPerfFlush(true)` → Forward 内 `if (defer_perf_flush_)` 跳过 sync+read；测试 benchmark 循环结束后单次 `FlushPerfStats(iterations)` sync `MulSumEnd`，读**最后一次迭代** markers × `iterations` 填充 `perf_stats` (维持"累加和"契约，测试侧 `/iterations` 除法不变)。`RunBenchmark` 用 `if constexpr (std::is_same_v<LayerT, HummingFp4MoeLayer>)` 设置 deferred 并传 `FlushPerfStats(kBenchIters)` 回调；Marlin 不变。语义: `total_ms` 仍全循环平均，子相位为"最后 1 次 × N"而非"N 次均值"。

**W4A16 quant 事件跳过**: W4A16 下 `QuantizeToFp8` 为 no-op，Quant1/Quant2 的 4 个 `RecordEvent` 用 `do_profile && is_fp8` 守卫跳过 (18→14 event)，`quant_time_ms` 保持 0 (legacy 与 deferred 路径均在 `!is_fp8` 时置 0)。W4A8 保持全部 18 event (quant 为真实工作)。

**测试侧 routing 统计** (`tests/fp4_moe_layer_test_perf.h`/`_print.h`): 测试通过 torch 复现 gating (`sqrt(softplus(gate))` + topk) 算专家命中，EP 按 `[0, E_local)` 过滤 rank 0 本地专家。**有效/真实双轨** FLOPS 与带宽: 真实值含 block-size padding (逐专家 `ceil(hit/bs)*bs`)，有效值仅算实际命中 token；W4A8 激活 byte 按 FP8 (1B) 非 BF16 (2B)。输出 **5 表格** (命中汇总/有效+真实 TFLOPS/有效+真实带宽，EP 不再 skip)；`FP4_PERF_CSV=1` 写 CSV 到 `./fp4_moe_layers_test/`。`PrintExpertRoutingSummary` 按 backend 逐个输出子表 (各 backend 选不同 moe block size，故 blk_sz1/2、real_pairs、waste% 各异)。

### TMA 描述符修复 (`launcher.cu`)

W4A8 路径下 `BuildTmaB`/`BuildTmaBs` 布局错误，重写以匹配 humming `make_tma_desc_b`/`make_tma_desc_bs`:

| 描述符 | 修复后 (正确) |
|--------|--------------|
| `BuildTmaB` | `pack_size_k=256/a_num_bits` (FP8→32)、repacked `[E*K/32, N/32, 128]` as UINT32、smem box `[block_k/32, block_n/32, 128]`、swizzle=0。新增 `a_num_bits`(8/16)/`b_num_bits`(4) 参数 |
| `BuildTmaBs` | `[E*K/gs, N/16, 16]` as UINT8、`gs=32` (E8M0 group quant 硬编码)、ceil-div num_groups |

权重 scale group size 由 `meta->block_k` 改为固定 `32`。W4A16 不受影响 (BF16 激活走 `a_num_bits=16`)。

## 升级风险点

humming 版本升级时最可能受影响的点:

| 风险项 | 位置 | 影响 | 应对 |
|--------|------|------|------|
| **17 个 kernel 参数** | `launcher.cu` arg packing | kernel 签名变化致 launch 参数错位 | 对比 humming `csrc/launcher/launcher.cpp` |
| **`__constant__` 符号名** | `LoadAndRegister` ELF | 符号改名致元数据读取失败 | 对比 humming Python 模板 `kernel/humming.py` 中 `extern "C" __constant__` 声明 |
| **`HummingKernel.__init__()` 参数** | `humming_tool.py` generate | 构造参数增减需同步 CLI + `GenerateSource` | 见 [更新操作指南](#更新操作指南) |
| **`get_heuristics_class()`/`get_configs()` 格式** | `humming_tool.py` query-configs | configs 结构变化影响归一化与 C++ JSON 解析 | 检查 `_normalize_config` + C++ `parse_config` |
| **`nlohmann/json` 依赖** | `humming_jit_compiler.cpp` | C++ 消费 query-configs 依赖此头 | 确认 `jit/CMakeLists.txt` 已 fetch |
| **TMA 描述符布局** | `launcher.cu` `BuildTmaB`/`BuildTmaBs` | repack 布局变化 (pack_size_k/group_size/swizzle) 使描述符错 | 对比 `make_tma_desc_b`/`make_tma_desc_bs` |
| **`compile_extend_configs` 参数** | `CompileKernels` 签名 | 调用方须显式传入 | layer 与编译器签名需一致 |
| **`kernel.code`/`kernel.kernel_expr`** | `humming_tool.py` generate | 输出格式变化影响源码生成和符号提取 | 验证生成的 .cu 可被 nvcc 编译 |
| **`humming.ops` 导入副作用** | `humming_tool.py` sys.modules stub | `__init__.py` 改变导入行为使 stub 失效 | 检查 `import humming` 是否触发 CUDA/编译 |
| **.cuh header 内容** | `include/humming/` | 本地修改与新版冲突 | rsync 后 `git diff` 重新应用 |
| **`NVCCCompiler.get_flags()`** | `CompileCubin` nvcc flags | 编译参数变化影响 cubin 生成 | 对比 humming 最新 flags |
| **`KernelRuntime.prepare`/`init_sm_version`** | `humming_tool.py` monkey-patch | 初始化流程改变使 patch 失效 | 检查 `HummingKernel.__post_init__` 调用链 |
| **`kHummingMoeMaxBlockSize` (=8 兜底 + JIT 扫描)** | `humming_fp4_moe_layer.cpp` | heuristic 引入 block_m > 64 时 align buffer 不足。已改为 `GetWorkspaceSizeT` 扫描 JIT kernel table 取 `max(block_m)` 动态定大小，常量降为 8 兜底 | 升级后检查 block_m 范围，确认 `GetWorkspaceSizeT` 扫描逻辑覆盖 |
| **`humming_stubs.cpp` 符号同步** | `humming_stubs.cpp` | 接口签名变化时 stub 需同步否则非 SM90 链接失败 | 升级后跑非 SM90 构建验证 |
| **H200 启发式调度** | `humming/tune/__init__.py` `get_heuristics_class()` | `if "H20" in name and "H200" not in name` 使 H200 走通用 `Sm90Heuristics` (非 `Sm90H20Heuristics`)，产生 block_m 最大 176 且 small-K large-N 分支缺 K 整除 while 循环 | 升级后验证 H200 下 `query-configs` 输出；见 [block_k 整除兜底](#block_k-整除兜底-_fix_block_k_divisibility) 与 [align buffer 动态 sizing](#workspace-内存布局优化) |
| **block_k 整除 size_k** | `humming_tool.py` `_fix_block_k_divisibility` | 通用 Sm90 heuristic 在 N>4096 分支不保证 `size_k % block_k == 0` (FP8 K=384 + bk=256)，不可编译 kernel 致运行时无 kernel 可选 | 兜底函数已修正/丢弃；升级后用 `query-configs` 检查所有候选 `K % block_k` |
| **`FP4_PROFILE_M_THRESH1` 阈值** | `humming_fp4_moe_layer.cpp` `Init` | 分级 profile iter 的 `rep_m` 阈值默认 512 (原 3-tier 的 `FP4_PROFILE_M_THRESH2`/`profile_m_thresh2_` 已移除，大 M tier 统一 2/3 iter)，H20 vs H200 (SM 数不同) 最优值可能不同 | 升级 heuristic 或迁移新卡时验证大 bucket 搜索质量，必要时调 env |
| **EP align/silu 模板与 EP kernel** | `moe_utils.{cu,h}`、`triton_wrapper.{h,cpp}`、`generate_triton_silu_mul_quant.py` | `InvokeSglMoeAlignBlockSize`/`Merged`/`InvokeMoeAlignBlockSize` 的 `UseExpertParallel` 模板参数 (`if constexpr` 跳 -1)；EP 版 `_silu_and_mul_quant_ep_kernel` 透传 topk_ids。签名/模板变化致 EP 路径退化为全量计算 | 升级 moe_utils 或 triton 生成器后验证 EP align/silu 仍跳过 -1 行 |
| **`FlushPerfStats` 延迟 flush 契约** | `humming_fp4_moe_layer.h`/`.cpp` | 测试侧 `defer_perf_flush_=true` 时 Forward 跳过 sync，须由调用方在循环后调 `FlushPerfStats(iterations)`；`perf_stats` 子相位为"最后一次迭代 × N"而非"N 次均值" | 升级 event 基础设施时保持 `FlushPerfStats` 与 `ForwardT` event 记录一致；production (`perf_stats_==nullptr`) 不受影响 |

## 更新操作指南

### 当 humming-kernels 版本升级时

```bash
# 1. 安装新版本
pip install humming-kernels==<new_version>

# 2. 同步本地 .cuh header 副本
HUMMING_PATH=$(python3 -c "import humming; import os; print(os.path.dirname(humming.__file__))")
rsync -a --delete "$HUMMING_PATH/include/humming/" \
  3rdparty/LLM_kernels/csrc/kernels/nvidia/others/humming/include/humming/

# 3. Review diff — 重新应用本地修改
git diff 3rdparty/LLM_kernels/csrc/kernels/nvidia/others/humming/include/

# 4. 重新编译 + 测试 (清空 JIT 缓存)
cd build && cmake .. && ninja -j64
rm -rf ~/.cache/KsanaLLM/humming_kernel_files/ && CUDA_VISIBLE_DEVICES=3 ./bin/fp4_moe_layers_test

# 5. 如果 humming Python API 有变化，同步修改 humming_tool.py:
#    - HummingKernel 构造参数 → generate CLI + GenerateSource
#    - get_configs() 格式 → _normalize_config / _make_cache_key / C++ parse_config
#    - kernel.code / kernel.kernel_expr 格式
#    - make_tma_desc_b / make_tma_desc_bs 布局 → BuildTmaB / BuildTmaBs
```

### humming_tool.py 中的 monkey-patch

`_monkey_patch_kernel_runtime()` 在 `generate` 与 `query-configs` 均调用 (后者在 `_validate_config` 构造 `HummingKernel` 时):
```python
KernelRuntime.prepare = lambda self: None  # 跳过实际编译
KernelRuntime.init_sm_version = lambda self: (  # 硬编码 SM 90a
    setattr(self, "sm_version", 90) or setattr(self, "sm_version_str", "90a"))
```

`humming_tool.py` 还通过预填充 `sys.modules["humming.ops"]` 为 stub，阻止 `humming/__init__.py` 中 `import humming.ops` 的副作用 (注册 9 个 torch custom op + 后台 JIT 编译，约 6s)。KsanaLLM 只需 `humming.tune`/`humming.layer`/`humming.kernel`/`humming.config`。

> **权威性声明**: 本文档与代码实现是唯一可靠真相源。不依赖 git 历史 (commit 可能被压缩/rebase)。版本与源码 pin (上表 `算子出处`) 为外部 humming-kernels 固定引用，与分支历史无关。各章节描述当前代码状态，升级/迁移时以代码 + 本文档为准。
