# Humming MoE Operator Migration Tracking

## 算子出处

| 字段 | 值 |
|------|------|
| 原始项目 | [inclusionAI/humming](https://github.com/inclusionAI/humming) |
| pip 包名 | `humming-kernels` |
| 迁移时版本 | `0.1.0` → `0.1.3` → **`0.1.5`** |
| 源码 commit | `d596b9f` (0.1.5), migrated from `a373046` (0.1.3) |
| 安装来源 | `pip install git+https://github.com/inclusionAI/humming.git@d596b9f` |
| 迁移日期 | 2026-05-16 |
| 目标分支 | `integrate-humming` |
| License | 见原始仓库 LICENSE |

## 迁移范围

### .cuh Header 文件

**本地拷贝**。52 个 `.cuh` header 文件从 `humming-kernels` 0.1.5 pip 包 (`d596b9f`) 的 `include/humming/` 目录复制到本地 `include/humming/`，支持在项目内直接修改以进行二次开发。

**Source**: `humming-kernels` 0.1.5 pip package `include/humming/` directory (copied verbatim to `include/humming/`).

**Include path resolution**:
- **JIT 编译**: `nvcc -I` 指向本地 include 路径。`HummingJitCompiler` 构造函数按以下顺序解析:
  1. `$HUMMING_INCLUDE_DIR` 环境变量（显式覆盖）
  2. 从 `__FILE__` 推导 base_dir (默认): `.../humming/jit/humming_jit_compiler.cpp` → `.../humming/include`
  - 哨兵检查: 验证 `humming/kernel/humming.cuh` 存在，缺失时报错。
  - 注意: `$KSANA_HUMMING_BASE_DIR` 用于 `humming_tool.py` 路径解析 (指向 `jit/` 目录)，不用于 include 路径解析。如需覆盖 include 路径，请使用 `$HUMMING_INCLUDE_DIR`。
- **CMake 构建** (preprocess): 直接使用本地 `${CMAKE_CURRENT_SOURCE_DIR}/../include`，不再查询 pip 包路径。

Header 内容未修改，与 pip 包 0.1.5 版本完全一致。修改本地 header 后需清空 JIT 缓存 (`rm -rf ~/.cache/KsanaLLM/humming_kernel_files/`) 重新触发编译。升级 pip 包后需重新同步本地副本（见下方"更新操作指南"）。

### KsanaLLM 自研代码（算子目录内）

以下代码为迁移过程中新写的，**非拷贝**:

| 文件 | 职责 |
|------|------|
| `jit/humming_tool.py` | 统一 Python 工具 (generate + query-configs 子命令，调用 humming API)，支持 `--force_stream_k`/`--force_block_n`/`--force_stages` 覆盖 |
| `jit/humming_jit_compiler.h` | JIT 编译器头文件 |
| `jit/humming_jit_compiler.cpp` | JIT 编译器实现 (含 ELF 解析、variant config 生成、GEMM2 优化配置注入) |
| `launcher/tma.h` / `tma.cu` | TMA 描述符构建 (KsanaLLM 版) |
| `launcher/launcher.h` / `launcher.cu` | Kernel 启动 (KsanaLLM 版) |
| `launcher/kernel_meta.h` | 内核元数据结构 (`KernelMeta`，含 `min_shape_m`/`max_shape_m` 范围字段) |
| `ops/ops.h` / `ops.cu` | 公共 API + dispatch (含 `forced_meta`/`required_block_m` 参数、`get_kernel_block_m` 带 `out_meta` 输出) |
| `ops/wrapper.h` / `wrapper.cu` | Workspace sizing |
| `preprocess/humming_preprocess.h` | 权重预处理接口 (stub) |

### KsanaLLM 自研代码（算子调用层 / 算子目录外）

以下代码不在 `humming/` 算子目录内，但属于 humming 算子迁移与优化的一部分（算子调用侧）。

| 文件 | 职责 |
|------|------|
| `csrc/layers/humming_fp4_moe_layer.h` | Humming FP4 MoE Layer 头文件 (workspace 布局结构、profiling 接口、GEMM meta 选择) |
| `csrc/layers/nvidia/humming_fp4_moe_layer.cpp` | Humming FP4 MoE Layer 实现 (Forward/Preprocess、profile-based kernel 选择、GEMM2 block_m 解耦、workspace 内存复用、W4A8 FP8 量化) |
| `csrc/utils/search_status.h` | `HummingMoeSearchStatus` 单例 (profiled kernel schedule 缓存 + right-align 二分查找) |
| `csrc/layers/fp4_moe_layer_test.cpp` | FP4 MoE 测试 (含 humming W4A16/W4A8 性能基准、精度验证、`QUANT_PROFILE` 开关) |

> 上述 layer 调用层的优化机制详见下文 [Layer 调用层优化](#layer-调用层优化) 章节。

## 运行时依赖

### Python 包依赖 (JIT 编译时)

| 包名 | 最低版本 | 用途 |
|------|----------|------|
| `humming-kernels` | 0.1.0 | 内核源码生成 (Jinja2 模板渲染) |
| `torch` | 2.7+ | humming 内部依赖 |
| `jinja2` | - | 模板引擎 (humming 内部) |
| `numpy` | - | humming 启发式计算 |
| `pyelftools` | - | humming 内部 ELF 解析 |

**注意**: `humming-kernels` 包仅在 **服务启动时** (JIT kernel 首次生成) 需要。一旦 cubin 缓存命中，后续启动不再调用 Python。

### 系统依赖

| 工具 | 版本 | 用途 |
|------|------|------|
| `nvcc` | CUDA 12.x (与 SM90 兼容) | JIT 编译 .cu → .cubin |
| `libcuda.so` | CUDA Driver API | cuModuleLoad, cuLaunchKernelEx |
| `flock` | POSIX | 多进程缓存安全 |

### 环境变量

| 变量 | 作用 | 默认值 |
|------|------|--------|
| `KSANA_HUMMING_KERNEL_PATH` | JIT cubin 缓存目录 | `~/.cache/KsanaLLM/humming_kernel_files/` |
| `HUMMING_INCLUDE_DIR` | 显式覆盖 .cuh header include 路径 (JIT 编译) | 从 `__FILE__` 推导 `.../humming/include` |
| `KSANA_HUMMING_BASE_DIR` | 覆盖 `humming_tool.py` 所在目录 (指向 `jit/`) | 从 `__FILE__` 推导 `.../humming/jit` |
| `HUMMING_REGENERATE` | `=1` 强制重新生成+编译所有 cubin (忽略缓存) | 未设置 (使用缓存) |
| `HUMMING_JIT_LOG` | `=1` 输出详细 JIT 编译日志到 stdout | 未设置 (静默) |
| `QUANT_PROFILE` | 启动时 profiling 迭代次数，`0` 禁用 profiling | `15` (humming layer) |
| `DISABLE_FORCE_INSTALL_HUMMING` | CMake 构建时跳过自动 `pip install humming + triton` | 未设置 (自动安装) |
| `HUMMING_DISABLE_PARALLEL_BUILD` | 禁止 humming 后台并行编译 launcher 扩展 (由 humming_tool.py 设置) | `"1"` ( humming_tool.py 内 default) |

> **注意**: `KSANA_HUMMING_BASE_DIR` 与 `HUMMING_INCLUDE_DIR` 语义不同，不可混用。前者指向 `jit/` 目录 (用于 Python 工具)，后者指向 `include/` 目录 (用于 nvcc 编译)。详见 [关键设计决策](#设计决策-rationale)。

## 与原生 humming 的差异

| 方面 | 原生 humming (pip 包) | KsanaLLM 迁移版 |
|------|----------------------|-----------------|
| 框架绑定 | PyTorch (pybind11, torch.Tensor) | 无框架依赖 (raw CUDA pointers) |
| Kernel 注册 | Python 侧 `register_kernel()` | C++ `HummingJitCompiler::LoadAndRegister()` |
| TMA 描述符 | 使用 PyTorch Tensor shape/stride | 手动传入 dims/strides |
| ELF 解析 | Python `pyelftools` | C++ 内嵌 ELF 解析器 (自研) |
| Launch API | `torch.ops.humming.launch_kernel` | `llm_kernels::nvidia::humming::launch_kernel()` |
| 编译调度 | humming JIT runtime (自动) | `HummingJitCompiler::CompileKernels()` (显式) |
| 缓存目录 | `~/.cache/humming/` | `$KSANA_HUMMING_KERNEL_PATH` (默认 `~/.cache/KsanaLLM/humming_kernel_files/`) |
| Kernel 选择 | Python 侧 configs 数组 | C++ `SelectKernel()` 最小化 M-tile 浪费 + `forced_meta` 支持 |
| 配置变体 | 仅 heuristic configs | heuristic + variant configs (stream_k/block_n/stages 覆盖) |
| GEMM2 优化 | 无 | 小-K 大-N 形状注入 stream-K + block_n=256 配置 |
| .cuh headers | pip 包内 include/ | 本地 include/ 副本 (从 0.1.5 pip 包 d596b9f 拷贝) |

### 未修改项

- **Kernel 源码** (.cuh 模板): 完全一致，verbatim copy
- **nvcc 编译参数**: 完全一致 (flags 从 humming 的 `NVCCCompiler.get_flags()` 抄录)
- **Kernel arguments**: 17 个参数位置完全一致 (从 humming 原始 csrc/launcher/launcher.cpp 验证)
- **extern "C" __constant__ 元数据**: 同样从 cubin ELF 中读取

## JIT 编译器优化机制

本章节记录 `HummingJitCompiler` (`jit/humming_jit_compiler.cpp`) 在迁移与优化过程中的关键设计。该编译器是 KsanaLLM 自研代码，**非 humming 原生组件**。

### 三步解耦 JIT 流水线

原始 humming 在 Python 侧通过 `torch.ops.humming.launch_kernel` 注册并调用 kernel，编译/加载/启动耦合在一起。KsanaLLM 将其拆解为三个独立步骤，全部由 C++ 驱动：

| 步骤 | 实现函数 | 说明 |
|------|---------|------|
| 1. Generate | `GenerateSource()` | 调用 `humming_tool.py generate` (Python/Jinja2) 渲染 .cu 源码。通过 monkey-patch 跳过 CUDA context 初始化，仅取 `kernel.code` 字符串 |
| 2. Cache (Compile) | `CompileCubin()` | 调用 nvcc 将 .cu → .cubin。编译参数从 humming `NVCCCompiler.get_flags()` 抄录，写入临时文件后原子 rename 防止并发读到半成品 |
| 3. Use (Load) | `LoadAndRegister()` | 通过 CUDA Driver API (`cuModuleLoad`/`cuModuleGetFunction`) 加载 cubin，从 ELF 读取 `__constant__` 元数据，注册到 kernel_table_ |

**为何解耦**: humming 原生流程在 `HummingKernel.__post_init__()` 中即触发编译，且依赖 PyTorch tensor 接口。解耦后 C++ 侧可独立控制编译时机、并行度、缓存策略，并去除 PyTorch 运行时依赖。

### 四阶段并行编译 (`CompileKernels(vector)`)

为加速首次启动（多 expert × 多 block_m × 多 variant 可能产生数十个 kernel），编译采用四阶段流水线：

| 阶段 | 线程 | 动作 |
|------|------|------|
| Phase 1 | 主线程 | 分离已缓存 (cached) vs 待编译 (uncached) 配置 |
| Phase 2 | 主线程 | 快速路径：直接 `LoadAndRegister` 已缓存的 cubin（跳过 Python/nvcc） |
| Phase 3 | 多线程 | 并行执行 Generate + Compile (nvcc 是 CPU 密集型，可良好并行)。**不在此阶段 Load** |
| Phase 4 | 主线程 | 统一 `LoadAndRegister` 所有新编译的 cubin |

**Phase 3/4 分离的关键原因**: `cuModuleLoad` 要求调用线程持有当前 CUDA context，而 `std::thread` 创建的子线程**不继承**主线程的 CUDA context。若在子线程中 Load 会失败。因此重活 (generate+nvcc) 并行，轻活 (cuModuleLoad) 回到主线程串行。

### 多进程 + 线程级安全

缓存安全分两层：

| 机制 | 粒度 | 作用 |
|------|------|------|
| `flock(LOCK_EX)` | 进程级 | 防止多进程同时编译同一 cache dir |
| `dir_mutexes_` (per-dir `std::mutex`) | 线程级 | `flock` **不阻塞同进程内的兄弟线程**，需额外 mutex 串行化同进程线程 |

加锁后执行 double-check（`CacheExists`），避免重复编译。cubin 写入用 `.tmp` + `rename` 原子发布。

### 损坏 cubin 自愈

`LoadAndRegister` 检测 `cuModuleLoad` 返回 `CUDA_ERROR_INVALID_IMAGE` (CUresult=201)，通常由早期并行编译的竞态条件导致 cubin 损坏。处理方式：打印 WARNING 并删除该 config 的缓存目录，下次运行自动重新生成。这是早期开发阶段遗留的防御机制。

### 编译去重

| 去重集合 | 键 | 作用 |
|---------|-----|------|
| `loaded_keys_` | `KernelConfig::CacheKey()` | 跳过已 `cuModuleLoad` 的 kernel，避免重复加载同一 cubin |
| `compiled_shapes_` | `(N, K, E, dt, gt)` 元组 | 跳过已查询 heuristic 的 shape，多 layer 共享同一 shape 时只查一次 |

因 MoE 模型有多层 HummingFp4MoeLayer，每层 Init 都会调用 `CompileKernels(shape)`，去重避免重复工作。

### Kernel 选择策略 (`SelectKernel`)

`SelectKernel(size_m, N, K, gemm_type, compute_dtype)` 从 `kernel_table_` 中选最优 kernel：

1. **范围匹配优先**: 若 kernel 带有 heuristic 的 `[min_shape_m, max_shape_m]` 范围，且 `size_m` 落入范围，直接返回（heuristic 认定最优）。
2. **tile-waste 兜底**: 对无范围信息的 kernel，按 `(size_m + block_m - 1) / block_m` 计算 padding 浪费，选浪费最小的。

调用方可通过 `forced_meta`（profiled 选择）或 `required_block_m`（强制 block_m 搜索）绕过此策略，见 [Layer 调用层优化](#kernel-选择路径)。

### Variant Config 生成

`QueryKernelConfigs` 在 heuristic 返回的基础配置上，为每个 config 生成至多 3 个变体以扩展候选空间（供 profile-based 选择）：

| 变体 | 覆盖字段 | 理由 |
|------|---------|------|
| stream_k variant | `force_stream_k=true` | heuristic 对小-K 可能关闭 stream-K，翻转 false→true 探索 |
| block_n variant | `force_block_n=256` | heuristic 默认 128，256 可提升大-N 吞吐 |
| stages variant | `force_stages=3` | 尝试 stages=3 作为 heuristic 默认的替代 |

变体经去重 (按 block_m/force_stream_k/force_block_n/force_stages) 后加入候选，硬上限 25 个。`KernelConfig::CacheKey()` 包含 force 覆盖字段，确保不同变体缓存到不同目录。

### GEMM2 优化配置注入

针对 GEMM2 (down projection) 的 small-K large-N 形状 (`size_k <= 512 && size_n >= 4096`)，heuristic 配置对小-中 M 不优（关闭 stream-K、block_n=128）。`QueryKernelConfigs` 中注入优化配置：

- **小-中 M (1-3072)**: 移除 heuristic 配置，注入两个 `stream_k=true, block_n=256, stages=3` 的 block_m=64 配置（M 范围 1-384 和 385-3072）
- **大 M (3073+)**: 保留 heuristic 配置（已近优）

效果见下方 [GEMM2 block_m 解耦 > 效果](#效果) 及文件变更记录 2026-06-19 条目。

## Layer 调用层优化

本章节记录 `csrc/layers/nvidia/humming_fp4_moe_layer.cpp` 中对 humming 算子**调用侧**的优化。

### Forward 整体流程

```
hidden_states ──► TopKSoftplusSqrt gating ──► topk_ids, topk_weights
                                                      │
                                                      ▼
                                              InvokeAlignBlockSize (GEMM1 block_m)
                                                      │
                                                      ▼
   hidden_states ──► [W4A8: QuantizeToFp8] ──► humming_moe_gemm (GEMM1, w1)
                                                      │
                                                      ▼
                                              SiluAndMul (gate*up → inter)
                                                      │
                                                      ▼
   silu_output ──► [W4A8: QuantizeToFp8] ──► humming_moe_gemm (GEMM2, w2)
                                                      │              ▲
                                                      │     GEMM2 block_m 解耦:
                                                      │     独立 SelectGemmMeta + 重跑 align
                                                      ▼
                                          InvokeMoeFusedMulSum (×topk_weight, sum)
                                                      │
                                                      ▼
                                                    output
```

关键设计点：
- **无 chunking**: humming 内部使用 stream-K 调度，一次性处理所有 token，不需要像 cutlass 那样按 batch 分块。
- **GEMM2 top_k=1**: GEMM2 的输入 `silu_output` 已展开为 `[tokens*topk, inter]`，每行是一个 token-expert pair。传 `top_k=1` 使 kernel 直接用 `sorted_ids[i]` 作行索引（不除以 topk），避免只读 1/topk 的行。
- **locks 清零**: 每次 GEMM launch 前执行 `cudaMemsetAsync(ws_.kernel.locks, 0, ...)`，防止上次运行残留的锁值导致 stream-K 死锁。

### 路由分支 (EP / DeepEP / EPLB)

Forward 中 `ProcessTokens` 之前根据部署模式有三条路由分支：

| 模式 | 条件 | 行为 |
|------|------|------|
| **DeepEP** | `using_deepep_` (多节点 EP + DP) | `expert_map_->InvokeExpertMapInverseInplace` 反映射 topk_ids → Dispatch (all-to-all) → `ProcessTokens(dispatch_tokens)` → Combine |
| **EP (无 DeepEP)** | `global_expert_para_size_ > 1 && !using_deepep_` | `expert_map_->InvokeExpertMapInplace` 重映射 topk_ids → `ProcessTokens(local_tokens)` |
| **单节点** | 其他 | 直接 `ProcessTokens(local_tokens)` |

EPLB (Expert-Level Load Balancing) 支持：
- **enable_load_eplb_config_**: 从 `eplb_map` 加载 per-layer expert 重映射，在 gating 后执行 `InvokeExpertMapInplace`
- **enable_dump_eplb_data_**: 将 topk_ids dump 为 .npy 文件用于离线 EPLB 分析
- **expert_map_**: `llm_kernels::nvidia::moe::ExpertMap`，DeepEP 或 EP 模式下创建

### <a id="kernel-选择路径"></a>Profile-based Kernel 选择机制

heuristic 配置基于通用 GPU 模型，对 H20 等特定卡未必最优。本分支引入启动时 profiling，为每个 GEMM shape 的每个 token 桶选择实测最快 kernel。

#### 数据结构

`HummingMoeSearchStatus` (`csrc/utils/search_status.h`) 是单例，缓存 profiled schedule：

```
ProfiledSchedule {
  vector<int64_t> size_ms;      // 升序排列的 profiled token 数
  vector<KernelMeta*> metas;    // 对应的最优 kernel (parallel array)
}
// 按 (N, K, E, compute_dtype, gemm_type) 为 key 存储
```

#### Profiling 流程 (`Preprocess` → `ProfileAllBuckets` → `ProfileSingleGemm`)

1. **`Preprocess`** (服务启动时调用一次): 对 GEMM1 (N=inter*2, K=hidden, top_k=expert_topk) 和 GEMM2 (N=hidden, K=inter, top_k=1) 分别 profiling。通过 `QUANT_PROFILE` 环境变量控制 iters（0=禁用，默认 15）。GEMM2 的 size_m 乘以 `expert_topk`（因 GEMM2 的实际 M = tokens × topk）。

2. **`ProfileAllBuckets`**: 对 2^0 ~ 2^13 (1~8192) 共 14 个 bucket 逐个 profiling。dummy 权重只准备一次（repack/process 昂贵，仅依赖 N/K/E/dtype），跨 bucket 复用。

3. **`ProfileSingleGemm`**: 对给定 (size_m, N, K) 的所有候选 kernel（按 N/K/gemm_type/dtype 过滤）逐个测时：
   - 为每个 distinct block_m 预跑 `InvokeAlignBlockSize` 生成对齐缓冲
   - **跳过 actual_tiles < num_ctas 的候选**: 避免 stream-K 死锁（tile 数不足以填满所有 CTA）
   - 每次 launch 前 `cudaMemsetAsync` 清零 locks（防止上次运行残留锁值导致死锁）
   - 用 `llm_kernels::utils::MeasureCudaExecutionTime` 测时 (warmup + profile_iters)
   - 返回最快 kernel 的 `KernelMeta*`

#### 运行时选择 (`SelectGemmMeta`)

Forward 中每个 GEMM 调用 `SelectGemmMeta(N, K, size_m)`：

1. **Profiled 优先**: `HummingMoeSearchStatus::GetProfiledMeta` 做 **right-align 二分查找**——找最小的 profiled size_m >= 实际 size_m。若实际 > max profiled，返回最大 profiled 条目（大 M 趋势一致）。
2. **Heuristic 兜底**: 若未 profiled，调用 `get_kernel_block_m`（→ `SelectKernel`）获取 heuristic 选择，并通过 `out_meta` 拿到 KernelMeta*。
3. **Block_m 兜底**: 若 heuristic 的 `SelectKernel` 也返回 nullptr（所有 kernel 是范围匹配且 size_m 落在范围外），返回 `required_block_m > 0`，由 `humming_moe_gemm` 内部按 block_m 线性搜索。

返回 `GemmMetaSelection { meta, block_m, required_block_m }`，传给 `humming_moe_gemm` 的 `forced_meta` / `required_block_m` 参数。

> **Right-align 策略原理**: 小 M 时 kernel 性能对 M 敏感（padding 占比高），用大于等于实际 M 的 profiled 点更保守（profiled 时 padding 更少，测出的耗时是上界）。大 M 时性能趋于稳定，用最大 profiled 点外推安全。

### GEMM2 block_m 解耦

#### 问题背景

GEMM1 (gate+up, 大 K 小 N) 和 GEMM2 (down, 小 K 大 N) 形状差异大，heuristic 选出的最优 `block_m` 通常不同。最初 GEMM2 复用 GEMM1 的 block_m，导致 GEMM2 用了次优 tile size，性能低下。

#### 解决方案

| 步骤 | GEMM1 | GEMM2 |
|------|-------|-------|
| SelectGemmMeta | 用 GEMM1 shape 选 sel1 | 用 GEMM2 shape 独立选 sel2 |
| Align | 用 `align_gemm1` workspace 对齐 | 若 `sel2.block_m != sel1.block_m`，用独立 `align_gemm2` workspace **重跑** `InvokeAlignBlockSize` |
| GEMM launch | `forced_meta=sel1.meta` | `forced_meta=sel2.meta` |

- **独立 align workspace**: `ws_.align_gemm2` 与 `ws_.align_gemm1` 是两块独立缓冲，避免 GEMM2 重对齐覆盖 GEMM1 尚未用完的对齐数据。
- **仅 block_m 不同时才重跑**: 若两个 GEMM 的 block_m 相同，GEMM2 直接复用 GEMM1 的 sorted_ids/expert_ids，省一次 align kernel。

#### 效果

Flash W4A8 GEMM2 TFLOPS（见 commit d2f2e755）：
- 64 tokens: 6.31 → 13.48 (+113%)
- 512 tokens: 21.73 → 80.72 (+271%)
- 8192 tokens: 198.21 → 198.94 (保持)

### Workspace 内存布局优化

所有 workspace 从单一连续缓冲 `workspace_buffer_` 中 carve-up，布局如下：

```
[gating.topk_weights] [gating.topk_ids]
[align_gemm1] [align_gemm2]
[cache.gemm1_output / cache.gemm2_output (共享)] [cache.silu_output]
[kernel.locks] [kernel.tma_descriptors]
[fp8.activation] [fp8.scales]
← softplus_sqrt 尾部切片 (与 fp8 区域时间安全重叠) →
```

| 优化 | 机制 | 理由 |
|------|------|------|
| **cache1/cache3 共享** | `gemm1_output` 和 `gemm2_output` 指向同一地址，workspace 取 `max(gemm1_size, gemm2_size)` | 生命周期不重叠：cache1 被 SiluAndMul 消费后才产生 cache3 |
| **softplus_sqrt 尾部切片** | topk 子层的 workspace 从 buffer 尾部切片 | gating 在 ProcessTokens 之前执行，与 fp8 区域时间安全重叠，省一份独立缓冲 |
| **num_ctas_per_sm 扫描** | `GetWorkspaceSizeT` 扫描 `jit_compiler_->GetKernelTable()` 取 max `num_ctas_per_sm`，而非硬编码 | heuristic 可能产生 num_ctas_per_sm>3 的 kernel（如 small-K large-N MoE），硬编码会欠分配 TMA 描述符缓冲 |
| **locks 4x 过配** | `get_locks_size` 返回值含 4x 余量 | num_ctas_per_sm 升级时安全，无需精确匹配 |

### W4A8 FP8 激活量化

W4A8 路径下，GEMM 前需将 BF16 激活量化为 FP8：

- **`QuantizeToFp8`**: 调用 `TritonWrapper::InvokeHummingFp8Quant` 生成 per-token FP8 scale。GEMM1 量化 `[tokens, hidden]`，GEMM2 量化 `[tokens*topk, inter]`。
- **per-expert global_scale**: W4A8 权重在预处理时 (`process_mxfp4_w4a8_weight`) 对每个 expert 计算 `global_scale = 2^(scale_min_new - 127)`，从权重 tensor 的 `weight_global_scale` 字段获取，传入 `humming_moe_gemm` 的 `global_scale` 参数。
- **W4A16 路径**: 不量化，`a_scale`/`global_scale` 传 nullptr，`compute_dtype=BF16`。

### Profiling 基础设施

`ProfileSingleGemm` 中的 profiling 实现：

| 要素 | 实现 |
|------|------|
| 权重 | `PrepareDummyWeights` 生成随机 FP4 权重 + E8M0 scale，走完整 repack/process 流水线（与真实推理一致），每 GEMM shape 只准备一次 |
| 激活/输出/locks/workspace | `torch::Tensor` (CUDA)，按候选 kernel 的 max block_m / max num_ctas_per_sm 分配 |
| topk_ids | `MakeDummyTopkIds`: `topk_ids[i*topk+j] = (i+j) % num_experts`，确保每个 expert 分到 token |
| 测时 | `llm_kernels::utils::MeasureCudaExecutionTime` (warmup + iters) |
| 流切换 | `at::cuda::setCurrentCUDAStream` 切到计算流，profiling 后恢复 |
| 异常处理 | try-catch 跳过 launch 失败的候选（smem 超限等） |

### 运行时性能统计 (`MoePerfStats`)

Layer 提供 `SetPerfStats(MoePerfStats*)` 接口，启用后每次 Forward 通过 CUDA event 测量 GEMM1/GEMM2 各自耗时，累加到 `perf_stats_->gemm1_time_ms` / `gemm2_time_ms`。事件惰性创建（首次 Forward 时 `cudaEventCreate`），跨 Forward 复用。默认 `nullptr`（禁用，零开销）。

## 关键设计决策与升级风险点

### 设计决策 rationale

| 决策 | 原因 |
|------|------|
| 本地拷贝 .cuh headers (非引用 pip 包) | 支持在项目内直接修改 kernel 源码进行二次开发（内存流水线、WGMMA、stream-K 调度等优化）|
| C++ 内嵌 ELF 解析器 (非 pyelftools) | 去除运行时 Python 依赖读取 cubin 元数据；cubin 缓存命中后纯 C++ 启动 |
| 延迟 cuModuleLoad 到主线程 | CUDA context 不跨线程继承，子线程并行编译后必须回主线程加载 |
| Right-align profiled 查找 | 小 M 对 tile 浪费敏感，用 ≥ 实际 M 的 profiled 点更保守；大 M 趋势一致可外推 |
| `KSANA_HUMMING_BASE_DIR` 不用于 include 解析 | 该变量指向 `jit/` 目录（用于 humming_tool.py），复用会导致 `.../jit/include` 错误路径 |
| GEMM2 top_k=1 | silu_output 已展开为 token-expert pair，top_k=1 使 kernel 直接用 sorted_ids 作行索引 |

### 升级风险点（humming 版本升级时最可能受影响）

| 风险项 | 位置 | 影响 | 应对 |
|--------|------|------|------|
| **17 个 kernel 参数** | `launcher.cu` arg packing | humming 改 kernel 签名会导致 launch 参数错位 | 对比 humming `csrc/launcher/launcher.cpp` 的 kernel 接口 |
| **`__constant__` 符号名** | `LoadAndRegister` ELF 读取 | 符号改名/增删会导致元数据读取失败 | 对比 humming `.cuh` 中的 `extern "C" __constant__` 声明 |
| **`HummingKernel.__init__()` 参数** | `humming_tool.py` generate | 构造参数增减需同步 | 见 [常见 API 变化点](#常见-api-变化点) |
| **`get_heuristics_class()` / `get_configs()` 返回格式** | `humming_tool.py` query-configs | configs 结构变化影响 JSON 解析 | 检查 `entry[0]/[1]/[2]` 的 tuple 结构 |
| **`kernel.code` / `kernel.kernel_expr`** | `humming_tool.py` generate | 输出格式变化影响源码生成和符号提取 | 验证生成的 .cu 文件可被 nvcc 编译 |
| **`humming.ops` 导入副作用** | `humming_tool.py` sys.modules stub | humming `__init__.py` 改变导入行为可能使 stub 失效 | 检查 `import humming` 是否触发不必要的 CUDA/编译 |
| **.cuh header 内容** | `include/humming/` | 本地修改与新版冲突 | rsync 后 `git diff` 重新应用本地改动 |
| **`NVCCCompiler.get_flags()`** | `CompileCubin` nvcc flags | 编译参数变化影响 cubin 生成 | 对比 humming 最新 flags |
| **`KernelRuntime.prepare` / `init_sm_version`** | `humming_tool.py` monkey-patch | 初始化流程改变使 patch 失效 | 检查 `HummingKernel.__post_init__` 调用链 |
| **`kHummingMoeMaxBlockSize` (=64)** | `humming_fp4_moe_layer.cpp` workspace 预分配 | humming heuristic 引入 block_m > 64 的 kernel 时，align buffer 预分配不足 | 升级后检查 heuristic 返回的 block_m 范围，必要时增大常量 |

## 更新操作指南

### 当 humming-kernels 版本升级时

```bash
# 1. 安装新版本
pip install humming-kernels==<new_version>

# 2. 确认 humming 安装路径
HUMMING_PATH=$(python3 -c "import humming; import os; print(os.path.dirname(humming.__file__))")

# 3. 同步本地 .cuh header 副本 (从新版本 pip 包)
rsync -a --delete "$HUMMING_PATH/include/humming/" \
  3rdparty/LLM_kernels/csrc/kernels/nvidia/others/humming/include/humming/

# 4. Review the diff — 重新应用任何本地修改 (二次开发改动)
git diff 3rdparty/LLM_kernels/csrc/kernels/nvidia/others/humming/include/

# 5. Re-run cmake to pick up any CMakeLists.txt changes
cd build && cmake ..

# 6. 验证编译
cd build && ninja -j64

# 7. 运行测试 (清空 JIT 缓存以触发重新编译)
cd build && rm -rf ~/.cache/KsanaLLM/humming_kernel_files/ && \
  CUDA_VISIBLE_DEVICES=3 ./bin/fp4_moe_layers_test

# 8. 如果 humming Python API 有变化，需同步修改 humming_tool.py
# 主要关注:
#   - HummingKernel 构造函数参数
#   - get_heuristics_class() 工厂函数
#   - kernel.code 输出格式
#   - kernel.kernel_expr 格式
```

### 常见 API 变化点

| humming 组件 | 当前使用方式 | 变化风险 |
|-------------|-------------|---------|
| `HummingKernel.__init__()` | flat params (block_shape, warp_shape, ...) | 中：参数可能增减 |
| `get_heuristics_class()` | factory, returns DeviceHeuristics subclass | 低：动态调度，自动适配 GPU |
| `HummingLayerMeta` | from `humming.layer` | 低 |
| `kernel.code` | 完整 .cu 源码字符串 | 低：核心功能 |
| `kernel.kernel_expr` | 模板实例化表达式 | 中：可能随模板参数变化 |
| `.cuh` header 文件 | 由 nvcc 在 JIT 时 include | 高：核心逻辑可能变化 |

### humming_tool.py 中的 monkey-patch

当前 `jit/humming_tool.py`（`generate` 子命令）中有一个 monkey-patch:
```python
# Monkey-patch to avoid needing CUDA context during source generation
KernelRuntime.prepare = lambda self: None
KernelRuntime.init_sm_version = lambda self: (
    setattr(self, "sm_version", 90) or setattr(self, "sm_version_str", "90a")
)
```

这是因为 `HummingKernel.__post_init__()` 会调用 `init_kernel()` → `prepare()` 尝试实际编译内核，但我们只需要 `kernel.code` 源码。`init_sm_version` 被替换为直接硬编码 SM 90a（避免查询设备）。如果未来 humming 改变了初始化流程，此 patch 可能需要调整。

> 注意: `humming_tool.py` 还通过预填充 `sys.modules["humming.ops"]` 等为 stub 模块，阻止 `humming/__init__.py` 中 `import humming.ops` 的副作用（注册 9 个 torch custom op + 后台 JIT 编译 launcher 扩展，约 6s）。KsanaLLM 只需要 `humming.tune`/`humming.layer`/`humming.kernel`/`humming.config`，不需要 `humming.ops`。

## 文件变更记录

| 日期 | 操作 | 说明 |
|------|------|------|
| 2026-05-16 | 初始迁移 | humming-kernels 0.1.0 @ 13c42d05 |
| 2026-05-28 | 移除本地 .cuh 副本 | Headers 改为编译时从 pip 包 include 路径引用，支持 `$HUMMING_INCLUDE_DIR` 覆盖 |
| 2026-06-03 | 版本升级适配 | humming-kernels 0.1.0→0.1.3 (13c42d05→a373046)。适配: process_mxfp4_w4a8.cu (max_range 12→11, +1步, global_scale /2), repack_weight.cu (+interleave_mode参数) |
| 2026-06-12 | H200 支持修复 | humming_tool.py: 硬编码 Sm90H20Heuristics → humming.tune.get_heuristics_class() 动态调度。C++ 侧 QueryH20Configs → QueryKernelConfigs 重命名，清理 H20 引用。 |
| 2026-06-12 | 版本升级适配 | humming-kernels 0.1.3→0.1.5 (a373046→d596b9f)。适配: humming_jit_compiler.cpp error消息更新版本引用, humming_tool.py 新增 humming.ops.hadamard stub, CMakeLists.txt 更新 commit ref。无需更新 .cuh header (仍通过 pip 包 include 自动引用)。humming API (HummingKernel, KernelRuntime, get_heuristics_class, LayerMeta) 未变。预处理 header (process.cuh, process_mxfp4.cuh, pack_weight.cuh) 未变。 |
| 2026-06-18 | 恢复本地 .cuh 副本 | .cuh headers 改回本地拷贝 (从 0.1.5 pip 包 d596b9f)。JIT 编译器 (`ResolveLocalIncludeDir` 替代 `ResolvePipIncludeDir`) 和 `preprocess/CMakeLists.txt` 改为使用本地 include 路径。保留 auto-install (用于 JIT 运行时 Python 依赖)。支持在项目内直接修改 humming 算子源码进行二次开发。 |
| 2026-06-19 | include 路径修复 | `ResolveLocalIncludeDir` 移除对 `$KSANA_HUMMING_BASE_DIR` 的误用 (该变量指向 `jit/` 目录，非 humming base)。仅使用 `$HUMMING_INCLUDE_DIR` 或 `__FILE__` 推导。 |
| 2026-06-19 | GEMM2 优化 (phase1) | `KernelConfig` 新增 `force_stream_k`/`force_block_n`/`force_stages` 覆盖字段。`humming_tool.py` 新增对应 CLI 参数。`QueryKernelConfigs` 为小-K 大-N GEMM2 形状注入 stream-K + block_n=256 配置。GEMM2 block_m 解耦：独立 re-align + 专用 `fused_id_buffer_w2` workspace。 |
| 2026-06-22 | Profile-based kernel 选择支持 | `humming_moe_gemm` 新增 `forced_meta` 参数，允许调用方直接指定 kernel。`get_kernel_block_m` 新增 `out_meta` 输出参数。`QueryKernelConfigs` 生成 variant configs (stream_k/block_n/stages 变体) 扩展候选空间。`KernelConfig::CacheKey()` 加入 force 覆盖字段。 |
