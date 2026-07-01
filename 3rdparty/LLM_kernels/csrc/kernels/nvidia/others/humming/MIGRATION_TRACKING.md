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
| `launcher/kernel_meta.h` | 内核元数据 `KernelMeta` (含 `min_shape_m`/`max_shape_m` 范围字段) |
| `ops/ops.h` / `ops.cu` | 公共 API + dispatch (`forced_meta`/`required_block_m` 参数、`get_kernel_block_m` 带 `out_meta`) |
| `ops/wrapper.h` / `wrapper.cu` | Workspace sizing |
| `preprocess/humming_preprocess.h` | 权重预处理接口 (`pack_weight_fp4`/`repack_weight_wgmma`/`process_mxfp4_w4a8_weight`/`process_scales`) |
| `preprocess/pack_weight.cu` / `repack_weight.cu` / `process_mxfp4_w4a8.cu` / `process_scales.cu` | 权重预处理实现 (实例化 humming 模板 kernel) |
| `humming_stubs.cpp` | **非 SM90 平台** stub 实现 (所有符号 no-op/零返回，运行时调用打 ERROR 安全返回) |

**算子调用层** (仓库根 `csrc/`):

| 文件 | 职责 |
|------|------|
| `csrc/layers/humming_fp4_moe_layer.h` | Humming FP4 MoE Layer 头文件 (workspace 布局、profiling 接口、18-event per-phase profiling、`GetJitKernelTable` getter) |
| `csrc/layers/nvidia/humming_fp4_moe_layer.cpp` | Layer 实现 (Forward/Preprocess、profile-based kernel 选择、GEMM2 block_m 解耦、merged align、fused silu+quant、W4A8 量化) |
| `csrc/layers/moe_perf_stats.h` | `MoePerfStats` (GEMM 耗时 + 子相位 + kernel meta 指针 + FLOP/TFLOPS) |
| `csrc/utils/search_status.h` | `HummingMoeSearchStatus` 单例 (profiled schedule 缓存 + right-align + tile-waste-aware) |
| `csrc/layers/fp4_moe_layer_test.cpp` | FP4 MoE 测试 (W4A16/W4A8 基准、精度验证、JIT kernel 表打印) |

> 另有 `3rdparty/LLM_kernels/csrc/kernels/nvidia/moe_utils/moe_utils.{cu,h}` 新增 `InvokeSglMoeAlignBlockSizeMerged` (GEMM1/GEMM2 双 block_m 一次 align)；`3rdparty/LLM_kernels/csrc/kernels/nvidia/moe/humming_triton_fp8_quant/generate_triton_silu_mul_quant.py` 为融合 SiluAndMul+FP8 量化 Triton kernel 生成器。两者路径根为 `3rdparty/LLM_kernels/csrc/`。

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
| `QUANT_PROFILE` | profiling 迭代次数，`0` 禁用。Layer `Init` 读一次存 `profile_iters_`，`warmup_iters_=max(1,profile_iters_/3)`，`compile_extend=(profile_iters_!=0)` 显式传入 `CompileKernels` (见 [候选配置生成 > 条件编译](#条件编译)) | `15` |
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
| TMA 描述符 | PyTorch Tensor shape/stride | 手动 dims/strides；FP4 B/Bs 按 `make_tma_desc_b/bs` 重写 | 见 [TMA 描述符修复](#tma-描述符修复) |
| 缓存目录 | `~/.humming/cache/` | `$KSANA_HUMMING_KERNEL_PATH` | 独立缓存隔离 |
| .cuh headers | pip 包内 include/ | 本地 include/ 副本 (0.1.5 d596b9f) | 支持项目内二次开发 |
| GEMM2 优化 | 无 | stream-K + block_n=256 + 多 block_m 注入，M 分档 | 见 [GEMM2 block_m 解耦](#gemm2-block_m-解耦) |
| 配置变体 | 仅 heuristic | heuristic + variant (stream_k/block_n/stages/block_k/num_ctas + 组合) | 扩展候选空间供 profiling 选择 |

**未修改项**: .cuh 模板 verbatim copy；17 个 kernel 参数位置一致；`extern "C" __constant__` 元数据从 cubin ELF 读取。nvcc 编译参数核心一致（从 `NVCCCompiler.get_flags()` 抄录），差异: 省略 `-fopenmp`/`-lgomp`，硬编码 `sm_90a`。

## JIT 编译器优化机制

`HummingJitCompiler` (`jit/humming_jit_compiler.cpp`) 是 KsanaLLM 自研代码，**非 humming 原生组件**。

### <a id="jit-流水线"></a>JIT 流水线

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
| GEMM2 注入 | `_inject_gemm2_configs` | 见下表 |
| variant 生成 | `_generate_variants` | 见下表 |
| 去重 | `_config_identity` | 14 字段 tuple (block_m/n/k, warp, stream_k, stages, ctas, tma_a/b/c/bs, mbarrier, wspec, f16acc) |
| 校验 | `_validate_config` | monkey-patch 下构造 `HummingKernel`，异常则丢弃 |
| cache_key | `_make_cache_key` | 含全部最终字段的确定性字符串 |

`generate` 子命令接收完整配置 CLI 参数 (block_n/block_k/warp_shape/use_stream_k/num_stages/num_ctas_per_sm/use_tma_*/use_mbarrier/use_warp_spec/use_f16_accum)，不再查询 heuristic。`KernelConfig` 存最终值 + `cache_key` + `is_heuristic`，不再有 `force_*` override 字段。

**Fallback**: Python 侧 `get_configs()` 异常 → 输出 2 个 fallback config；C++ 侧 Python/JSON 失败 → 构造 block_m=64/128 的 `KernelConfig`。两层防御。

**Variant Config 生成** (`_generate_variants`): 每个 heuristic base config 生成变体，硬上限 72 个。变体 `min_shape_m`/`max_shape_m` 重置为 `[1, INT64_MAX]`（override 后 heuristic M 范围不再适用），由 profiling 选出赢家。

| 变体 | 覆盖字段 | 理由 |
|------|---------|------|
| stream_k | `use_stream_k=True` | 翻转 heuristic 小-K 关闭 |
| block_n=256 | `block_n=256` | 提升 large-N 吞吐 |
| stages ∈ {3,4,5} | `num_stages` | 探索 pipeline 深度 |
| block_k=64 (FP8, K>64) | `block_k=64` + `warp_shape[2]=64` | 放宽 K≤512 限制 |
| num_ctas ∈ {1,2,3,4} | `num_ctas_per_sm` | 小 M 减 tile 浪费、增寄存器 |
| stages4+bn256 / stages5+ctas2 | 组合 | 交叉探索 |

**GEMM2 注入** (`_inject_gemm2_configs`): 针对 `size_k<=512 && size_n>=4096` 的 small-K large-N 形状，heuristic 对小-中 M 不优。注入前移除 `max_shape_m<=3072` 的 heuristic config:

| M 范围 (max_shape_m) | 注入 block_m | 配置 |
|---------------------|-------------|------|
| 1-384 | 8, 64 | `stream_k=True, block_n=256, stages=3` |
| 385-1536 | 16, 32, 64 | 同上 |
| 1537-3072 | 48, 64 | 同上 |
| 3073+ | (保留 heuristic) | heuristic 已近优 |

### <a id="条件编译"></a>条件编译 (compile_extend_configs)

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
   ┌──── merged align (block_m 不同 && expert>=224): 一次 launch 算 GEMM1+GEMM2 align ────┐
   │ 否则: InvokeAlignBlockSize(GEMM1)，GEMM2 align 延后                                  │
   ▼                                                                                     ▼
    hidden_states ──► [W4A8: QuantizeToFp8] ──► humming_moe_gemm (GEMM1, w1)
                                                       │
                                                       ▼
    W4A16: SiluAndMul            W4A8: FusedSiluAndMul+Fp8Quant (一次 Triton kernel)
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

### <a id="profile-based-kernel-选择机制"></a>Profile-based Kernel 选择机制

heuristic 基于通用 GPU 模型，对 H20 等特定卡未必最优。本分支引入启动时 profiling。

**`HummingMoeSearchStatus`** (`csrc/utils/search_status.h`) 单例，按 `(N,K,E,compute_dtype,gemm_type)` 存储 `ProfiledSchedule { size_ms (升序), metas, block_ms }` (parallel arrays)。

**Profiling 流程** (`Preprocess` → `ProfileAllBuckets` → `ProfileSingleGemm`):
1. `Preprocess` 对 GEMM1 (N=inter*2, K=hidden, top_k=expert_topk) 和 GEMM2 (N=hidden, K=inter, top_k=1, size_m×=expert_topk) 分别 profiling。iters 由 `profile_iters_` (`QUANT_PROFILE`) 控制，`warmup_iters_=max(1,profile_iters_/3)`。
2. `ProfileAllBuckets`: 26 个 bucket `kBuckets = {1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096,6144,8192}` (中间值覆盖 decode batch size，中点减半 right-align gap)。dummy 权重仅依赖 N/K/E/dtype，跨 bucket 复用。dummy topk_ids 用**随机均匀采样** (seed=42 `std::mt19937`)——匹配 benchmark 真实 expert 分布 (小 M 稀疏、大 M 收敛均匀)，避免选出仅均匀负载下快的 kernel。
3. `ProfileSingleGemm` 对所有候选 kernel 逐个测时: M 范围过滤 (跳过 `max_shape_m < size_m`)；per distinct block_m 预跑 align；跳过 `actual_tiles < num_ctas` (防 stream-K 死锁)；launch 前清零 locks；`MeasureCudaExecutionTime` (warmup+iters)；catch 块 `cudaDeviceSynchronize()`+`cudaGetLastError()` 清除 sticky CUDA error。返回最快 kernel。

**运行时选择** (`SelectGemmMeta`):
1. **Profiled 优先**: `GetProfiledMeta` right-align 二分查找 (最小 profiled size_m ≥ 实际；若实际 > max profiled，返回最大条目)。**tile-waste-aware**: 若 right-align kernel padding 浪费 >25%，检查下一个更小 bucket，选浪费更小者 (需 `waste_left < waste_right - 5pp`)。
2. **Heuristic 兜底**: `get_kernel_block_m` (→ `SelectKernel`) + `out_meta`。
3. **Block_m 兜底**: 返回 `required_block_m > 0`，由 `humming_moe_gemm` 内部线性搜索。

> **Right-align 原理**: 小 M 对 padding 敏感，用 ≥ 实际 M 的 profiled 点更保守 (耗时上界)；大 M 趋势一致可外推。tile-waste-aware 修正 right-align 系统性偏向大 block_m kernel 的偏差。

### <a id="gemm2-block_m-解耦"></a>GEMM2 block_m 解耦

GEMM1 (gate+up, 大 K 小 N) 和 GEMM2 (down, 小 K 大 N) 形状差异大，heuristic 最优 `block_m` 通常不同。GEMM2 的 `SelectGemmMeta` 在 align 之前独立完成，`forced_meta=sel2.meta`。

**两条 align 路径** (`use_merged_align = (sel2.block_m != sel1.block_m) && expert_num_per_node_ >= 224`):

| 路径 | 条件 | 行为 |
|------|------|------|
| **Merged align** | block_m 不同 && expert ≥ 224 | `InvokeSglMoeAlignBlockSizeMerged` 一次 launch 算两个 block_m 的 align (共享 per-expert token 计数) |
| **Deferred separate** | block_m 不同 && expert < 224 | GEMM1 先 align；GEMM2 用独立 `align_gemm2` workspace 重跑 `InvokeAlignBlockSize` |
| **复用** | block_m 相同 | GEMM2 直接复用 GEMM1 的 sorted_ids/expert_ids |

`ws_.align_gemm2` 与 `ws_.align_gemm1` 独立缓冲，避免重对齐覆盖未用完数据。

效果 (Flash W4A8 GEMM2 TFLOPS, commit `d2f2e755`): 64 tokens 6.31→13.48 (+113%)，512 tokens 21.73→80.72 (+271%)，8192 tokens 198.21→198.94 (保持)。

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
| **locks 4x 过配** | `get_locks_size` 含 4x 余量 | num_ctas_per_sm 升级时安全 |

### W4A8 FP8 激活量化

- **GEMM1** (`QuantizeToFp8`): `TritonWrapper::InvokeHummingFp8Quant` 生成 per-token FP8 scale，量化 `[tokens, hidden]`。
- **GEMM2** (融合 SiluAndMul+Quant): `InvokeHummingFusedSiluAndMulFp8Quant` 读 `[M*topk, inter*2]` BF16，写 `[M*topk, inter]` FP8 + `[M*topk]` FP32 scale (per-token absmax/448.0)。省一次 kernel launch + HBM round-trip。
- **per-expert global_scale**: 预处理时 `process_mxfp4_w4a8_weight` 计算 `global_scale = 2^(scale_min_new-128)`，从 `weight_global_scale` 字段获取。
- **W4A16**: 不量化，`a_scale`/`global_scale` 传 nullptr，`compute_dtype=BF16`，`InvokeSiluAndMul`。

### Profiling 基础设施

| 要素 | 实现 |
|------|------|
| 权重 | `PrepareDummyWeights` 随机 FP4 + E8M0 scale，走完整 repack/process，每 shape 一次 |
| 激活/输出/locks/workspace | `torch::Tensor` (CUDA)，按 max block_m / max num_ctas_per_sm 分配 |
| topk_ids | `MakeDummyTopkIds` 随机均匀采样 (seed=42)，匹配 benchmark 分布 |
| 测时 | `MeasureCudaExecutionTime` (warmup + iters) |
| 流切换 | `at::cuda::setCurrentCUDAStream` 切计算流，后恢复 |
| 异常处理 | try-catch 跳过 launch 失败候选 (smem 超限等) |

### 运行时性能统计 (`MoePerfStats`)

`SetPerfStats(MoePerfStats*)` 启用后，每次 Forward 通过 **18 个 CUDA event** (`std::vector<cudaEvent_t>` + `enum class Evt`: Gate/Align1/Quant1/Gemm1/Silu/Align2/Quant2/Gemm2/MulSum 各 Start/End) 测量。事件惰性创建 (`EnsureProfilingEvents`)，析构统一 destroy，跨 Forward 复用。默认 `nullptr` (零开销)。

累加子相位: `gating_time_ms` / `align_time_ms`(=Align1+Align2) / `quant_time_ms`(=Quant1+Quant2) / `gemm1_time_ms` / `silu_time_ms` / `gemm2_time_ms` / `mulsum_time_ms` / `total_time_ms`。另存 `last_gemm1_meta`/`last_gemm2_meta` + FLOP 计数 (TFLOPS)。静态 `GetJitKernelTable()` 返回 JIT `kernel_table_` 指针供测试打印。Forward 结束 `cudaEventSynchronize(MulSumEnd)`。

### <a id="tma-描述符修复"></a>TMA 描述符修复 (`launcher.cu`)

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
| **`kHummingMoeMaxBlockSize` (=64)** | `humming_fp4_moe_layer.cpp` | heuristic 引入 block_m > 64 时 align buffer 不足 | 升级后检查 block_m 范围，必要时增大常量 |
| **`humming_stubs.cpp` 符号同步** | `humming_stubs.cpp` | 接口签名变化时 stub 需同步否则非 SM90 链接失败 | 升级后跑非 SM90 构建验证 |

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

## 文件变更记录

| 日期 | 操作 | 说明 |
|------|------|------|
| 2026-05-16 | 初始迁移 | humming-kernels 0.1.0 @ 13c42d05 |
| 2026-06-02 | .cuh 引用方式调整 | 先移除本地副本改用 pip 包 include 路径 (06-02)，后于 06-18 恢复本地拷贝 (0.1.5 d596b9f)，支持项目内二次开发 |
| 2026-06-03 | 版本升级 0.1.0→0.1.3 | process_mxfp4_w4a8.cu (max_range 12→11, global_scale /2), repack_weight.cu (+interleave_mode) |
| 2026-06-05 | FP8 量化 Triton 内核 | 新增 `generate_triton_silu_mul_quant.py` + `InvokeHummingFp8Quant` (W4A8 GEMM1 前激活量化) |
| 2026-06-12 | H200 支持 + 升级 0.1.3→0.1.5 | Sm90H20Heuristics → `get_heuristics_class()` 动态调度；0.1.3→0.1.5 (a373046→d596b9f)，API 未变 |
| 2026-06-18 | include 路径修复 | `ResolveLocalIncludeDir` 移除 `$KSANA_HUMMING_BASE_DIR` 误用 |
| 2026-06-19 | GEMM2 优化 (phase1) | `KernelConfig` 新增 force_* 覆盖字段 (后于 06-29 重构删除)；GEMM2 block_m 解耦 + stream-K + block_n=256 注入 |
| 2026-06-22 | Profile-based kernel 选择 | `humming_moe_gemm` 新增 `forced_meta`；`get_kernel_block_m` 新增 `out_meta`；variant configs 扩展候选空间 |
| 2026-06-26 | 开发工具 | clangd LSP + `compile_commands.json` 导出 |
| 2026-06-27 | Profiling 基础设施 | `MoePerfStats` 子相位耗时 + kernel meta + FLOP；4 event → 18-event 向量；`GetJitKernelTable()` getter；`BenchResult` meta 改 by value 修 use-after-free |
| 2026-06-28 | W4A8 性能优化 | TMA 描述符修复 (swizzle=0, pack_size_k=256/a_num_bits)；融合 SiluAndMul+FP8Quant Triton kernel；`InvokeSglMoeAlignBlockSizeMerged`；GEMM2 block_m 提前到 align 前；sticky CUDA error 清除 |
| 2026-06-28 | Variant 扩展 + 非致命编译 | stages {3,4,5}、num_ctas {1,2,3,4}、block_k=64、组合变体 (cap 72)；GEMM2 注入加 block_m=8 + medium M 分档；`max_shape_m < size_m` 过滤；nvcc 失败非致命 (C4+BF16 寄存器不足) |
| 2026-06-28 | Profile buckets + tile-waste-aware | 14→26 bucket (含 decode batch size)；right-align 后 >25% padding 浪费检查更小 bucket；`ProfiledSchedule` 加 `block_ms` 向量 |
| 2026-06-29 | Profiler expert 分布修复 | `MakeDummyTopkIds` round-robin → 随机均匀 (seed=42)。Flash TP4 2048 +9.16, Pro TP8 2048 +4.85 TFLOPS |
| 2026-06-29 | 候选配置生成迁移到 Python | `query-configs` 拥有全部候选生成；`generate` 改接收完整配置 CLI；C++ 用 nlohmann/json 消费，`KernelConfig` 改存最终值 (删 force_*)；variant M 范围重置为全范围 (含 06-29 修复) |
| 2026-06-30 | JIT 编译器修复 | `compile_extend_configs` 显式传入 (单一真相源)；WARNING 门控 `if (verbose_)`；`dir_mutexes_` 裸 `new` → `unique_ptr` 修泄漏；`--active_count` 移出 verbose 块修非 verbose 挂起；移除 `humming_stubs.cpp` stale `CacheKey()` 修非 SM90 链接 |
