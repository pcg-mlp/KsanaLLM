# Humming MoE Operator Migration Tracking

## 算子出处

| 字段 | 值 |
|------|------|
| 原始项目 | [inclusionAI/humming](https://github.com/inclusionAI/humming) |
| pip 包名 | `humming-kernels` |
| 迁移时版本 | `0.1.0` |
| 源码 commit | `13c42d05d27b9e4186500da98cda2a484686fa44` |
| 安装来源 | `pip install git+https://github.com/inclusionAI/humming.git@13c42d05` |
| 迁移日期 | 2026-05-16 |
| 目标分支 | `integrate-humming` |
| License | 见原始仓库 LICENSE |

## 迁移范围

### .cuh Header 文件

**不再本地拷贝**。`.cuh` header 文件在编译时从 `humming-kernels` pip 包的 `include/` 目录直接引用。

**Source**: pip package `humming/include/` directory (auto-detected at build/JIT time).

**Include path resolution**:
- **JIT 编译**: `nvcc -I` 指向 pip 包 include 路径。`HummingJitCompiler` 构造函数按以下顺序解析:
  1. `$HUMMING_INCLUDE_DIR` 环境变量（显式覆盖）
  2. `python3 -c "import humming; ..."` 自动检测
  3. `$KSANA_HUMMING_BASE_DIR + "/include"` 回退（向后兼容）
- **CMake 构建** (preprocess): `execute_process` 在 cmake configure 时查询 pip 包路径，失败时回退到本地路径并发出 warning。
  - 优先使用 `find_package(Python3)` 获得的 `${Python3_EXECUTABLE}`
  - 回退到 `find_program(python3)`

Header 内容未修改，与 pip 包中的版本完全一致。更新 pip 包后重新运行 cmake configure 即可生效。

### KsanaLLM 自研代码

以下代码为迁移过程中新写的，**非拷贝**:

| 文件 | 职责 |
|------|------|
| `jit/humming_tool.py` | 统一 Python 工具 (generate + query-configs 子命令，调用 humming API) |
| `jit/humming_jit_compiler.h` | JIT 编译器头文件 |
| `jit/humming_jit_compiler.cpp` | JIT 编译器实现 (含 ELF 解析) |
| `launcher/tma.h` / `tma.cu` | TMA 描述符构建 (KsanaLLM 版) |
| `launcher/launcher.h` / `launcher.cu` | Kernel 启动 (KsanaLLM 版) |
| `launcher/kernel_meta.h` | 内核元数据结构 |
| `ops/ops.h` / `ops.cu` | 公共 API + dispatch |
| `ops/wrapper.h` / `wrapper.cu` | Workspace sizing |
| `preprocess/humming_preprocess.h` | 权重预处理接口 (stub) |
| `tests/humming_jit_test.cu` | JIT 编译烟雾测试 |

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

## 与原生 humming 的差异

| 方面 | 原生 humming (pip 包) | KsanaLLM 迁移版 |
|------|----------------------|-----------------|
| 框架绑定 | PyTorch (pybind11, torch.Tensor) | 无框架依赖 (raw CUDA pointers) |
| Kernel 注册 | Python 侧 `register_kernel()` | C++ `HummingJitCompiler::LoadAndRegister()` |
| TMA 描述符 | 使用 PyTorch Tensor shape/stride | 手动传入 dims/strides |
| ELF 解析 | Python `pyelftools` | C++ 内嵌 ELF 解析器 (自研) |
| Launch API | `torch.ops.humming.launch_kernel` | `llm_kernels::nvidia::humming::launch_kernel()` |
| 编译调度 | humming JIT runtime (自动) | `HummingJitCompiler::CompileKernels()` (显式) |
| 缓存目录 | `~/.cache/humming/` | `HUMMING_CACHE_DIR` (默认 `~/.cache/humming_kllm/`) |
| Kernel 选择 | Python 侧 configs 数组 | C++ `SelectKernel()` 最小化 M-tile 浪费 |
| .cuh headers | pip 包内 include/ | 编译时从 pip 包 include 路径引用 |

### 未修改项

- **Kernel 源码** (.cuh 模板): 完全一致，verbatim copy
- **nvcc 编译参数**: 完全一致 (flags 从 humming 的 `NVCCCompiler.get_flags()` 抄录)
- **Kernel arguments**: 17 个参数位置完全一致 (从 humming 原始 csrc/launcher/launcher.cpp 验证)
- **extern "C" __constant__ 元数据**: 同样从 cubin ELF 中读取

## 更新操作指南

### 当 humming-kernels 版本升级时

```bash
# 1. 安装新版本
pip install humming-kernels==<new_version>

# 2. 确认 humming 安装路径
HUMMING_PATH=$(python3 -c "import humming; import os; print(os.path.dirname(humming.__file__))")

# 3. Headers are auto-resolved from pip package — no manual sync needed
#    Verify include path auto-detection:
python3 -c "import humming; import os; print(os.path.join(os.path.dirname(humming.__file__), 'include'))"

# 4. Re-run cmake to pick up new include path (if pip package path changed)
cd build && cmake ..

# 5. 验证编译
cd build && make -j llm_kernels_nvidia_kernel_humming_ops

# 6. 运行测试
HUMMING_TEST_SOURCE_DIR=<path_to_humming_dir> ./bin/llm_kernels_nvidia_kernel_humming_test

# 7. 如果 humming Python API 有变化，需同步修改 humming_tool.py
# 主要关注:
#   - HummingKernel 构造函数参数
#   - Sm90H20Heuristics.get_config() 接口
#   - kernel.code 输出格式
#   - kernel.kernel_expr 格式
```

### 常见 API 变化点

| humming 组件 | 当前使用方式 | 变化风险 |
|-------------|-------------|---------|
| `HummingKernel.__init__()` | flat params (block_shape, warp_shape, ...) | 中：参数可能增减 |
| `Sm90H20Heuristics.get_config()` | classmethod, returns dict | 低：稳定接口 |
| `HummingLayerMeta` | from `humming.layer` | 低 |
| `kernel.code` | 完整 .cu 源码字符串 | 低：核心功能 |
| `kernel.kernel_expr` | 模板实例化表达式 | 中：可能随模板参数变化 |
| `.cuh` header 文件 | 由 nvcc 在 JIT 时 include | 高：核心逻辑可能变化 |

### humming_tool.py 中的 monkey-patch

当前 `jit/humming_tool.py`（`generate` 子命令）中有一个 monkey-patch:
```python
# Monkey-patch to avoid needing CUDA context during source generation
from humming.jit.runtime import KernelRuntime
KernelRuntime.prepare = lambda self: None
KernelRuntime.init_sm_version = lambda self: None
```

这是因为 `HummingKernel.__post_init__()` 会调用 `init_kernel()` → `prepare()` 尝试实际编译内核，但我们只需要 `kernel.code` 源码。如果未来 humming 改变了初始化流程，此 patch 可能需要调整。

## 性能参考

来源: humming 官方 benchmark (H200, DeepSeek-V4-Flash, vllm)

| Backend | Prefill TPS | vs Marlin |
|---------|-------------|-----------|
| Marlin W4A16 | 7,272 | baseline |
| Humming W4A16 | 9,060 | +25% |
| Humming W4A8 | 9,917 | +36% |

## 文件变更记录

| 日期 | 操作 | 说明 |
|------|------|------|
| 2026-05-16 | 初始迁移 | humming-kernels 0.1.0 @ 13c42d05 |
| 2026-05-28 | 移除本地 .cuh 副本 | Headers 改为编译时从 pip 包 include 路径引用，支持 `$HUMMING_INCLUDE_DIR` 覆盖 |
| 2026-06-03 | 版本升级适配 | humming-kernels 0.1.0→0.1.3 (13c42d05→a373046)。适配: process_mxfp4_w4a8.cu (max_range 12→11, +1步, global_scale /2), repack_weight.cu (+interleave_mode参数) |
