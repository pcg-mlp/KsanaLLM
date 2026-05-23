# Marlin MoE WNA16 算子迁移文档

## 1. 算子来源

### 源项目

- **项目**: [SGLang](https://github.com/sgl-project/sglang)
- **仓库地址**: `https://github.com/sgl-project/sglang.git`
- **迁移时仓库 HEAD commit**: `b69485ff4272b5a306045fa0cf7fc2c5692d391f`

### 源文件路径与 commit

| 源文件路径 | 最后相关 commit | 说明 |
|-----------|----------------|------|
| `sgl-kernel/csrc/moe/marlin_moe_wna16/ops.cu` | `2ef83105bea3` (2026-04-28) | 算子入口和调度逻辑 |
| `sgl-kernel/csrc/moe/marlin_moe_wna16/marlin_template.h` | `2ef83105bea3` (2026-04-28) | 核心 kernel 实现模板 |
| `sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h` | `2ef83105bea3` (2026-04-28) | Kernel 前向声明 |
| `sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_marlin.cuh` | `2ef83105bea3` (2026-04-28) | Kernel 实例化汇总 |
| `sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4.cuh` | `2ef83105bea3` (2026-04-28) | BF16 U4 模板实例化 |
| `sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4b8.cuh` | `2ef83105bea3` (2026-04-28) | BF16 U4B8 模板实例化 |
| `sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku8b128.cuh` | `2ef83105bea3` (2026-04-28) | BF16 U8B128 模板实例化 |
| `sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4.cuh` | `2ef83105bea3` (2026-04-28) | FP16 U4 模板实例化 |
| `sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4b8.cuh` | `2ef83105bea3` (2026-04-28) | FP16 U4B8 模板实例化 |
| `sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku8b128.cuh` | `2ef83105bea3` (2026-04-28) | FP16 U8B128 模板实例化 |
| `sgl-kernel/csrc/moe/marlin_moe_wna16/generate_kernels.py` | `2ef83105bea3` (2026-04-28) | Kernel 实例化代码生成脚本 |
| `sgl-kernel/csrc/gemm/marlin/marlin.cuh` | `3de09aadbc03` (2025-12-01) | CUDA 基础原语(cp_async 等) |
| `sgl-kernel/csrc/gemm/marlin/marlin_dtypes.cuh` | `3de09aadbc03` (2025-12-01) | 数据类型 traits (FragA/FragB/FragC) |
| `sgl-kernel/csrc/gemm/marlin/dequant.h` | `3de09aadbc03` (2025-12-01) | 反量化例程 (INT4/INT8/FP4/FP8) |
| `sgl-kernel/include/scalar_type.hpp` | `3de09aadbc03` (2025-12-01) | 量化类型定义 (ScalarType 类) |

### 关键 PR

- `Deepseek_v4 support w4(mxfp4)a16 on hopper (#23686)` — 2026-04-28，添加 MXFP4 MoE 支持
- `Add new moe wna16 marlin gemm (#14122)` — 2025-12-01，marlin MoE 基础框架

---

## 2. 迁移修改清单

以下是相比原始 sglang 代码所做的全部修改。**核心 kernel 算法逻辑未做任何修改**，所有改动均为接口适配。

### 2.1 命名空间替换

| 修改项 | 原始代码 | 迁移后 |
|--------|---------|--------|
| 量化类型命名空间 | `namespace sglang { ... }` | `namespace marlin_moe_wna16 { ... }` |
| Kernel 内部命名空间 | `MARLIN_NAMESPACE_NAME = marlin_moe_wna16` (不变) | `marlin_moe_wna16` (不变) |
| 类型引用 | `sglang::kU4`, `sglang::kFE2M1f` 等 | `marlin_moe_wna16::kU4`, `marlin_moe_wna16::kFE2M1f` 等 |
| 对外接口命名空间 | 无（torch注册） | `llm_kernels::nvidia::marlin_moe_wna16` |

涉及文件：所有文件（约 110+ 处 `sglang::` → `marlin_moe_wna16::` 替换）

### 2.2 依赖去除

| 原始依赖 | 位置 | 处理方式 |
|---------|------|---------|
| `#include <ATen/cuda/CUDAContext.h>` | `marlin.cuh` | 删除 |
| `#include <c10/cuda/CUDAGuard.h>` | `marlin.cuh` | 删除 |
| `#include <torch/all.h>` | `marlin.cuh` | 删除 |
| `#include <torch/library.h>` | `scalar_type.hpp` | 替换为 `<stdexcept>`, `<tuple>` 等标准库头 |
| `#include <iostream>` | `marlin.cuh` | 删除 |
| `torch::Tensor` 接口 | `ops.cu` (924-1293行) | 完全重写为 raw CUDA 指针接口 |
| `at::cuda::getCurrentCUDAStream()` | `ops.cu` | 由调用方传入 `cudaStream_t` |
| `at::cuda::OptionalCUDAGuard` | `ops.cu` | 删除，由调用方管理 device |
| `TORCH_CHECK(cond, msg...)` | `scalar_type.hpp`, `ops.cu` | 见下方 2.3 节 |

### 2.3 错误检查宏替换

| 位置 | 原始代码 | 迁移后 |
|------|---------|--------|
| `scalar_type.hpp` (10处) | `TORCH_CHECK(cond, ...)` | `MARLIN_CHECK(cond)` (本文件内定义的轻量宏，抛出 `std::runtime_error`) |
| `ops.cu` (~65处) | `TORCH_CHECK(cond, arg1, arg2, ...)` (stream风格) | `KLLM_KERNEL_CHECK_WITH_INFO(cond, "format %d", val)` (printf风格) |

### 2.4 类型命名冲突解决

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `ScalarType` 名称冲突 | 原始代码中 `scalar_type.hpp` 的 `ScalarType` 类在 `sglang::` 命名空间，`marlin_dtypes.cuh` 的 `ScalarType<T>` 模板类在 `marlin_moe_wna16::` 命名空间，互不冲突。迁移后两者都在 `marlin_moe_wna16::` 导致冲突 | 将 `marlin_dtypes.cuh` 中的 `template<T> class ScalarType` 重命名为 `template<T> class FragType` |

涉及文件：`marlin_dtypes.cuh` (3处), `marlin_template.h` (34处)

### 2.5 include 路径修改

| 文件 | 原始 include | 迁移后 |
|------|-------------|--------|
| `kernel.h` | `"gemm/marlin/marlin.cuh"`, `"gemm/marlin/marlin_dtypes.cuh"`, `"scalar_type.hpp"` | `"marlin.cuh"`, `"marlin_dtypes.cuh"`, `"scalar_type.hpp"` (本地) |
| `marlin_template.h` | `"gemm/marlin/dequant.h"`, `"gemm/marlin/marlin.cuh"`, `"gemm/marlin/marlin_dtypes.cuh"`, `"scalar_type.hpp"` | `"dequant.h"`, `"marlin.cuh"`, `"marlin_dtypes.cuh"`, `"scalar_type.hpp"` (本地) |

### 2.6 ops.cu 入口函数重写

原始 sglang 的入口函数：
```cpp
torch::Tensor moe_wna16_marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> const& c_or_none,
    torch::Tensor& b_q_weight, ...) { ... }
```

迁移后的入口函数（raw CUDA 指针）：
```cpp
namespace llm_kernels::nvidia::marlin_moe_wna16 {
void moe_wna16_marlin_gemm(
    void* a, void* c, void* c_tmp,
    const void* b_q_weight, ...,
    cudaStream_t stream);
}
```

关键变化：
- 所有 `torch::Tensor` 参数改为 `void*` 指针 + 显式 shape 参数
- 所有 buffer 分配（`torch::empty`）移至调用方责任
- `std::optional<torch::Tensor>` 参数改为可空指针（`nullptr` 表示无）
- `at::ScalarType` 改为 int 参数（`compute_dtype`: 0=fp16, 1=bf16）
- scale 类型推断改为 `scale_dtype` int 参数（0=fp16, 1=bf16, 2=fp8_e4m3fn, 3=fp8_e8m0fnu）
- `infer_scale_type` 函数重写：原始取 `b_scales.scalar_type()`，迁移后由调用方传入 `scale_dtype`

### 2.7 新增文件

| 文件 | 用途 |
|------|------|
| `ops.h` | 对外接口头文件声明 |
| `marlin_moe_wna16_wrapper.h` | 辅助工具函数声明（workspace/buffer 大小计算） |
| `marlin_moe_wna16_wrapper.cu` | 辅助工具函数实现 |
| `marlin_moe_wna16_test.cu` | 单元测试（MXFP4 W4A16 路径） |
| `CMakeLists.txt` | 构建配置 |

---

## 3. 接口说明

### 3.1 核心计算接口 (`ops.h`)

```cpp
#include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/ops.h"

namespace llm_kernels::nvidia::marlin_moe_wna16 {

void moe_wna16_marlin_gemm(
    void* a,                          // 输入激活 [size_m, size_k], fp16 或 bf16
    void* c,                          // 输出 [size_m * top_k, size_n], 同 a 类型 (预分配)
    void* c_tmp,                      // FP32 临时缓冲 (use_fp32_reduce 时需要, 否则可传最小buffer)
    const void* b_q_weight,           // 量化权重 [num_experts, size_k/16, size_n*16/pack_factor]
    const void* b_bias,               // 可选 bias [num_experts, size_n], 或 nullptr
    const void* b_scales,             // 量化 scales [num_experts, num_groups, size_n]
    const void* global_scale,         // NVFP4 的 global scale, 或 nullptr
    const void* b_zeros,              // 可选 zero points, 或 nullptr
    const void* g_idx,                // act_order 的 group indices, 或 nullptr
    const void* perm,                 // act_order 的排列, 或 nullptr
    void* a_tmp,                      // act_order 临时缓冲 [size_m*top_k, size_k], 或 nullptr
    const int32_t* sorted_token_ids,  // MoE 排序后的 token IDs
    const int32_t* expert_ids,        // MoE 每个 block 对应的 expert ID
    const int32_t* num_tokens_past_padded,  // device 上的 int32 指针
    const float* topk_weights,        // Top-k 权重 [size_m, top_k]
    int* workspace,                   // 锁 workspace (预分配, 见 get_workspace_size)
    int64_t moe_block_size,           // MoE block 大小: 8, 16, 32, 48, 64
    int64_t top_k,                    // 每个 token 选择的 expert 数
    bool mul_topk_weights,            // 是否将 topk 权重乘入输出
    bool is_ep,                       // Expert Parallelism 模式
    ScalarTypeId b_q_type_id,         // 权重量化类型 ID (如 kFE2M1f.id())
    int scale_dtype,                  // Scale 数据格式: 0=fp16, 1=bf16, 2=fp8_e4m3fn, 3=fp8_e8m0fnu
    int64_t size_m,                   // M 维度 (token 数)
    int64_t size_n,                   // N 维度 (每个 expert 的输出特征数)
    int64_t size_k,                   // K 维度 (输入特征数)
    int64_t num_groups,               // Scale 分组数
    int64_t sorted_token_ids_len,     // sorted_token_ids 长度
    int64_t g_idx_len,                // g_idx 长度 (0 = 无 act_order)
    int64_t perm_len,                 // perm 长度 (0 = 无 act_order)
    int64_t b_zeros_len,              // b_zeros 最后一维长度 (0 = 无 zero points)
    bool is_k_full,                   // K 维度是否完整 (act_order 相关)
    bool use_atomic_add,              // 是否使用 atomic add 做规约
    bool use_fp32_reduce,             // 是否使用 fp32 全局规约
    bool is_zp_float,                 // zero points 是否为 float 类型
    int compute_dtype,                // 计算类型: 0=fp16, 1=bf16
    cudaStream_t stream);             // CUDA stream
}
```

### 3.2 辅助工具接口 (`marlin_moe_wna16_wrapper.h`)

```cpp
#include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/marlin_moe_wna16_wrapper.h"

namespace llm_kernels::nvidia::marlin_moe_wna16 {

// 常量
static constexpr int kTileSize = 16;
static constexpr int kMinThreadN = 64;
static constexpr int kMaxThreadN = 256;

// 计算 workspace 大小 (int 元素数)
int64_t get_workspace_size(int64_t size_n, int64_t sorted_token_ids_len, int64_t moe_block_size);

// 计算 c_tmp 大小 (float 元素数), 仅 use_fp32_reduce=true && use_atomic_add=false 时需要
int64_t get_c_tmp_size(int64_t size_n, int64_t sorted_token_ids_len,
                       int64_t moe_block_size, ScalarTypeId b_q_type_id);

// 计算 a_tmp 大小 (计算类型元素数), 仅 has_act_order=true 时需要
int64_t get_a_tmp_size(int64_t size_m, int64_t size_k, int64_t top_k);
}
```

### 3.3 DSv4 MXFP4 推理的典型调用参数

```cpp
// DeepSeek-V4 Flash FP4: W4(MXFP4)A16, BF16 计算
moe_wna16_marlin_gemm(
    a, c, c_tmp, b_q_weight,
    /*b_bias=*/nullptr, b_scales,
    /*global_scale=*/nullptr,   // MXFP4 不需要 global_scale (那是 NVFP4 用的)
    /*b_zeros=*/nullptr,
    /*g_idx=*/nullptr, /*perm=*/nullptr, /*a_tmp=*/nullptr,
    sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights, workspace,
    moe_block_size,             // 推荐 16 或 32
    top_k,                      // DSv4: 8 (routed experts)
    mul_topk_weights,           // true
    is_ep,                      // 取决于部署配置
    kFE2M1f.id(),               // 权重类型: float4_e2m1f (MXFP4)
    /*scale_dtype=*/3,          // fp8_e8m0fnu
    size_m, size_n, size_k,
    /*num_groups=*/size_k / 32, // MXFP4 group_size=32
    sorted_token_ids_len,
    /*g_idx_len=*/0, /*perm_len=*/0, /*b_zeros_len=*/0,
    /*is_k_full=*/true,
    /*use_atomic_add=*/false,
    /*use_fp32_reduce=*/false,
    /*is_zp_float=*/false,
    /*compute_dtype=*/1,        // bf16
    stream);
```

### 3.4 量化类型常量 (`scalar_type.hpp`)

命名空间 `::marlin_moe_wna16` 中定义了以下常量：

| 常量 | 含义 | DSv4 用途 |
|------|------|----------|
| `kFE2M1f` | float4_e2m1f (MXFP4 权重) | **DSv4 权重格式** |
| `kFE8M0fnu` | float8_e8m0fnu (MXFP4 scale) | **DSv4 scale 格式** |
| `kFE4M3fn` | float8_e4m3fn (NVFP4 scale) | NVFP4 路径 |
| `kU4B8` | uint4 with bias 8 (GPTQ INT4) | INT4 量化 |
| `kU8B128` | uint8 with bias 128 (GPTQ INT8) | INT8 量化 |
| `kU4` | uint4 (带 zero point) | AWQ INT4 |
| `kFloat16` / `kBFloat16` | fp16 / bf16 scale 类型 | INT4/INT8 路径 |

---

## 4. 文件结构

```
marlin_moe_wna16/
├── CMakeLists.txt                    # 构建配置
├── ops.h                             # 对外接口声明 ★ 调用入口
├── ops.cu                            # 算子调度实现
├── marlin_moe_wna16_wrapper.h        # 辅助工具函数声明
├── marlin_moe_wna16_wrapper.cu       # 辅助工具函数实现
├── scalar_type.hpp                   # 量化类型定义
├── marlin.cuh                        # CUDA 基础原语
├── marlin_dtypes.cuh                 # 数据类型 traits (FragType<T>)
├── dequant.h                         # 反量化实现
├── kernel.h                          # Kernel 前向声明
├── kernel_marlin.cuh                 # Kernel 实例化汇总 include
├── marlin_template.h                 # Kernel 核心实现 (1914行, 未修改算法)
├── kernel_bf16_ku4.cuh               # 模板实例化 (auto-generated)
├── kernel_bf16_ku4b8.cuh
├── kernel_bf16_ku8b128.cuh
├── kernel_fp16_ku4.cuh
├── kernel_fp16_ku4b8.cuh
├── kernel_fp16_ku8b128.cuh
├── generate_kernels.py               # 实例化代码生成脚本
├── marlin_moe_wna16_test.cu          # 单元测试
└── README.md                         # 本文档
```
