# Kunlun kernel wrapper 模板类型命名约定

本目录下的各 `*_kernel_wrapper.{h,cpp}` 使用一套**统一的模板类型参数命名约定**。
由于这些 wrapper 没有公共头文件，此文档作为约定的唯一权威来源；各 wrapper 顶部只放
"本文件用到哪些 + 特有语义"的指针，通用含义一律以本表为准。

修改任何模板签名、或新增类型参数时，请同步更新本表。

## 通用词汇表

| 名称 | 含义 | 语义说明 | 出现的 wrapper |
|------|------|----------|----------------|
| `T` | 主计算 / 激活类型 | fp16 / bf16 / fp32，kernel 的核心元素类型；输入输出同类型时统一用 `T` | 全部 |
| `TW` | 权重类型 (weight) | GEMM/FC 权重、rmsnorm 权重；常默认 `= float` | basic, attention |
| `TGEMM` | GEMM 内部量化 / 中间累加类型 | FC 内部量化路径，默认 `= int16_t` | basic |
| `TID` | 索引 / ID 类型 | token id、position、topk_index、gather/scatter 下标；int32/int64 | basic, attention, moe |
| `TX` / `TY` | 输入 / 输出张量类型 | in/out 不同类型时使用（如 cast、量化 GEMM） | basic |
| `TACC` | 累加 / 标量类型 (accumulator) | 如 arange 的 begin/step，默认 `= int64_t` | basic |
| `TCACHE` | KV cache 存储类型 | paged / flash attention 的 KV cache | attention |
| `TR` | rotary_pos_emb 类型 (rotary) | RoPE 旋转位置编码张量 | attention |
| `TS` | score / scale 类型 | MoE softmax-topk 输出 score | moe |
| `TI` / `TO` | input / output 类型（内部实现） | MoE softmax-topk 的 `.cpp` 内部实现 | moe (.cpp) |
| `TWEIGHT` | topk 权重类型（内部固定 fp16） | MoE 融合内部 `using TWEIGHT = xdnn_fp16` | moe (.cpp) |
| `TOutput` | 输出类型（内部实现） | batched matmul `.cpp` 内部 | batched_matmul (.cpp) |
| `Tr` / `Traits` | 局部 / traits 辅助类型（内部实现） | 供 xdnn/xblas 内部特化用 | attention, basic, moe, batched_matmul (.cpp) |

## 约定要点

- **`T` 表示"输入输出同类型"**。当且仅当输入、输出类型不同时才拆成 `TX`/`TY`（或
  `TI`/`TO`）。因此绝大多数 kernel 只需要一个 `T`。
- **`TD`（destination/data 类型）在本代码库中不存在**。参考实现区分 `TD` 与
  `TWEIGHT`，但 KsanaLLM 中输入输出类型相同，`TD` 被并入 `T`/`TWEIGHT` 而省略。
  `moe_kernel_wrapper.cpp` 里 `TD` 仅出现在对齐注释中，**不是真实模板参数**。
- 通用类型（`T`、`TID`）在各 wrapper 中含义完全一致，各文件不再重复解释，只在本表定义一次。
