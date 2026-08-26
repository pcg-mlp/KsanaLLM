---
spec_version: "1"
spec_type: change
id: "1070131059136538942"
status: draft
subdomain: models
cross_subdomains: [cache_manager]
complexity: XL
work_type: feature
---

> TAPD：[【Kimi-K3 模型适配】](https://tapd.woa.com/tapd_fe/70131059/story/detail/1070131059136538942)
> 反写说明：本澄清由已合入提交 `50261617bb247e06b502331826709ef0e7136144` 回溯，描述**已落地行为**，不是待开发需求。该提交跨 `cache_manager`（新增 KDA 状态布局）与 `models`（新增 KimiK3 attention_meta 通路 + 模型/配置/权重改造）两子域，主子域 `models`。按「影响面 ≥6 模块（cache_layout / kimi_k3 模型 / attention_meta / configure / kernels / layers / executor 共 32 文件）、跨 2 子域、修改内部 kernel 接口（cu_seqlens int32→uint64 ABI）」就高定为 XL。`work_type=feature` 按提交 `[feat] Add KDA cache layout + attention meta` 落地。

### A. 工作类型（work_type）【必填，先于目标】
`feature` — 功能实现（新增能力/配置/对外行为）

次要动机：状态布局占位转真实实现含 refactor 成分；recurrent_state dtype 双路径含 perf 成分。主类型仍是 feature。

### B. 目标（goal）

问题 1：Kimi-K3 是 KDA（线性注意力）层与 MLA 全注意力层混合的架构。KDA 层的运行时状态（`conv_state` + `recurrent_state`）此前在缓存布局里只有占位——`MambaBlockCacheLayout` 的 `block_cache_size_` 恒为 0，注释标「linear-attn state 尺寸待 linear-attn 布局单独实现（待重构）」，没有逐层 conv/recurrent 偏移与真实块大小，KDA 层拿不到正确显存池。

问题 2：Kimi-K3 原先复用 `DeepSeekV3AttentionMetaCreator`，但 KDA 路径需要额外元数据——per-request 的 mamba 状态索引、causal-conv1d 的 `has_initial_state` / `max_extend_len`、chunked-prefill 的 `cu_seqlens` / `chunk_indices` / `chunk_offsets`，DSV3 creator 不提供。

**为 Kimi-K3 KDA 混合注意力建立状态缓存布局（`KDAStateLayout`：block 内逐层 `[conv_state | recurrent_state]`，K/V 共偏移，conv 恒 bf16、recurrent 默认 fp32 可覆盖）与 KimiK3 专属 attention_meta 通路（继承 DSV3 全部 MLA/DSA/Indexer 步骤并叠加 KDA 准备），让 KDA 层拿到正确的逐层显存偏移与 per-request slot 索引/分块张量。**

### C. 非目标（non_goals）

- 不含 KDA 层 forward 实际算子编排——`KimiK3DecoderLayer::CommonAttn` 在 `mla_` 为空（KDA 层）时直接 `return Status()`，KDA kernel 调用接入是后续 MR
- 不含多模态视觉塔——`KimiK3VisionConfig` / `PrepareKimiK3VisionAttributes` / `has_vision` / `media_placeholder_token_id` / `vision_tower.` / `mm_projector.` 权重 skip 已移除，视觉能力不在本提交
- 不改 DSV3 / DSV4 既有 attention_meta 行为——仅把 `InitDeepSeekV3SharedMeta` 提到 protected 供复用、删一行过期注释
- 不改 KDA kernel 数值实现，仅改 `cu_seqlens` 入参类型 ABI（int32 → uint64）
- 不暴露 recurrent_state dtype 运行时动态切换开关——构造期固定，默认 fp32

### D. 影响面（impact_surface）

- `csrc/cache_manager/cache_layout/kda_cache_layout.h`（新增 `KDAStateLayout` / `KDALayerCacheLayout`）
- `csrc/cache_manager/cache_layout/mamba_cache_layout.h`（`MambaModelCacheLayout` 按 `kda_layers` 非空注册 `KDAStateLayout`；kv-cache 按 `use_mla` 选 `MlaBlockCacheLayout`；ctor 改取 `model_config`）
- `csrc/cache_manager/cache_layout/cache_layout_factory.h`（`MambaModelCacheLayout` 传 `model_config`）
- `csrc/configure/schedule_config_parser.cpp`（mamba-state sizing 去 `dynamic_pointer_cast<MambaBlockCacheLayout>` 强校验，改通用 `GetBlockCacheSize`）
- `csrc/models/kimi_k3/attention_meta/{kimi_k3_attention_meta, kimi_k3_attention_meta_builder, kimi_k3_attention_meta_creator}.{h,cpp}`（新增四件套 + 单测）
- `csrc/models/deepseek_v3/attention_meta/deepseek_v3_attention_meta_creator.h`（`InitDeepSeekV3SharedMeta` 提 protected）
- `csrc/models/deepseek_v4/attention_meta/deepseek_v4_attention_meta.cpp`（删过期注释）
- `csrc/models/kimi_k3/kimi_k3_model.{h,cpp}`（creator→`KimiK3AttentionMetaCreator`；`RegisterInflightResources` 注册 `mamba_slot`；layer ctor 去 `is_moe` 形参；`CommonAttn` 派发）
- `csrc/models/kimi_k3/kimi_k3_config.h` / `kimi_k3_config_parser.cpp`（config 简化 + 新增 parser 单测）
- `csrc/configure/model_config.h` / `model_config_parser.cpp`（`KimiLinearAttnConfig` 简化、去 vision 字段）
- `csrc/models/kimi_k3/kimi_k3_weight_loader.{h,cpp}`（去 vision skip；`kda_num_heads ← head_num`）
- `csrc/kernels/nvidia/kda_kernel_wrapper.h` / `kda_kernel_wrapper_test.cpp` / `csrc/layers/nvidia/kda_attention_layer.cpp` / `csrc/kernels/nvidia/kda/generate_kda.py`（`cu_seqlens` int32→uint64 ABI）
- `csrc/executor/model_runner.cpp`（`GetHiddenShape` attn_res `per_token_width *=` 修正）
- `csrc/models/CMakeLists.txt`（注册两个新单测）

本次不涉及：PD / 调度、其他模型、serving 入口。

### E. 行为不变量（invariants）

- 非 KDA（full-attn 混合）层在 `KDAStateLayout` 中 `cache_size=0`、conv/recurrent offset=0，且访问点由 `model_config.kimi_linear_config.IsKdaLayer(layer_idx)` 守卫，永不读——违反即越界
- KDA 层 K/V 共偏移：`GetKvCacheLayerOffsets(*, "conv"|"recurrent").k_offset == .v_offset`；`KimiK3SharedAttentionMeta::Init` 对 conv / recurrent 各做一次 `k_offset == v_offset` 断言，不等即 `throw std::runtime_error`
- `conv_state` 恒 bf16，`recurrent_state` 默认 fp32；二者均不受 runtime `kv_cache_dtype` 影响（KV cache dtype 改 fp32 不改 KDA block size）
- `recurrent_state` dtype 仅构造期可覆盖（`KDAStateLayout` ctor 第 4 参），bf16 时 recurrent 体积减半、conv 不变
- 非线性注意力模型（`kda_layers` 为空）走原 `MambaBlockCacheLayout` 占位（block size 0），`KDAStateLayout` 不注册
- `MambaModelCacheLayout` 无参重载（executor 路径）kv-cache 占位随 `use_mla` 选 Mla / Mha，须与 `RegisterLayouts` 一致——避免 `Initialize` 前 `dynamic_pointer_cast<MlaBlockCacheLayout>` 误判 nullptr
- `cu_seqlens` 全链路统一 `uint64_t`（wrapper 三 params + layer + test + python gen）；任一侧 int32 即 ABI 错位
- `mamba_state_buffer` 形状固定 `[slot_pool_capacity, decode_token_num, state_block_size]`，`state_indices[i] = slot_id * decode_token_num`（`shape[1]`），`slot_pool_capacity = max_batch_size * max_pp_batch_num`
- `slot_id` 来自 `CompressorSlotResource::Get(req_id)`；`slot_resource_` 未 wire（`WireInflightResources` 未调）即 `throw std::runtime_error`
- `use_linear_attn` 与 `use_dsv4_attn` / `use_dsa` 互斥（`CacheLayoutFactory` 有 `KLLM_CHECK`）；Kimi-K3 是 `use_mla` + `use_linear_attn` 混合，kv-cache 走 `MlaBlockCacheLayout`

### F. 功能分支（functional_branches）

- KDA 层 vs full-attn 层派发：layer ctor 里 `IsKdaLayer(layer_idx)` 为真时不构造 `mla_`（留空），`CommonAttn` 据 `mla_` 非空派发 MLA，否则（KDA 分支）直接 `return Status()`（stub）
- dense vs MoE 派发：移除 `is_moe` 形参，改由 `layer_idx >= first_k_dense_replace` 推导；`<` 为 dense（`mlp_`），`>=` 为 MoE（`moe_` + `shared_mlp_` + `expert_gate_`）
- recurrent_state dtype fp32 / bf16 双路径：fp32 为 delta-rule 参考（默认），bf16 状态池减半；每种 storage dtype 是独立 AOT artifact（由 SSM specialization key 选）
- chunked-prefill 边界：`extend_len == 0` 的请求产出 0 chunk（`cu_seqlens` 出现重复项，仅非零请求贡献 `chunk_indices`）；空 batch 时 `num_chunks=0`、`chunk_indices` 为空
- 无参重载路径（executor）：未 `Initialize` 前 `GetBlockCacheSize` 返回 0，仅注册存在性
- conv1d initial_state：`prefix_cache_len > 0` 的请求 `has_initial_state=1`（带初始状态续算）

### G. 建议测试点（suggested_tests）

- id: T01
  scenario: `kda_cache_layout_test`——`KDAStateLayout` 几何计算（block size 542720 等）、TP 分片头数减半、dtype 独立于 kv_cache_dtype、inactive 层零偏移、非法 layer_index / kv_cache_type `throw`、`CacheLayoutFactory` 按 `kda_layers` 非空接 `KDAStateLayout`（`mamba-state`）并返回 `{"conv","recurrent"}` 类型表
  expect: 全部断言命中；`GetActiveKvCacheTypes()=={"conv","recurrent"}`；非法入参 `throw std::runtime_error`
  covers: K/V 共偏移不变量、inactive 层守卫、dtype 不变量、非线性模型不注册 KDA 分支
  check_type: unit
- id: T02
  scenario: `kimi_k3_attention_meta_builder_test`——`SharedInit` 张量与偏移填充（`state_indices` DEVICE INT32 `[max_batch]`、`layer_mamba_kv_cache_ptr_tensor` HOST INT64 `[1+2N]`、`mamba_state_buffer` 形状）、conv/recurrent `k!=v` 各 `throw`；Flash `PrepareStateIndices`（含 `decode_token_num` 乘子）/ `PrepareConv1D`（混 `prefix_cache_len`）/ `PrepareChunkKDA`（64→1 chunk、100→2 chunks、0 extend 不贡献、空 batch `num_chunks=0`）；Paged `PrepareStateIndices`；`slot_resource_` null `throw`
  expect: host 向量与 device tensor shape 与公式一致；`null slot_resource` throw
  covers: chunked-prefill 边界分支、null slot_resource 失败路径、k!=v 失败路径
  check_type: unit
- id: T03
  scenario: `kimi_k3_config_parser_test`——`linear_attn_config` 解析（`head_dim` / `short_conv_kernel_size` / `use_full_rank_gate` / `kda_layers` 1-based 集合）、`head_dim == v_head_dim` 与 `num_heads == num_attention_heads` 校验
  expect: 字段映射正确；不满足断言 `KLLM_CHECK` 失败
  covers: config 简化分支
  check_type: unit
- id: T04
  scenario: `kda_kernel_wrapper_test`——`cu_seqlens` 改 `uint64_t` 后 chunk / recurrent / conv1d 三路径真实 launch + sync + `cudaGetLastError()==cudaSuccess`
  expect: 全部 launch 成功；参数槽类型与 C++ 调用侧一致
  covers: cu_seqlens ABI 不变量
  check_type: unit
  carrier: sm90 实机执行（KDA 仅 nvidia 后端，`ENABLE_CUDA`）
- id: T05
  scenario: 端到端 Kimi-K3 混合 forward——MLA full-attn 层经 `CommonAttn` 派发仍正确，KDA 层 stub 不 crash（KDA kernel 调用未接入，输出对 KDA 层为空）
  expect: MLA 层 forward 行为与改造前一致；KDA 层不产生运行期错误
  covers: KDA/full-attn 派发分支、dense/MoE 派发分支
  check_type: manual
  note: KDA 层端到端正确性挂后续 KDA kernel 接入 MR；本提交 dispatch 为 stub

### H. 资源/硬件约束（resource_constraints）

- `recurrent_state` 默认 fp32（delta-rule 参考），显存敏感可切 bf16（按 SSM specialization key 选独立 AOT artifact，体积减半）
- `mamba_state_buffer` 为 DEVICE 连续池：`[max_batch_size * max_pp_batch_num, decode_token_num, state_block_size]`
- KDA kernel 仅 nvidia 后端（`ENABLE_CUDA`），非 CUDA 编译期 `kKdaChunkSize=64` 占位
- 状态池 per-request slot（复用 DSV4 `CompressorSlotResource` "mamba_slot"），非 prefix-cache 兼容（SplitFuse OK）——沿用 DSV4 slot 语义

### I. 回退方案（rollback）

- revert 本 commit 即完全回退：新文件整体删除，接线点恢复 `MambaBlockCacheLayout` 占位 + `DeepSeekV3AttentionMetaCreator` + `is_moe` 形参
- 无存量调用方依赖 `KDAStateLayout`（KDA forward dispatch 为 stub，本提交未引入运行期 KDA 推理路径）
- `cu_seqlens` ABI 回退须 wrapper / layer / test / python gen 四处同改

### J. 风险备注（risk_notes）

- KDA 层 `CommonAttn` 为空实现（`mla_` null → `return Status()`）：本提交只搭「布局 + 元数据」通路，KDA kernel 实际调用未接入，端到端 KDA 推理在本提交**不可用**——若误以为可用会得到静默零输出（非报错）
- 复用 DSV4 `CompressorSlotResource` 作为 mamba slot 池：DSV4 压缩器 slot 语义与 KDA mamba 状态 slot 是否完全一致（分配 / 回收 / DP 对齐）依赖 DSV4 实现，`ai_docs` 未覆盖 KDA 复用场景（见 `0-context-snapshot.md` §C）
- `cu_seqlens` 改 uint64 是跨语言 ABI（python `generate_kda.py` ↔ C++ wrapper ↔ layer ↔ test）：任一侧漏改即内存错位，且 AOT cubin 按 SSM key 选，dtype / key 错配在加载期才 `throw`
- 删除 Kimi-K3 多模态视觉塔是能力回退（`vision_tower.` / `mm_projector.` 权重被 `FilterWeightNames` 跳过且 config 字段移除）：存量视觉权重会静默忽略，后续多模态 MR 需重新接入
- `use_linear_attn` 与 `use_mla` / `use_dsa` / `use_dsv4_attn` 互斥（factory 有 `KLLM_CHECK`）；Kimi-K3 是 `use_mla` + `use_linear_attn` 混合，kv-cache 走 `MlaBlockCacheLayout`，mamba-state 走 `KDAStateLayout`

### K. 需要人确认的点（待确认清单）

- [?] TAPD 标题【Kimi-K3 模型适配】范围大于本提交（适配含权重 / forward / 多模态等），本澄清只覆盖 cache layout + attention meta 子集——需确认该 TAPD 是否拆子单或本提交即该单的一个阶段；不影响本澄清自洽性
- [?] KDA 层 forward stub（`CommonAttn` `return Status()`）是否本提交有意分阶段 vs 漏接入——影响 §B 目标表述（"通路" vs "完整推理"），当前按"通路"落地
- [?] 复用 `CompressorSlotResource` 是否与 DSV4 压缩器 slot 池共享 / 隔离——影响 §H 资源约束
- [AUTO?] T05 端到端验证挂后续 KDA kernel 接入 MR（本提交 KDA dispatch 为 stub，单测为本地验证上限）
