---
spec_version: "1"
spec_type: change
id: "hy3-nv-bf16-fp8-adaptation"
status: draft
subdomain: models
cross_subdomains: []
complexity: L
work_type: feature
---

### A. 工作类型（work_type）【必填，先于目标】
`feature` — 功能实现（新增模型适配能力 + 配套修复；对外表现为可加载并推理 hy3 bf16/fp8 checkpoint）

### B. 目标（goal）
现状：KsanaLLM 的 Nv（CUDA）平台模型列表中没有 Hy3（HunYuan V3，`model_type=hy_v3`）的适配。Hy3 的结构是 Qwen3-MoE 风格的 attention（GQA + pre-RoPE QK-Norm、无 QKV bias、全维度 rotary）与 DeepSeek-V3 风格的 MoE（sigmoid 打分 + aux-loss-free expert bias + shared expert + first_k_dense）的组合，并带 MTP（NextN）草稿层；其 config 与权重命名与现有模型均不完全一致（`rope_theta` 嵌套在 `rope_parameters` 中、`config.json` 不声明 `torch_dtype`、权重名存在 `router.gate`/`expert_bias`/`shared_mlp` 等差异），transformers 也没有 hy_v3 的 config 类，现有通用解析路径无法直接加载。

同时，打开 Hy3 的 MTP 后暴露了两个 FlashAttn 存量缺陷：(1) flash prefill 的区间切分按"decode 请求数"而非"decode token 数"扣除 qkv 缓冲，MTP verify 阶段每个 decode 请求携带 1+draft_num 个 token，混批时 prefill 区间被算长，flash 内 per-token kernel 越界写 KV；(2) GQA paged decode 只取 `page_metas.back()` 一个桶，而上游 `BuildAttentionMetas` 已按 q_seq_len 分桶，MTP 打开后 q_seq_len=2 与 q_seq_len=1 的 decode 请求混批，其余桶被静默丢弃，表现为"混批乱码"（触发条件是混批而非并发）。

目标：
1. **hy3 适配**：新增 `csrc/models/hy_v3/` 模型实现（bf16 与 fp8 两种 checkpoint），复用为主——attention 段参数照 Qwen3-MoE、MoE 段照 DeepSeek-V3、MTP 走上游 `DraftModelInterface`；config 差异通过专用 config parser 收敛，权重命名差异通过 `python/weight_map/hy_v3_weight_map.json` 正则收敛；除注册接线的最小 shim 外不改任何已有模型的行为。
2. **FlashAttn 修复**：flash prefill 区间改用 decode token 数（`dp_decode_tokens`）扣除，字段改名 `decode_token_num` 防复发；`PagedAttention::Forward` 改为遍历所有 q_seq_len 桶逐桶调 kernel（与 MLA 路径同款模式），单桶路径保持字节级一致；ACL/Kunlun 平台遇到多桶时显式返回 `RET_NOT_IMPLEMENTED`，拒绝静默出错。
3. **fallback 路径算子修复**：`topkSigmoidKernel` 补上 e_score_correction_bias 选专家、top-k 权重归一化与 routed_scaling_factor；`Fp8MoeLayer` 的 routing bias 改从 `e_score_bias` 读取（原来误读 `eplb_map` 专家并行映射表）。

**在 Nv 平台上跑通 Hy3 bf16/fp8 模型推理（含 MTP），并修复 MTP 混批下的 attention 越界/丢桶与 sigmoid 路由 fallback 算子缺失，且不影响任何已有模型。**

### C. 非目标（non_goals）
- 不做 ACL（Ascend）/昆仑芯平台的 hy3 适配与多桶 paged attention 实现（这两个平台遇多桶显式返回 `RET_NOT_IMPLEMENTED`，拒绝而非误算）
- 不引入 vllm 式"非稳态 decode 并入 prefill 走 varlen attn"的批处理重构（本需求选择完善 paged attn 分桶启动机制，保持 batch 组织不变）
- 不做 w4afp8 等其他量化方式的 hy3 适配（仅 bf16 与 fp8 static/blockwise）
- 不做 hy3 以外的模型行为改动；不为其他 MoE 模型切换 sigmoid 路由路径
- 不做性能调优（kernel 融合、调度策略等），正确性优先
- 不做多机/EP（专家并行）下 hy3 的端到端验证；`eplb_map` 相关专家并行路径不在本次范围

### D. 影响面（impact_surface）
- `csrc/models/hy_v3/`（新增：`hy_v3_model.h/.cpp`、`hy_v3_config.h`、`hy_v3_config_parser.h/.cpp`、`hy_v3_weight_loader.h/.cpp`、`hy_v3_mtp_draft_model.h/.cpp`、`hy_v3_test.cpp`）
- `csrc/models/common_moe/moe_config.h`（新增 `PrepareHyV3Attributes`；不动既有 `Prepare*Attributes`）
- `csrc/models/base/model_arch.h/.cpp`、`csrc/runtime/model_instance.cpp`、`csrc/models/config_parser/model_config_parser_factory.cpp`、`csrc/models/weight_loader/model_weight_loader_factory.cpp`、`csrc/models/CMakeLists.txt`（注册接线 shim）
- `csrc/configure/model_config_parser.cpp`（`hy_v3` 分支接入 `PrepareHyV3Attributes`）及 `csrc/configure/model_config_parser_test.cpp`
- `csrc/modules/basic/flash_attention.cpp`、`csrc/layers/flash_attention_layer.h`、`csrc/layers/nvidia/flash_attention_layer.cpp`（`single_token_request_num` → `decode_token_num`，取 `dp_decode_tokens`）
- `csrc/modules/basic/paged_attention.cpp`、`csrc/layers/paged_attention_layer.h`、`csrc/layers/nvidia/paged_attention_layer.cpp`、`csrc/layers/attention_layer_test.cpp`（多桶分发；`PagedAttentionInputs` 新增默认关闭的 per-bucket 字段）
- `csrc/kernels/nvidia/mixture_of_experts/`（`moe_kernels.cu/.h`、`moe_wrapper.cu/.h`：topk sigmoid 补 bias/renorm/scaling；新增 `topk_sigmoid_test.cu`）、`csrc/kernels/nvidia/moe_kernel_wrapper.cpp`（仅 sigmoid 路径接线新参数）
- `csrc/layers/nvidia/fp8_moe_layer.cpp`、`csrc/layers/nvidia/moe_layer.cpp`（bias 从 `e_score_bias` 读取 + fp32 校验）
- `csrc/models/weight_method/fp8_e4m3_method.cpp`（fp8 权重处理适配）
- `python/ksana_llm/arg_utils.py`（transformers 无 hy_v3 config 类时回退直读 `config.json`）、`python/weight_map/hy_v3_weight_map.json`（新增权重名映射）
- `examples/hy3_bf16.yaml`、`examples/hy3_fp8.yaml`、`examples/hy3_fp8_moedefault.yaml`、`examples/ksana_llm_hy3.yaml`（新增示例配置）

本次**不涉及**：ACL/Kunlun 的 attention 多桶实现、`PdV2*`/调度器、真实 EP 部署链路。

### E. 行为不变量（invariants）
- MTP 关闭的所有既有模型：attention 前向结果与改前**字节级一致**（单桶路径不经过新增的 per-bucket 字段，`PagedAttentionInputs` 新增字段保持默认关闭语义）
- 非 hy_v3 模型的 config 解析、权重加载、MoE 前向行为完全不变（`PrepareHyV3Attributes`/weight_map/注册分支只对 `hy_v3` 生效；`moe_kernel_wrapper` 新增参数仅 sigmoid 路径消费，softmax 路径保持 `NT` 原行为）
- MTP 开启的混批场景（同一 batch 内 prefill 与 token 数不同的 decode 请求共存）：每个 decode 请求的输出 token 与单独运行（无混批）时一致，不出现乱码、不越界写 KV
- paged attention 各桶严格铺满 decode 区：`sum(bucket token) == dp_decode_tokens`，越界即 CHECK 失败而非静默错算
- sigmoid 路由：选专家结果 = sigmoid(logits) + e_score_correction_bias 的 top-k；输出权重经 top-k 归一化并乘 routed_scaling_factor，与 HuggingFace 参考实现一致
- fp8 MoE 路径读取的 routing bias 与 MoE 模块传入的 `e_score_correction_bias_weight_` 是同一个 tensor

### F. 功能分支（functional_branches）
- bf16 checkpoint（无 quant config）vs fp8 static quant vs fp8 blockwise quant 三条加载/前向路径
- MTP 开启（含 verify 混批）vs MTP 关闭（不加载 nextn 层、不注册 draft model）
- first_k_dense 前缀 dense 层（TwoLayeredFFN）vs 后续 MoE 层（sigmoid 路由 + shared expert）
- sigmoid 路由（默认）vs softmax 路由回退配置；有/无 shared expert；有/无 expert bias
- config.json 缺 `torch_dtype`（强制 BF16）vs 显式声明 dtype；`rope_theta` 嵌套 `rope_parameters` vs 顶层
- transformers 抛 `ValueError`（无 hy_v3 类）时 Python 前端回退直读 config.json
- 单桶 decode（无 MTP verify）vs 多桶 decode；ACL/Kunlun 多桶 → `RET_NOT_IMPLEMENTED`
- 不支持的量化方法（非 fp8）应在 config 解析阶段明确报错而非带病加载

### G. 建议测试点（suggested_tests）
- id: T01
  scenario: hy_v3 config parser 字段解析（架构字段、MoE gating 字段、rope_theta 嵌套/顶层、缺省默认值、tie_embeddings、first_k_dense、shared expert 有无）
  expect: 各字段映射正确，缺省配置落入声明的默认值（num_experts=192、topk=8、sigmoid、norm_topk_prob 等）
  covers: config 差异收敛不变量
  check_type: unit
- id: T02
  scenario: bf16 非量化与 fp8 static/blockwise 量化配置解析；不支持的量化方法
  expect: bf16 不触发量化；fp8 两档正确识别；非 fp8 量化方法明确报错
  covers: 三条量化路径分支
  check_type: unit
- id: T03
  scenario: 权重加载与过滤（weight_map 映射 `router.gate`/`expert_bias`/`shared_mlp`；MTP 关闭时丢弃 nextn 层与 shared head；PP 范围外层丢弃；main+MTP 层融合）
  expect: 每个权重族路由到正确处理方法，MTP on/off 下过滤行为符合预期
  covers: 权重命名差异收敛、MTP 分支
  check_type: unit
- id: T04
  scenario: 模型创建（decoder/MoE/MTP 层按 config 生成，`GetModelRunConfig` 反映 pre-RoPE QK-Norm）
  expect: 层数/类型与 config 一致，draft model 走 `DraftModelInterface`
  covers: 模型结构正确性
  check_type: unit
- id: T05
  scenario: topkSigmoidKernel 正确性（bias 参与选专家、top-k 归一化、routed scaling）
  expect: 与参考计算逐元素一致；关闭 bias/renorm 时退化为原行为
  covers: sigmoid 路由不变量
  check_type: unit
- id: T06
  scenario: 多桶 paged attention（构造 q_seq_len 不同的 decode 混批）
  expect: 每个桶都被 dispatch，各请求输出与单独运行一致；桶偏移不越过 `dp_decode_tokens`（越界 CHECK 生效）
  covers: 混批不乱码不变量
  check_type: unit
- id: T07
  scenario: 单桶回归（MTP off 既有模型 attention 前向）
  expect: 输出与改前一致（既有 `attention_layer_test` 用例全绿）
  covers: 字节级一致不变量
  check_type: unit
- id: T08
  scenario: Hy3 bf16/fp8 端到端推理冒烟（真机 checkpoint，含 MTP 开/关、prefix cache）
  expect: 可加载、可生成，输出文本合理；MTP 混批无乱码
  covers: 整体目标
  check_type: integration（真机人工/CI GPU 任务，非本仓库单测）

### H. 资源/硬件约束（resource_constraints）
- C++ 单测（config parser/weight loader/model 创建/topk sigmoid kernel）：需要 CUDA 编译环境；kernel 测试在无 CUDA 设备时跳过（`c8628584` 已加 skip 保护）
- T08 端到端冒烟需要 Nv GPU 真机与 hy3 bf16/fp8 checkpoint，不在本仓库自动化范围

### I. 回退方案（rollback）
hy3 适配为纯新增目录 + 注册 shim，回退 = 删除 `csrc/models/hy_v3/`、`python/weight_map/hy_v3_weight_map.json`、examples 并还原各工厂/`model_instance.cpp` 的注册行即可；FlashAttn 修复对单桶路径行为透明，如需回退直接 revert 对应提交，无数据迁移、无配置开关残留（per-bucket 字段默认关闭）。

### J. 风险备注（risk_notes）
- 多桶分发依赖"桶按 q_seq_len 降序且在 token 布局上连续"这一上游不变量（`ReorderInferRequests` 排序 + `AttentionMetaCreator::BuildAttentionMetas` 降序开桶）；已用 `bucket_offset + bucket_len <= dp_decode_tokens` 的 CHECK 做防越界兜底，但若上游排序/分桶语义变化，需同步复核本处
- `flash_attention.cpp` 取 `fs.shape[12]` 作为 `dp_decode_tokens`，依赖 flexible shape 布局定义，布局演进时需同步
- `arg_utils.py` 通过 `ValueError` 文本匹配 `"hy_v3"` 触发回退，transformers 升级后若报错文案变化需复核
- ACL/Kunlun 平台在 MTP verify 混批下会返回 `RET_NOT_IMPLEMENTED`（原为静默丢桶出错误结果）——行为从"错"变"拒绝"，属预期收紧，但相关平台若计划开 MTP 需另行适配
- `PrepareHyV3Attributes` 强制 `topk_method="fast"`（fp8 路径仅接线了 (fast, sigmoid) 组合），若后续支持其他组合需放开

### K. 需要人确认的点（待确认清单）
- [x] 子域归属 → **models**（hy3 适配 + attention/MoE 修复均落在 `csrc/models/`、`csrc/layers/`、`csrc/kernels/`，均属 modules.md 中 models 子域路径）
- [x] 混批乱码修复方案选型 → 完善 paged attn 分桶启动机制（而非 vllm 式批处理重构），batch 组织不变
- [x] ACL/Kunlun 多桶行为 → 显式 `RET_NOT_IMPLEMENTED`，不静默错算，不在本期实现
- [ ] [?] T08 真机端到端冒烟的验收标准（由谁、在哪个 checkpoint 上、MTP 开几档 draft_num）待需求方确认
