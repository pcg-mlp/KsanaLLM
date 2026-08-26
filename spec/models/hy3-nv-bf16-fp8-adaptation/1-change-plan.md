### A. 问题分析与方案总述

**问题分析（代码调研发现）**：

1. **Hy3 是"既有零件的新组合"，适配应以复用为主**：走查 `csrc/models/qwen3_moe/` 与 `csrc/models/deepseek_v3/` 后确认，Hy3 decoder layer 的 attention 段（GQA + pre-RoPE QK-Norm、无 QKV bias、全维度 rotary）与 Qwen3-MoE 参数面一致，MoE 段（sigmoid 打分、expert bias、shared expert、first_k_dense）与 DeepSeek-V3 一致，MTP 可走上游新的 `DraftModelInterface`（`csrc/models/base/draft_model_interface.h`）。差异集中在三处：(a) config 形态——`rope_theta` 嵌套在 `rope_parameters` 对象中、`config.json` 不声明 `torch_dtype`（通用 parser 会默认 fp16 导致 MatMul 权重类型不匹配）、字段名不同（`num_experts_per_tok`/`moe_intermediate_size`/`expert_hidden_dim` 等）；(b) 权重命名——`mlp.router.gate`、`mlp.expert_bias`、`mlp.shared_mlp.*`；(c) transformers 无 hy_v3 config 类，Python 前端 `AutoConfig.from_pretrained` 直接抛 `ValueError`。**方案**：(a) 新增 `PrepareHyV3Attributes`（`moe_config.h`）+ `hy_v3_config_parser` 收敛 config 差异，缺 `torch_dtype` 时强制 BF16；(b) 新增 `python/weight_map/hy_v3_weight_map.json` 用正则把权重名收敛到既有命名（`gate.weight`/`gate.e_score_correction_bias`/`shared_expert.*`），模型侧零特判；(c) `arg_utils.py` 捕获 `ValueError` 后回退直读 `config.json` dict，仅用于挑选 model_type 与 tokenizer。

2. **FlashAttn kv 越界：prefill 区间应按 token 数而非请求数扣除**：prefill 与 decode 共用 qkv 缓冲，切分时原来减的是"decode 请求数"（`single_token_request_num`），隐含"每 req 1 token"。MTP verify 阶段每 req 携带 1+draft_num 个 token，混批时 prefill 区间被算长，flash 内 per-token kernel 越界写 KV。**方案**：改为减 `dp_decode_tokens`（token 数，`fs.shape[12]`，与 `qkv->shape[0]` 同口径的 dp 域值），字段一并改名 `decode_token_num` 防复发（`flash_attention_layer.h`、`flash_attention.cpp`、`nvidia/flash_attention_layer.cpp`）。

3. **attn 分桶：GQA paged decode 需遍历所有 q_seq_len 桶**：上游 `AttentionMetaCreator::BuildAttentionMetas` 已按 q_seq_len 分桶，但 `PagedAttention::Forward` 只取 `page_metas.back()`。MTP 打开后 verify 造成 q_seq_len=2 与 =1 混批，其他桶被静默丢弃——这就是此前"并发乱码"的真实触发条件（混批，不是并发）：一个 decode 请求带 2 个 token，而 prefill 后的第一个 decode 请求只有 1 个 token，同 batch 混入不同 token 数的 decode 请求即乱码。**方案**：改为循环所有桶逐桶调 kernel（与 MLA 路径 `multihead_latent_attention.cpp` 同款模式），利用"桶按 q_seq_len 降序且在 token 布局上连续"的上游不变量，用 `bucket_offset_tokens` 游标顺序走 decode 区，并加 `bucket_offset + bucket_len <= dp_decode_tokens` 的 CHECK 防越界；`PagedAttentionInputs` 新增 `bucket_batch_size`/`bucket_token_num`/`bucket_offset_tokens` 字段，仅在多桶时填充（默认关闭），单桶路径字节级一致，不影响任何 MTP off 的模型。对比 vllm 的做法（非稳态 decode 并入 prefill 走 varlen attn、稳态 decode 走 paged attn，只启动两个 attn 算子），本方案保持 batch 组织不变，改动面更小。ACL/Kunlun 的 kernel 只消费单桶，遇多桶显式返回 `RET_NOT_IMPLEMENTED`，拒绝而非静默错算。

4. **fallback 路径算子缺失，需补齐而非绕开**：(a) `topkSigmoidKernel` 只做了 `w = sigmoid(topk(raw_logits))`，选专家没加 `e_score_correction_bias`、权重没归一化、没乘 `routed_scaling_factor`——与 DeepSeek-V3/GLM4-MoE 的 sigmoid 路由语义不符；(b) `Fp8MoeLayer` 从 `eplb_map`（专家并行映射表）读 routing bias，而 MoE 模块把 bias 传在 `e_score_bias` 上——之前不炸只是因为 `InvokeMoeCutlassGemm` 丢弃了该指针未解引用。**方案**：kernel 增加 bias/renorm/scaling 三个参数（`moe_kernels.cu/.h`、`moe_wrapper.cu/.h`），`moe_kernel_wrapper.cpp` 仅在 `FAST_TOPK_SIGMOID_SCORE` 路径接线新参数（softmax 路径保持 `NT` 原行为）；`Fp8MoeLayer` 改从 `e_score_bias` 读并加 fp32 dtype CHECK。

**方案总体思路**：三条工作流一次落地——(1) `csrc/models/hy_v3/` 新增模型实现 + 注册 shim；(2) FlashAttn 两处修复（token 数扣除 + 多桶分发）；(3) sigmoid 路由 fallback 算子补齐。除注册接线的最小 shim 外，不改任何已有模型的行为。

### B. 代码定位

- `csrc/models/hy_v3/hy_v3_model.h/.cpp`（新增）— `HyV3DecoderLayer`（dense `TwoLayeredFFN` 前缀 + MoE block + 单 shared expert）、`HyV3MtpLayer`（enorm/hnorm/concat/eh_proj 后接标准 decoder 层）、`HyV3Model`
- `csrc/models/hy_v3/hy_v3_config.h`、`hy_v3_config_parser.h/.cpp`（新增）— config 解析；`csrc/models/common_moe/moe_config.h::PrepareHyV3Attributes`（新增函数，不动既有 `Prepare*Attributes`）
- `csrc/models/hy_v3/hy_v3_weight_loader.h/.cpp`（新增）— 权重过滤（MTP off 丢 nextn 层/shared head、PP 范围外丢弃）与 main+MTP 层融合
- `csrc/models/hy_v3/hy_v3_mtp_draft_model.h/.cpp`（新增）— `DraftModelInterface` 实现；hy_v3 每个 MTP 层有独立 final_layernorm（区别于 GLM4-MoE 复用主模型 norm，同 DeepSeek-V3 注入方式）
- `csrc/models/hy_v3/hy_v3_test.cpp`（新增）— T01~T04 单测（config parser 16 例、weight loader 8 例、model 创建 2 例、draft model 3 例、factory 2 例等）
- 注册 shim：`csrc/models/base/model_arch.h/.cpp`（`ARCH_HY_V3`）、`csrc/runtime/model_instance.cpp`（hy_v3 分支，`enable_qk_pre_norm_before_rotary_pos=true`、`use_qk_norm=false`）、`csrc/models/config_parser/model_config_parser_factory.cpp`、`csrc/models/weight_loader/model_weight_loader_factory.cpp`、`csrc/models/CMakeLists.txt`、`csrc/configure/model_config_parser.cpp`
- FlashAttn 修复：`csrc/modules/basic/flash_attention.cpp`（`decode_token_num = fs.shape[12]`）、`csrc/layers/flash_attention_layer.h`、`csrc/layers/nvidia/flash_attention_layer.cpp`、`csrc/modules/basic/paged_attention.cpp`（多桶循环 + 越界 CHECK；ACL/Kunlun 单桶守卫）、`csrc/layers/paged_attention_layer.h`（per-bucket 字段）、`csrc/layers/nvidia/paged_attention_layer.cpp`、`csrc/layers/attention_layer_test.cpp`
- MoE 修复：`csrc/kernels/nvidia/mixture_of_experts/moe_kernels.cu/.h`、`moe_wrapper.cu/.h`、`topk_sigmoid_test.cu`（新增，T05）、`csrc/kernels/nvidia/moe_kernel_wrapper.cpp`、`csrc/layers/nvidia/fp8_moe_layer.cpp`、`csrc/layers/nvidia/moe_layer.cpp`、`csrc/models/weight_method/fp8_e4m3_method.cpp`
- Python/配置：`python/ksana_llm/arg_utils.py`（hy_v3 回退直读 config.json）、`python/weight_map/hy_v3_weight_map.json`（新增）、`examples/hy3_*.yaml`（新增示例）

### C. 调用链影响

- `ModelInstance::Load`（`csrc/runtime/model_instance.cpp`）— 新增 `hy_v3` 分支，既有分支顺序与行为不变
- `PagedAttention::Forward`（`csrc/modules/basic/paged_attention.cpp`）— 唯一调用方为 attention layer 前向；单桶走原路径，多桶为新增行为，既有模型（MTP off 恒单桶）不受影响
- `InvokeMoeCutlassGemm`（`csrc/kernels/nvidia/moe_kernel_wrapper.cpp`）— 签名扩展三个参数，仅 sigmoid 路径消费；softmax 路径传参语义不变
- `EnvModelConfigParser::ParseModelConfig`（`csrc/configure/model_config_parser.cpp`）— 新增 `hy_v3` else-if 分支，其他 model_type 不经过
- `EngineArgs`（`python/ksana_llm/arg_utils.py`）— 仅当 transformers 抛含 "hy_v3" 的 `ValueError` 时进入新回退分支，既有 deepseek_v32/v4 分支逻辑等价保留

### E. 测试要点

来自 `0-clarification.md §G` 的逐条映射：

- T01（config parser）→ `hy_v3_test.cpp::HyV3ConfigParserTest.*`（16 例：架构/gating 字段、真实 config、rope_theta 顶层与嵌套、缺省默认值、softmax 回退、无 shared expert、`expert_hidden_dim` 回退、tie_embeddings、无 MTP）
- T02（量化路径）→ `ParsesFp8StaticQuantConfig`/`ParsesFp8BlockwiseConfig`/`BF16NotQuantized`/`RejectsUnsupportedQuantMethod`
- T03（权重加载）→ `HyV3WeightLoaderTest.*`（factory、shared head/nextn 层过滤、PP 范围外丢弃、bf16/fp8 method 注册、权重族路由、main+MTP 融合）
- T04（模型创建）→ `HyV3ModelCreationTest.*`、`HyV3MtpDraftModelTest.*`、`HyV3FactoryTest.*`
- T05（sigmoid 路由 kernel）→ `csrc/kernels/nvidia/mixture_of_experts/topk_sigmoid_test.cu`（bias/renorm/scaling 与参考实现逐元素对比）
- T06（多桶分发）→ `csrc/layers/attention_layer_test.cpp` 新增多桶构造用例
- T07（单桶回归）→ 既有 `attention_layer_test`/`moe` 相关用例全绿即视为回归通过
- T08（端到端冒烟）→ CI GPU 任务编译执行 `hy_v3_test`（`11c1cd0e`），真机 checkpoint 冒烟走人工/集成验证，不在本仓库单测范围

### G. 风险待办

- **多桶分发的上游不变量依赖**：桶降序 + 连续布局是 `ReorderInferRequests`/`BuildAttentionMetas` 的既有语义，本次未新增约束但新增了对该语义的消费；`ai-review` 阶段需重点核对 `bucket_offset_tokens` 游标与 `dp_decode_tokens` 的配对关系（CHECK 已兜底，但 CHECK 触发即推理失败，需确认正常路径永不触发）
- **`fs.shape[12]` 布局耦合**：flexible shape 下标是隐式协议，后续 shape 布局调整时必须同步本处（已在代码注释中写明语义）
- **`arg_utils.py` 的 ValueError 文本匹配**：transformers 升级后报错文案变化会导致回退失效（退化为抛错，可被发现，不会静默错）
- **ACL/Kunlun 行为收紧**：MTP verify 混批从"静默错"变为"显式拒绝"，相关平台开 MTP 前需另行适配
- **`topk_method` 被强制为 "fast"**：fp8 路径仅接线 (fast, sigmoid)，配置其他组合不会生效；若需求方需要 (greedy, softmax) 之外的组合需另行评估

### Env 执行环境提示

- 本次改动能否在编码环境内做语法/类型级自检？**部分能**——纯 C++/Python 代码，但 kernel（`.cu`）与 CUDA 相关单测需要 CUDA 编译环境；无 GPU 环境下 `hy_v3_test` 的 GPU 用例按 `c8628584` 的 skip 保护跳过，编译验证仍可进行
- 是否涉及需要容器内才能验证的行为？T01~T07 均为进程内单测，无外部服务依赖；T08 端到端需要 Nv GPU 真机 + hy3 checkpoint，属集成验证
- 建议的验证范围（增量单测目标）：`hy_v3_test`、`topk_sigmoid_test`、`model_config_parser_test`、`attention_layer_test`
- 是否预判需要集成测试：**是（范围外）**——T08 真机冒烟由需求方在具备 hy3 checkpoint 的环境执行，作为本需求的最终验收，不阻塞合入
