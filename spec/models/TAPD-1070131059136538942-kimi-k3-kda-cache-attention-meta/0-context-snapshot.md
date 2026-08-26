## 0. Summary Card【必读】
- 检索是否命中：是；命中文档数：4
- 关键结论数：4；待确认缺口数：2
- 最近一次更新阶段：clarify

## A. 检索到的相关文档【clarify 起草，design 可追加】
- `ai_docs/modules.md` — 相关原因：子域划分真源 — 结论摘要：`cache_layout` 归 `cache_manager`；`kimi_k3` 模型 / `attention_meta` 归 `models`；本需求跨两者，主 `models`
- `ai_docs/code_wiki/6.2. 块分配与缓存布局.md` §5 — 相关原因：cache_layout 三级基类 + `CacheLayoutFactory` 优先级与四种布局偏移计算 — 结论摘要：工厂判型优先级 `use_dsv4_attn → use_dsa → use_mla → 默认 MHA`；MHA/MLA/DSA/DSV4 各布局逐层偏移规则；`KDAStateLayout` 是**新增第五种 block 布局**（注册名 `mamba-state`），ai_docs 尚未收录 KDA——本提交为新增真源
- `ai_docs/code_wiki/8.1. 模型框架与实现.md` §5 — 相关原因：attention_meta 工厂体系 — 结论摘要：`AttentionMetaCreator` 经 `Config.block_cache_layout` 映射各层偏移；DSV3/V4 各有独立 creator；`KimiK3AttentionMetaCreator` 是**新增第三种模型级 creator**（继承 DSV3 叠加 KDA），ai_docs 仅列 DSV3/V4
- `ai_docs/code_wiki/8. 模型与算子框架.md` §2.4/§6.2 — 相关原因：模型接入路径与 creator 覆写约定 — 结论摘要：需自定义注意力的模型覆写 `CreateCustomAttentionMetaCreator` 返回专属 creator；Kimi K3 注册为 MoE/MLA——本提交把 KimiK3 creator 从 DSV3 升级为专属（叠加 KDA），ai_docs 需更新

## B. 关键结论（供 go 编码遵守）
- cache_layout 三级基类约定：`KDAStateLayout` 须继承 `BaseBlockCacheLayout`，实现 `Initialize` / `GetBlockCacheSize` / `GetKvCacheLayerOffsets` / `GetActiveKvCacheTypes`；逐层累加得 `block_cache_size_`，`KvCacheLayerOffsets` 用绝对字节偏移（已含层基址）——见 `6.2 §5.1`
- 模型级布局用 `RegisterBlockCacheLayout` 按类型名注册 / 查询；KDA 走 `"mamba-state"` 名（与 DSV4 压缩器状态池同命名约定），**不**写进主 KV `"kv-cache"` 名空间
- attention_meta creator 须通过 `Config.block_cache_layout` 映射各层缓存偏移（见 `8.1 §5.3`）；`KimiK3SharedAttentionMeta::Init` 从 `mamba-state` layout 取 conv / recurrent offset 灌入 `layer_mamba_kv_cache_ptr_tensor`，`InitMambaBasePointersIfNeeded` 首次 Prepare 时把偏移解析为绝对指针
- 需自定义注意力的模型覆写 `CreateCustomAttentionMetaCreator` 返回专属 creator（见 `8 §2.4`）；标准 MHA 复用默认，KimiK3 为 MLA+KDA 混合须专属，且继承 DSV3 而非从零写（复用 MLA/DSA/Indexer 步骤）

## C. 待确认缺口（ai_docs 未覆盖）
- `CompressorSlotResource` 复用语义 — 假设：DSV4 slot resource 的 `Alloc` / `Get` / 回收契约可直接复用于 KDA mamba 状态池（per-request slot，SplitFuse OK）— 影响：若 DSV4 slot 含压缩器专属回收逻辑会与 KDA 状态生命周期不符
- KDA forward dispatch stub — 假设：本提交 `CommonAttn` 对 KDA 层 `return Status()`（空操作）为有意分阶段，KDA kernel 调用后续 MR 接入 — 影响：若误当可用，端到端 KDA 输出静默错误（非报错）

## D. 供 review 核对的约束清单
- 不改 DSV3 / DSV4 attention_meta 既有行为（仅 `InitDeepSeekV3SharedMeta` 提 protected + 删一行过期注释）
- `cu_seqlens` 全链 `uint64_t`（wrapper 三 params + layer + test + python gen 四处同改）
- KDA 层不构造 `mla_`（`mla_` null）；非 KDA 层构造 `mla_`——派发由 `kimi_linear_config.IsKdaLayer` 决定
- dense / MoE 派发由 `first_k_dense_replace` 推导（移除 `is_moe` 形参）
- `MambaModelCacheLayout` 无参重载（executor）kv-cache 随 `use_mla` 选 Mla / Mha，须与 `RegisterLayouts` 一致
- mamba-state sizing 用通用 `GetBlockCacheSize`（去 `dynamic_pointer_cast<MambaBlockCacheLayout>` 强校验，因 `mamba-state` 现可能是 `KDAStateLayout`）
- 删除多模态视觉塔（`KimiK3VisionConfig` / `vision_tower.` / `mm_projector.` weight skip）
- `use_linear_attn` 与 `use_dsv4_attn` / `use_dsa` 互斥（factory `KLLM_CHECK`）
