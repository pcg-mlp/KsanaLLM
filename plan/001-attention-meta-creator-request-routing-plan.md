# 001 - AttentionMetaCreator 请求路由策略下沉重构

> **分支**: `feature/001-attention-meta-creator-request-routing`
> **创建时间**: 2026-05-29
> **TAPD链接**: 待补充
> **问题来源**: Compressor 路径（DeepSeekV4）的请求分桶策略与通用策略不同，当前逻辑散落在框架层 `ModelInput::PrepareInputInfo`，无法由模型特定的 Creator 覆写。
> **相关现状**: `csrc/models/base/attention_meta/attention_meta_creator.h`、`csrc/models/base/model_input.cpp`、`csrc/models/deepseek_v4/attention_meta/deepseek_v4_attention_meta_creator.h`、`csrc/models/deepseek_v4/attention_meta/deepseek_v4_compressor_builder.h`

---

## 一、目标

1. 将"哪些请求进 FlashAttentionMeta、哪些进 PagedAttentionMeta、如何按 `q_seq_len` 分 bucket"的路由判断逻辑，从 `ModelInput::PrepareInputInfo` 下沉到 `AttentionMetaCreator`。
2. 新增虚方法 `BuildAttentionMetas(forward_reqs, attn_dp_group_id)` 并返回 `FlashAttentionMeta`/`PagedAttentionMeta` 数组，默认实现复刻现有 `PrepareInputInfo` 逻辑。
3. `DeepSeekV4AttentionMetaCreator` 重写该方法，使 Compressor 路径使用 `q_seq_len == 1` → Paged、其余 → Flash 的专属策略，消除 `model_input.cpp` 中的 `dynamic_cast` 和 `use_dsv4_attn` 特判代码。
4. `ModelInput::PrepareInputInfo` 退化为调用 creator 并接收结果，不再包含任何路由逻辑。

一句话目标：

> **让每个模型通过重写 `AttentionMetaCreator::BuildAttentionMetas` 自主决定请求分桶策略，框架层不再感知模型特定的路由规则。**

---

## 二、当前问题判断

### 2.1 路由逻辑与 Creator 职责不匹配

`AttentionMetaCreator` 当前仅提供无参数工厂方法 `CreateFlashAttentionMeta()` / `CreatePagedAttentionMeta()`，负责"创建什么类型的对象"。但"为每批次创建几个、哪些请求放入哪个对象"的路由逻辑，却完全写在 `ModelInput::PrepareInputInfo()` 中：

```cpp
// model_input.cpp PrepareInputInfo()
for (const auto& req : forward_reqs) {
  if (req.GetType() == ForwardRequestType::kFlash) {
    flash_meta.dp_reqs.emplace_back(...);
  } else {
    if (page_metas.empty() || page_metas.back()->q_seq_len != input_ids_len) {
      page_metas.emplace_back(attention_meta_creator_->CreatePagedAttentionMeta());
      ...
    }
    page_metas.back()->dp_reqs.emplace_back(...);
  }
}
```

此路由策略硬编码在框架层，模型特定的 Creator 无法覆写。

### 2.2 Compressor 路径的特殊路由无处放置

DeepSeekV4 Compressor 要求不同于通用策略：
- **通用策略**：`ForwardRequestType::kFlash`（`GetInputIdsLength() > decode_token_num_threshold` 或 `kv_cached_token_num == 0`）→ Flash；`kPage` → Paged。
- **Compressor 策略**：`q_seq_len == 1` → PagedAttentionMeta；其余（包括多 token decode MTP 场景）→ FlashAttentionMeta。

当前的妥协方案是 `ParseFromRequests` 在调用 `PrepareInputInfo` 之后再执行 `dynamic_cast` 修正：

```cpp
// model_input.cpp ParseFromRequests()
if (model_config_.use_dsv4_attn) {
  auto* dsv4_flash_meta = dynamic_cast<DeepSeekV4FlashAttentionMeta*>(flash_meta.get());
  DeepSeekV4PagedAttentionMeta* dsv4_paged_meta = nullptr;
  for (auto& pi : page_metas) {
    if (pi->q_seq_len == 1) {
      dsv4_paged_meta = dynamic_cast<DeepSeekV4PagedAttentionMeta*>(pi.get());
      break;
    }
  }
  PrepareCompressor(forward_reqs, dsv4_shared->compressor,
                    dsv4_flash_meta->compressor, dsv4_paged_meta->compressor, context_);
}
```

这段代码的问题：
1. 框架层 (`model_input.cpp`) 直接引用模型层头文件 `deepseek_v4_attention_meta.h`、`deepseek_v4_compressor_builder.h`，形成向上依赖。
2. `PrepareCompressor` 依赖"q_seq_len==1 的 paged bucket 已经被 `PrepareInputInfo` 创建好"这一隐式前提；若未来 Compressor 的路由策略继续演化，需要同时修改两处。
3. 纯 prefill 批次（没有 q_seq_len==1 的 decode 请求）时需要临时构造 `DeepSeekV4PagedAttentionMeta tmp_paged_meta`，是 workaround 而非设计意图。

### 2.3 FlashAttentionMeta 与 CompressorAttentionMeta 一一绑定，但创建逻辑分离

`DeepSeekV4FlashAttentionMeta` 内嵌 `CompressorAttentionMeta compressor`；`DeepSeekV4PagedAttentionMeta` 同理。即每个 Flash/Paged meta 对象都对应一个 Compressor meta 对象。但当前"何时创建 Paged meta"由 `PrepareInputInfo` 决定，"Compressor meta 如何路由"由 `PrepareCompressor` 决定，两段逻辑割裂，维护时必须同步修改。

---

## 三、专项设计决策 / 落地方案

### 3.1 在 AttentionMetaCreator 新增 BuildAttentionMetas 虚方法

**接口设计**：

在 `AttentionMetaCreator` 中新增结构体 `MetaArrays` 及虚方法 `BuildAttentionMetas`：

```cpp
// attention_meta_creator.h

struct MetaArrays {
  // 当前批次应使用的 Flash meta（dp_reqs 已填充）。
  std::unique_ptr<FlashAttentionMeta> flash_meta;
  std::vector<std::unique_ptr<PagedAttentionMeta>> page_metas;

  // 请求统计，供 ModelInput 更新相关计数器。
  size_t multi_token_request_num = 0;
  size_t dp_multi_token_request_num = 0;
  size_t single_token_request_num = 0;
  size_t dp_single_token_request_num = 0;
};

// 根据本批次请求列表和当前 DP 组 ID，构建并返回填好 dp_reqs 的 Flash/Paged meta 对象数组。
// 默认实现使用通用路由策略（ForwardRequestType::kFlash / kPage）。
// 模型特定的 Creator 可重写此方法以实现不同的路由逻辑（如 Compressor 的 q_seq_len 策略）。
virtual MetaArrays BuildAttentionMetas(const std::vector<ForwardRequest>& forward_reqs,
                                        uint32_t attn_dp_group_id);
```

**默认实现**（复刻现有 `PrepareInputInfo` 路由逻辑）：

```cpp
MetaArrays AttentionMetaCreator::BuildAttentionMetas(
    const std::vector<ForwardRequest>& forward_reqs, uint32_t attn_dp_group_id) {
  MetaArrays result;
  result.flash_meta = CreateFlashAttentionMeta();
  auto& flash_meta = *result.flash_meta;

  for (const auto& req : forward_reqs) {
    if (req.GetType() == ForwardRequestType::kFlash) {
      ++result.multi_token_request_num;
      if (req.attn_dp_group_id == attn_dp_group_id) {
        ++result.dp_multi_token_request_num;
        flash_meta.dp_reqs.emplace_back(const_cast<ForwardRequest*>(&req));
      }
    } else {
      ++result.single_token_request_num;
      if (req.attn_dp_group_id == attn_dp_group_id) {
        ++result.dp_single_token_request_num;
        const size_t input_ids_len = req.GetInputIdsLength();
        if (result.page_metas.empty() ||
            result.page_metas.back()->q_seq_len != input_ids_len) {
          result.page_metas.emplace_back(CreatePagedAttentionMeta());
          result.page_metas.back()->q_seq_len = input_ids_len;
          result.page_metas.back()->expanded_batch_size =
              (input_ids_len & 1 ? input_ids_len : input_ids_len >> 1);
        }
        result.page_metas.back()->dp_reqs.emplace_back(
            const_cast<ForwardRequest*>(&req));
      }
    }
  }
  return result;
}
```

### 3.2 DeepSeekV4AttentionMetaCreator 重写为 Compressor 策略

**路由规则**：遍历 `forward_reqs`，按 `q_seq_len`（即 `req.GetInputIdsLength()`）分桶：
- `q_seq_len == 1` → PagedAttentionMeta（decode 路径）
- 其余 → FlashAttentionMeta（prefill / MTP 多 token decode 路径）

`PrepareCompressor` 调用逻辑从 `model_input.cpp::ParseFromRequests` 移入 `DeepSeekV4AttentionMetaCreator::BuildAttentionMetas`，在构建完 `flash_meta` / `page_metas` 后立即执行。

**纯 prefill 批次的处理**：若 `page_metas` 为空（无 q_seq_len==1 的请求），`BuildAttentionMetas` 内直接创建临时 paged meta 并调用 `PrepareCompressor`，不再在 `model_input.cpp` 中 workaround。

### 3.3 ModelInput::PrepareInputInfo 退化为 thin wrapper

`PrepareInputInfo` 改为：

```cpp
void ModelInput::PrepareInputInfo(const std::vector<ForwardRequest>& forward_reqs) {
  // Reset shared tensors（保留现有 shape reset 代码）
  ...

  // 委托 creator 完成请求路由
  auto arrays = attention_meta_creator_->BuildAttentionMetas(forward_reqs, attn_dp_group_id_);

  // 将结果写入 ModelInput 成员
  flash_meta = std::move(arrays.flash_meta);
  page_metas = std::move(arrays.page_metas);
  multi_token_request_num   = arrays.multi_token_request_num;
  dp_multi_token_request_num = arrays.dp_multi_token_request_num;
  single_token_request_num   = arrays.single_token_request_num;
  dp_single_token_request_num = arrays.dp_single_token_request_num;
  dp_batch_size = dp_single_token_request_num + dp_multi_token_request_num;
}
```

### 3.4 model_input.cpp 中的 DSV4 特判代码删除

`ParseFromRequests` 中以下代码块可完全删除：

```cpp
// 待删除
if (model_config_.use_dsv4_attn) {
  auto* dsv4_shared = dynamic_cast<DeepSeekV4SharedAttentionMeta*>(...);
  auto* dsv4_flash_meta = dynamic_cast<DeepSeekV4FlashAttentionMeta*>(flash_meta.get());
  DeepSeekV4PagedAttentionMeta* dsv4_paged_meta = nullptr;
  for (auto& pi : page_metas) { ... }
  PrepareCompressor(...);
}
```

同时可移除 `model_input.cpp` 对 `deepseek_v4_attention_meta.h` 和 `deepseek_v4_compressor_builder.h` 的 `#include`，消除框架层对模型层的向上依赖。

---

## 四、建议改动范围 / 影响范围

### 4.1 `csrc/models/base/attention_meta/attention_meta_creator.h`

- 新增 `MetaArrays` 结构体（嵌套在 `AttentionMetaCreator` 内或独立于命名空间内）。
- 新增虚方法 `virtual MetaArrays BuildAttentionMetas(const std::vector<ForwardRequest>&, uint32_t attn_dp_group_id)`。
- 原有 `CreateFlashAttentionMeta()` / `CreatePagedAttentionMeta()` 保留为 `protected` 工具方法，供子类 `BuildAttentionMetas` 调用。

### 4.2 `csrc/models/base/attention_meta/attention_meta_creator.cpp`

- 实现 `BuildAttentionMetas` 默认逻辑（见 3.1 节代码）。

### 4.3 `csrc/models/base/model_input.h` / `model_input.cpp`

- `PrepareInputInfo` 改为 thin wrapper（见 3.3 节）。
- 移除对 `deepseek_v4_attention_meta.h`、`deepseek_v4_compressor_builder.h` 的 `#include`。
- 删除 `ParseFromRequests` 中的 DSV4 特判块（见 3.4 节）。
- `ModelInput::flash_meta` 保持 `unique_ptr<FlashAttentionMeta>` 类型，每批次直接接收 `arrays.flash_meta`（`std::move`）。

### 4.4 `csrc/models/deepseek_v4/attention_meta/deepseek_v4_attention_meta_creator.h` / `.cpp`

- 声明并实现 `BuildAttentionMetas` override，内含 Compressor 路由策略及 `PrepareCompressor` 调用。
- 添加对 `deepseek_v4_compressor_builder.h` 的 `#include`（原本在 `model_input.cpp` 中）。

### 4.5 `csrc/models/deepseek_v4/attention_meta/deepseek_v4_compressor_builder.h` / `.cpp`

- 函数签名不变。
- 若纯 prefill 临时 meta 的逻辑完全移入 creator，可删除 `model_input.cpp` 端的 fallback 构造。

---

## 五、推荐实施步骤 / 预期收益

### Phase A：新增接口并落地默认实现

1. 在 `attention_meta_creator.h` 中定义 `MetaArrays` 和 `BuildAttentionMetas` 虚方法声明。
2. 在 `attention_meta_creator.cpp` 中实现默认逻辑，与现有 `PrepareInputInfo` 等价。
3. 修改 `ModelInput::PrepareInputInfo` 调用新接口，验证通用路径行为不变（单元测试 + 集成测试通过）。

### Phase B：DeepSeekV4 Creator 实现 Compressor 策略

1. 在 `DeepSeekV4AttentionMetaCreator` 中实现 `BuildAttentionMetas` override。
2. 内部逻辑：按 `q_seq_len` 路由（1 → Paged，其余 → Flash），并在方法末尾调用 `PrepareCompressor`。
3. 确保纯 prefill 批次（无 q_seq_len==1 请求）走新的内部 fallback，不再需要 `model_input.cpp` 的 workaround。

### Phase C：清理框架层残余依赖

1. 删除 `model_input.cpp` 的 DSV4 特判块。
2. 移除 `model_input.cpp` 对 `deepseek_v4_attention_meta.h` 和 `deepseek_v4_compressor_builder.h` 的 `#include`。
3. 跑全量回归，确认无行为差异。

**预期收益**：
- 框架层（`model_input.cpp`）不再包含任何模型特定逻辑，消除向上依赖。
- 新增 attention 路径（如未来的 Indexer、MQA 等）可直接在各自的 Creator 中声明分桶策略，无需修改 `PrepareInputInfo`。
- Compressor 的路由逻辑与 Compressor meta 的绑定关系在同一个 Creator 中，可读性与可维护性显著提升。

---

## 六、验收标准 / 验证关注点

### 6.1 通用路径行为不变

1. 所有现有非 DSV4 模型（llama、qwen 等）在接入新接口后，推理结果与原有输出逐 token 一致。
2. `multi_token_request_num`、`single_token_request_num`、`dp_*` 等计数器在各批次的值与原实现完全相同。

### 6.2 DSV4 Compressor 路径正确路由

1. `q_seq_len == 1` 的请求出现在 `page_metas` 的对应 bucket 中，不再出现在 `flash_meta.dp_reqs`。
2. `q_seq_len > 1` 的 decode 请求（MTP 场景）出现在 `flash_meta.dp_reqs` 而非 `page_metas`。
3. 纯 prefill 批次（无 q_seq_len==1 请求）时，`PrepareCompressor` 能正常完成，不崩溃、不报错。

### 6.3 框架层依赖消除

1. 编译后 `model_input.cpp` 不再包含对 `deepseek_v4_attention_meta.h` / `deepseek_v4_compressor_builder.h` 的直接引用。
2. `nm` / `objdump` 检查：`ModelInput::PrepareInputInfo` 符号不引用任何 `DeepSeekV4*` 符号。

### 6.4 性能无回归

1. 通用路径（llama/qwen）的单步推理延迟 P99 无显著变化（±2% 以内）。
2. DSV4 路径的单步推理延迟 P99 无显著变化（±2% 以内）。

---

## 七、实际验证记录（可选）

> 待实现后补充。

---

## 九、风险与注意事项

1. **`flash_meta` 生命周期变化**：当前 `flash_meta` 在构造时分配一次，改为每步接收 creator 返回的新对象后，对象的地址会变化。若下游代码保存了裸指针（如 `*flash_meta.get()` 的临时引用），需检查并修正。
2. **attn_dp_group_id 传参**：`BuildAttentionMetas` 需要 `attn_dp_group_id_` 作为参数，该字段是 `ModelInput` 的私有成员，需确保在调用 `PrepareInputInfo` 之前已正确初始化。
3. **请求排序前提**：现有代码假设同一 dp 组的请求按 `q_seq_len` 降序排列（保证分 bucket 正确）。此前提应在新接口的默认实现中保留，并在注释中明确。
4. **单元测试覆盖**：`PrepareInputInfo` 目前缺少独立单元测试；建议在 Phase A 结束后补充对 `BuildAttentionMetas` 默认实现的参数化测试，覆盖纯 prefill、纯 decode、混合三种批次。
5. **MTP 场景兼容性**：DSV4 MTP 场景中，同一批次可能同时存在 `q_seq_len > 1` 的 page 请求（expanded decode）和正常 prefill 请求。`DeepSeekV4AttentionMetaCreator::BuildAttentionMetas` 的路由逻辑需与 `PrepareCompressor` 内部的请求遍历保持一致，避免统计错位。

---

## 十、结论

根本原因：`AttentionMetaCreator` 仅负责对象类型创建，请求路由逻辑留在框架层，导致模型特定路由规则（Compressor 的 `q_seq_len` 策略）无法封装，只能以 `dynamic_cast` 特判的形式嵌入框架层代码。

修复方向：在 `AttentionMetaCreator` 新增 `BuildAttentionMetas` 虚方法，将"为本批次构建多少 Flash/Paged meta、每个 meta 包含哪些请求"的完整决策权交给 Creator，框架层退化为结果消费者，实现关注点分离。
