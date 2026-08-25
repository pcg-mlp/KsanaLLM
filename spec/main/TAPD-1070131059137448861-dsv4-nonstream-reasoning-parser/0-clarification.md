---
spec_version: "1"
spec_type: change
id: "1070131059137448861"
status: draft
subdomain: main
cross_subdomains: []
complexity: M
work_type: bugfix
---

# DeepSeek V4 非流式响应 reasoning/content 混合修复

### A. 工作类型（work_type）【必填，先于目标】

`bugfix` — bug 修复（纠正 DeepSeek V4 非流式响应中的 reasoning 内容拆分错误，使返回语义对齐 OpenAI 兼容协议）。

### B. 目标（goal）

> [BLOCK] TAPD 拉取失败，已转为会话口述澄清，建议事后补拉核对。

在最新合入的 master 版本启动 DeepSeek V4 服务时，已显式配置 `--reasoning-parser=deepseek_v4`。向 `/v1/chat/completions` 发送非流式请求，并在 `chat_template_kwargs` 中指定 `"thinking": true` 后，每次都能复现返回结果错误：思考内容、`</think>` 结束标记和最终答案全部落入 `message.content`，没有正常拆分至 `message.reasoning_content` 与 `message.content`。

期望非流式响应与已经验证正常的流式响应保持一致语义：思考过程写入 `message.reasoning_content`，最终回答写入 `message.content`；最终回答中不得残留思考过程或 `</think>` 标记。修复范围限定在 DeepSeek V4 已启用 reasoning parser 的非流式响应解析路径，不改变当前工作正常的流式解析行为、启动参数约定或对外请求字段。

根因已由现有代码改动确认：`python/openaiapi/serving_chat.py` 的非流式响应组装分支调用 `get_enable_thinking_from_request` 时，错误地传入了 `reasoning_parser._in_reasoning` 布尔状态，而该函数需要的是 parser 配置名。对于 DeepSeek V4，这会导致请求中的 `thinking: true` 无法按 `deepseek_v4` 规则识别。最终修复是改为传入 `self.config.reasoning_parser`，使非流式 parser 正确进入 reasoning 状态并执行内容拆分。

现场依据：

- 现象摘要：结果错误，非流式响应 `chatcmpl-9f18717f46444cec8b251bcdb8d593ed` 的 `message.content` 同时包含思考内容、`</think>` 与最终答案；复现频率为每次必现。
- 触发条件：用户确认复现于最新合入的 master；模型为 `DeepSeek-V4-Flash-0731`；请求为非流式，且 `chat_template_kwargs.thinking=true`。精确 master commit SHA 未由用户提供，澄清时工作区 HEAD 为 `231ee7760e8fcde9a15a69cb7790afd2be269284`，两者是否完全一致留待验证阶段核对。
- 启动配置：`SELECT_ALL_REDUCE_BY_SIZE=1`、`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`、`CUDA_VISIBLE_DEVICES=4,5,6,7`；服务使用 `dsv4.yaml`，监听 `0.0.0.0:12244`，并传入 `--reasoning-parser=deepseek_v4 --enable-auto-tool-choice --tool-call-parser=deepseek_v4`。
- 对照材料：同一模型的流式请求已实测正常；响应 `chatcmpl-9e5d3236bc644a498ded37c302073201` 先连续输出 `delta.reasoning_content`，随后输出 `delta.content`，最终正常结束。因此流式链路作为不应回归的已知基线，不属于本次故障路径。
- 与已知模式的关系：`ai_docs` 已说明 reasoning parser 应把思考过程与正式回答分离，也说明 DeepSeek V4 复用注册为 `deepseek_v4` 的 Qwen3 reasoning parser；知识库未记录本次非流式分支传参错误。

验收时使用原始非流式请求复测：响应必须同时具备语义正确的 `message.reasoning_content` 和 `message.content`，且 `message.content` 不含思考内容或 `</think>`；另以已验证的流式请求做回归，确认其 reasoning/content 增量拆分行为保持正常。

**结论：修正 DeepSeek V4 非流式分支传给 thinking 判断函数的 parser 参数，使 reasoning_content 与 content 按协议稳定分离，同时保持流式链路行为不变。**
