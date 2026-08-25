## 0. Summary Card【必读】
- 检索是否命中：是；命中文档数：3
- 关键结论数：8；待确认缺口数：1
- 最近一次更新阶段：design

## A. 检索到的相关文档【clarify 起草，design 可追加】
- `ai_docs/modules.md` — 相关原因：子域划分真源 — 结论摘要：现有业务子域未覆盖 OpenAI API 适配层，本需求经用户确认使用兜底子域 `main`
- `ai_docs/code_wiki/2. 推理服务与OpenAI接口.md` — 相关原因：描述 reasoning parser 注册及响应组装 — 结论摘要：parser 应将模型输出拆分为 `reasoning_content` 与 `content`；DeepSeek V4 使用注册名 `deepseek_v4` 的 Qwen3 parser
- `ai_docs/code_wiki/2.2. OpenAI兼容API.md` — 相关原因：描述 OpenAI chat serving 与启动配置 — 结论摘要：`OpenAIConfig.reasoning_parser` 控制 parser 初始化，chat serving 负责非流式与流式响应组装

## B. 关键结论（供 go 编码遵守）【必填，可为空但需说明为空原因】
- 启动命令已显式传入 `--reasoning-parser=deepseek_v4`，不得把本问题误归因为 parser 漏配或改成自动选择 parser 的需求
- DeepSeek V4 请求以 `chat_template_kwargs.thinking=true` 显式开启 thinking；判断函数必须收到 parser 配置名，不能收到运行时布尔状态
- 故障只在非流式响应组装路径必现；流式链路已有真实响应证据表明 reasoning/content 拆分正常
- 目标响应必须把思考内容与最终答案分别放入 `reasoning_content` 和 `content`，不得向 `content` 泄漏 `</think>`

## C. 待确认缺口（ai_docs 未覆盖，凭经验假设的点）【必填，可为空】
- 实际 DeepSeek V4 服务构建的精确 SHA 未从部署环境读取 — 当前 design 工作区为 `231ee7760e8fcde9a15a69cb7790afd2be269284`（`team/master`）并包含问题引入提交 `0117f2ad` — 影响：服务级验证前需确认部署版本一致

## D. 供 review 核对的约束清单【必填，可为空】
- 只修正非流式 thinking 状态判断所用的 parser 参数，不改 parser 注册名、启动参数语义或请求协议
- 非流式 `thinking=true` 响应必须同时返回正确拆分的 `reasoning_content` 与 `content`
- `content` 中不得残留思考过程或 `</think>` 标记
- 流式响应现有的 `delta.reasoning_content` → `delta.content` 切换行为不得回归
- 新增回归测试必须使用无 `<think>` 起始文本、仅含 `</think>` 分隔符的现场输出形态，保证旧代码确实失败
- `get_enable_thinking_from_request` 的其它三个调用点和 `ReasoningParser.extract_reasoning_content` 本体不在改动范围内
- pytest 必须从 `python/` 目录以 `PYTHONPATH=.` 运行，并检查输出摘要，避免 `conftest.py` 的退出钩子掩盖收集错误

## E. 代码调研结果摘要【design 补充】
- `0117f2ad` 将类内 thinking 判断抽成共享 helper；非流式调用点在迁移时把 parser 名误写成 `_in_reasoning` 布尔状态，流式和 sampling config 调用点参数正确
- 非流式主链为 `OpenAIAdapter.chat_completions` → `create_chat_completion` → `_chat_completion_full_generator` → `extract_reasoning_content` → `ChatMessage`
- 最小生产改动仅涉及 `python/openaiapi/serving_chat.py::_chat_completion_full_generator` 的一个实参；工作区中的一行 diff 即用户确认的最终修复
- 自动化回归落点为 `python/tests/test_xgrammar_request_converter.py`，该文件已有完整响应 mock 基础设施；正确工作目录下已成功收集 58 个现有用例
