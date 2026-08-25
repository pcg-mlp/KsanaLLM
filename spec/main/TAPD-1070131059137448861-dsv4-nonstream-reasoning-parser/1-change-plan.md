### A. 问题分析与方案总述【必填】

- 问题分析：提交 `0117f2ad` 将 `KsanaOpenAIServingChat._get_enable_thinking_from_request()` 抽取为共享函数 `openaiapi.utils.get_enable_thinking_from_request(request, reasoning_parser)`。新函数的第二个参数契约是 `Optional[str]` parser 配置名；流式解析、采样配置和 structural-tag 调用点均传入配置名，唯独 `_chat_completion_full_generator()` 误传 `reasoning_parser._in_reasoning` 布尔状态。DeepSeek V4 请求开始解析时该状态为 `False`，helper 因而返回 `False`；当模型完整输出没有 `<think>` 起始文本、只有 `</think>` 结束标记时，`ReasoningParser.extract_reasoning_content()` 也无法自行判定正处于 reasoning，最终把完整文本原样放入 `content`。
- 方案总体思路：保留工作区中已经确认的最小修复，只把非流式调用点的第二个参数改为 `self.config.reasoning_parser`，恢复抽取前“按 parser 配置解释请求开关”的语义。选择修调用点而不修改 helper、parser 或协议，是因为其余调用点和真实流式请求均已工作正常；扩大改动反而会引入跨模型行为风险。补充一个直接覆盖 `_chat_completion_full_generator()` 的异步单测，使用“无 `<think>` 起始文本、含 `</think>`”的 DeepSeek V4 输出，确保测试在旧代码上失败并能长期锁住本次参数契约。

### B. 代码定位【必填】

- `python/openaiapi/serving_chat.py` — 保留现有最终修复：非流式完整响应组装时，用 parser 配置名初始化实例的 `_in_reasoning`，随后调用完整文本拆分逻辑 — 涉及函数：`KsanaOpenAIServingChat._chat_completion_full_generator(...)`、`get_enable_thinking_from_request(request, self.config.reasoning_parser)`。
- `python/tests/test_xgrammar_request_converter.py` — 在已有 `_chat_completion_full_generator()` mock 测试附近新增 `test_deepseek_v4_non_streaming_response_splits_reasoning_content`；引入 `ReasoningParserManager`，将 `deepseek_v4` parser 的 `think_start_token_id` / `think_end_token_id` 设为有效测试值，构造 `reasoning text</think>final answer` 的 decode 结果与 `chat_template_kwargs={"thinking": True}` 请求，断言 `reasoning_content == "reasoning text"`、`content == "final answer"` 且 `content` 不含 `</think>` — 涉及接口：`ReasoningParserManager.get_reasoning_parser("deepseek_v4")`、`KsanaOpenAIServingChat._chat_completion_full_generator(...)`。

### C. 调用链影响【必填】

- `POST /v1/chat/completions` — 调用方：`python/openaiapi/openai_adapter.py::register_all_routes._chat_completions()` → `OpenAIAdapter.chat_completions()` → `KsanaOpenAIServingChat.create_chat_completion()`；只有 `request.stream == False` 时进入本次修正的 `_chat_completion_full_generator()`。
- `get_enable_thinking_from_request(request, reasoning_parser)` — 调用方：`serving_chat.py::_chat_completion_stream_generator()`、`serving_chat.py::create_chat_completion()` 的 sampling config、`serving_chat.py::_chat_completion_full_generator()`、`tool_parsers/structural_tag.py::build_model_structural_tag()`；本次只改第三个调用点，其余调用点已传 `self.config.reasoning_parser` 或 `reasoning_parser_name`，保持不动。
- `ReasoningParser.extract_reasoning_content(model_output)` — 调用方：`_chat_completion_full_generator()`；依赖调用前正确设置 `_in_reasoning`，再将返回的二元组写入 `openai_protocol.ChatMessage.reasoning_content` 与 `content`。方法签名和响应模型不变。
- `--reasoning-parser=deepseek_v4` — 传播链：`python/ksana_llm/arg_utils.py` → `KsanaService.initialize()` → `OpenAIConfig.reasoning_parser` → `KsanaOpenAIServingChat._init_parsers()`；启动链已实测有效，不在本次改动范围内。

### E. 测试要点【必填】

- 单元回归：新增的非流式测试必须使用不含 `<think>`、但含 `</think>` 的完整输出；在修复前应得到 `reasoning_content is None` 且完整文本落入 `content`，在修复后必须精确拆分为 reasoning 与最终答案。若测试输入带 `<think>`，旧实现也可能通过，不能证明本 bug 已被覆盖。
- 单元断言：`message.reasoning_content == "reasoning text"`、`message.content == "final answer"`、`"</think>" not in message.content`，并确认 `finish_reason == "stop"`。
- Python 回归：从仓库 `python/` 目录运行整个相关测试文件，预期现有 58 个用例加新增用例全部通过；命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -p no:cacheprovider -q tests/test_xgrammar_request_converter.py
```

- 服务级复测：使用澄清文档中的 DeepSeek V4 启动参数和原始非流式 curl，断言最终 JSON 同时含正确的 `message.reasoning_content` 与 `message.content`，且 `content` 不泄漏 reasoning 或 `</think>`。
- 流式回归：重放已验证正常的 `stream=true` 请求，确认 SSE 仍先输出 `delta.reasoning_content`、后输出 `delta.content`，结束块为 `finish_reason="stop"`；本项用于证明一行非流式修复没有影响流式基线。

### G. 风险待办【必填，可为空】

- 回归测试若包含 `<think>` 起始文本，会绕过错误调用参数并在旧代码上误通过；必须使用现场一致的“隐式进入 thinking、仅输出 `</think>`”文本形态。
- 当前 design 工作区为 detached HEAD `231ee7760e8fcde9a15a69cb7790afd2be269284`（`team/master`），包含引入问题的 `0117f2ad`；服务级验证前需确认实际部署构建也基于该链路，避免拿不同版本结果归因。
- `python/tests/conftest.py` 注册了 `atexit -> os._exit(0)`；从错误目录运行时曾出现 collection error 但进程退出码仍为 0。`ai-go` / `ai-verify` 必须从 `python/` 目录运行，并同时检查 pytest 输出中的 collected/passed/error 摘要，不能只判断 shell 退出码。

### Env 执行环境提示【必填，本项目新增】

- 本次改动能否在编码环境内做语法/类型级自检？能；当前环境从 `python/` 目录使用 `PYTHONPATH=.` 已成功收集 `test_xgrammar_request_converter.py` 的 58 个现有用例。新增用例落地后可直接运行目标 pytest 文件。
- 是否涉及需要容器内才能验证的行为？涉及。真实 DeepSeek V4 输出需要 `dsv4.yaml`、模型权重、4 张可见 GPU 及启动中的 OpenAI 服务，编码环境只能完成 mock 单测，不能把真实服务验证写成已通过。
- 建议的验证范围（增量单测目标）：`python/tests/test_xgrammar_request_converter.py::test_deepseek_v4_non_streaming_response_splits_reasoning_content`，并回归整个 `python/tests/test_xgrammar_request_converter.py`。
- 是否预判需要集成测试：是；单元测试验证调用参数与拆分语义，真实模型请求验证端到端响应字段。建议自动化程度：需人工触发服务与 curl，但字段是否分离可用 JSON/SSE 断言自动判定。
