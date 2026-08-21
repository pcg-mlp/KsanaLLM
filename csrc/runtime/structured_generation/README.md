# KsanaLLM XGrammar 结构化生成维护说明

本文面向维护 KsanaLLM 结构化生成、OpenAI Function Calling 和 XGrammar
后端的开发者。当前代码固定使用 XGrammar v0.2.5，版本入口位于
`cmake/external/xgrammar.cmake`。

## 1. 支持范围

KsanaLLM 当前通过同一个 XGrammar 后端支持四类约束：

| 类型 | 请求字段 | XGrammar 编译入口 |
| --- | --- | --- |
| JSON Schema | `sampling_config.json_schema` | `CompileJSONSchema` |
| Regex | `sampling_config.regex` | `CompileRegex` |
| EBNF Grammar | `sampling_config.grammar` | `CompileGrammar` |
| Structural Tag | `sampling_config.structural_tag` | `CompileStructuralTag` |

一个请求最多只能设置一种约束。`SamplingConfig::VerifyArgs()` 会拒绝同时设置
多种约束；`ApplyStructuredGeneratorConfig()` 将实际字段转换为统一的
`StructuredGeneratorConfig`。

Structural Tag 与前三类约束有一个重要区别：它描述的是完整输出状态机，可以同时
包含 reasoning、普通文本和 tool call 的协议边界。因此
`StructuredGeneratorFactory::CreateGenerator()` 不会再用旧的
`ReasoningStructuredGenerator` 包装 Structural Tag。JSON、Regex 和 Grammar 仍沿用
旧 reasoning-prefix Wrapper。

## 2. Strict Function Calling 链路

OpenAI 请求中的 `tools[].function.strict` 在 Python API 层处理，完整链路如下：

```text
ChatCompletionRequest.tools[].function.strict
  -> tool_parsers.structural_tag.build_model_tool_structural_tag()
  -> XGrammar v0.2.5 get_model_structural_tag()
  -> RequestConverter.sampling_config.structural_tag
  -> SamplingConfig / StructuredGeneratorConfig
  -> GrammarBackendNvidia::CompileStructuralTag()
  -> GrammarMatcherWrapper
  -> FillNextTokenBitmask() / AcceptToken()
  -> 对应模型的 ToolParser
  -> OpenAI tool_calls 响应
```

`tool_choice` 的处理规则：

- `none`：不创建 tool Structural Tag。
- `auto` 或未显式指定：只有至少一个函数设置 `strict: true` 时才创建；模型仍可
  选择普通文本或调用工具。
- `required`：使用模型原生 Structural Tag 强制至少一次工具调用。
- 指定函数：Structural Tag 只允许调用指定函数。
- 混合 strict/non-strict tools：strict tool 使用其参数 JSON Schema；显式
  `strict: false` 的 tool 在同一 Structural Tag 内使用不受 Schema 限制的 JSON。

模型原生 Structural Tag 的优先级高于旧的通用 JSON tool Wrapper。生成完成后，
`serving_chat.py` 必须继续使用模型自己的 ToolParser，而不能把整段模型协议当成普通
JSON 解析。

## 3. Strict 的三个不同含义

以下概念不能混用：

1. `tools[].function.strict`
   - OpenAI 每个 tool 的参数 Schema 遵循开关。
   - 由 XGrammar 的模型 Structural Tag builder 读取。
2. `response_format.json_schema.strict`
   - OpenAI 协议元数据，当前不向 C++ GrammarBackend 传递。
3. `GrammarCompiler::CompileJSONSchema(..., strict_mode=true)`
   - XGrammar 编译器内部策略。
   - KsanaLLM 固定传 `true`，用于补全 Schema 未明确声明的 object/array 行为。
   - 它不代表某个 OpenAI tool 设置了 `strict: true`。

## 4. 当前模型映射

只有当 KsanaLLM 已有 ToolParser 且其输出语法与 XGrammar 模板完全一致时，才能设置
`ToolParser.structural_tag_model`。

| KsanaLLM ToolParser | XGrammar v0.2.5 模型键 |
| --- | --- |
| `deepseek_v3` | `deepseek_r1` |
| `deepseek_v31` | `deepseek_v3_1` |
| `deepseek_v32` | `deepseek_v3_2` |
| `deepseek_v4` | `deepseek_v4` |
| `kimi_k2` | `kimi` |
| `glm47` | `glm_4_7` |
| `llama3_json` / `llama4_json` | `llama` |

XGrammar 支持但 KsanaLLM 尚无对应 ToolParser 的模板不能只靠增加映射宣称支持；必须
先实现并验证完整、流式两种输出解析。

## 5. XGrammar Python builder 的来源

KsanaLLM C++ 侧直接链接 XGrammar，不安装其完整 Python native package。原因是
XGrammar v0.2.5 的 Python 包依赖范围与当前项目的 Transformers 版本不兼容。

模型 Structural Tag builder 本身是纯 Python：

- 源码开发时，`python/openaiapi/_xgrammar/__init__.py` 从 CMake 固定版本的
  FetchContent 源码目录加载。
- 构建 wheel 时，`setup.py` 将同一份 checkout 中的 `structural_tag.py`、
  `openai_tool_call_schema.py` 和 `builtin_structural_tag.py` 原样复制进 wheel。
- 调试非标准构建目录时，可设置 `KSANA_XGRAMMAR_PYTHON_PATH` 指向
  `xgrammar/python/xgrammar`。

不要在 KsanaLLM 中复制或手写模型模板，否则会产生第二套协议实现并与 XGrammar
版本漂移。

## 6. Matcher 生命周期

`GrammarStructuredGenerator` 暴露以下 XGrammar matcher 能力：

- `FillNextTokenBitmask()`：获得下一 token 的允许集合。
- `AcceptToken()`：提交生成 token。
- `Rollback()`：回退已接受 token。
- `Reset()`：恢复到初始 matcher 状态。
- `IsCompleted()`：约束是否已完整匹配。
- `IsTerminated()` / `IsValid()`：matcher 异常或可用状态。

`FindJumpForwardTokens()` 当前明确返回 `false`，主生成链路尚未接入 JumpForward。
批量 matcher 也不在当前实现范围内。

## 7. 新增模型的步骤

1. 确认固定版本 XGrammar 已存在目标模型模板。
2. 实现或确认 KsanaLLM ToolParser 能解析该模板的准确输出语法。
3. 在 ToolParser 类上设置 `structural_tag_model`，不要在 `serving_chat.py` 增加
   模型名分支。
4. 如果协议分隔符属于特殊 token，在 `adjust_request()` 中设置
   `skip_special_tokens = False`。
5. 补齐以下测试：
   - `strict: true + tool_choice=auto` 的 Structural Tag 转换；
   - `required` 和指定函数；
   - strict/non-strict 混合 tools；
   - XGrammar 模板输出可被 ToolParser 完整解析；
   - streaming 跨 token 边界；
   - reasoning 开启和关闭。

## 8. 调试日志

启用 debug 日志后，可按以下顺序定位问题：

1. `ApplyStructuredGeneratorConfig`：请求是否启用结构化输出、选择了哪种约束、
   constraint spec 的长度和内容。
2. `GrammarBackendNvidia::Compile*`：进入了哪个 XGrammar 编译入口、原始 spec、
   编译后的 grammar 内存大小。
3. `ModelDriver::HandleSchedule`：请求是否携带约束以及是否标记创建 generator。

这些日志可能包含用户提交的 Schema、Regex 或 Grammar，仅应在受控调试环境启用
debug 级别日志。

## 9. 验证命令

```bash
DISABLE_GENERATE_TRITON_KERNEL=1 cmake --build build \
  --target grammar_test xgrammar_generator_test -j

./build/bin/grammar_test --gtest_color=no
./build/bin/xgrammar_generator_test --gtest_color=no

PYTHONPATH=python pytest -q python/tests/test_xgrammar_request_converter.py
```
