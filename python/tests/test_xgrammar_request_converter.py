# Copyright 2026 Tencent Inc.  All rights reserved.

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from openaiapi.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    StructuredOutputsParams,
)
from openaiapi.openai_adapter import OpenAIConfig
from openaiapi.reasoning import ReasoningParserManager
from openaiapi.request_converter import RequestConverter
from openaiapi.serving_chat import KsanaOpenAIServingChat
from openaiapi.tool_parsers import ToolParserManager


def make_request(**kwargs):
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        **kwargs,
    )


def make_tools(*, strict=True):
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "strict": strict,
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "unit"],
                "additionalProperties": False,
            },
        },
    }]


def strict_converter(tool_parser, reasoning_parser=None):
    return RequestConverter(
        OpenAIConfig(
            tool_call_parser=tool_parser,
            reasoning_parser=reasoning_parser,
            enable_auto_tool_choice=True,
        )
    )


def walk_formats(value):
    if isinstance(value, dict):
        if "type" in value:
            yield value
        for child in value.values():
            yield from walk_formats(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_formats(child)


@pytest.mark.parametrize(
    ("structured_outputs", "config_key", "expected"),
    [
        ({"json": {"type": "string"}}, "json_schema", '{"type":"string"}'),
        ({"regex": "[A-Z]+"}, "regex", "[A-Z]+"),
        ({"grammar": 'root ::= "yes" | "no"'}, "grammar", 'root ::= "yes" | "no"'),
    ],
)
def test_structured_outputs_are_forwarded(structured_outputs, config_key, expected):
    converted = RequestConverter().convert_chat_completion(
        make_request(structured_outputs=structured_outputs)
    )

    sampling_config = converted["sampling_config"]
    assert sampling_config["enable_structured_output"] is True
    assert sampling_config[config_key] == expected


def test_native_structural_tag_response_format_is_forwarded():
    response_format = {
        "type": "structural_tag",
        "format": {"type": "const_string", "value": "ok"},
    }
    converted = RequestConverter().convert_chat_completion(
        make_request(response_format=response_format)
    )

    payload = json.loads(converted["sampling_config"]["structural_tag"])
    assert payload == response_format


def test_legacy_structural_tag_response_format_is_normalized():
    converted = RequestConverter().convert_chat_completion(
        make_request(
            response_format={
                "type": "structural_tag",
                "structures": [
                    {
                        "begin": "<tool>",
                        "schema": {"type": "object"},
                        "end": "</tool>",
                    }
                ],
                "triggers": ["<tool>"],
            }
        )
    )

    payload = json.loads(converted["sampling_config"]["structural_tag"])
    assert payload["type"] == "structural_tag"
    assert payload["format"]["type"] == "triggered_tags"
    assert payload["format"]["tags"][0]["content"]["json_schema"] == {
        "type": "object"
    }


def test_structured_outputs_reject_multiple_constraints():
    with pytest.raises(ValidationError):
        StructuredOutputsParams(regex="[0-9]+", grammar="root ::= [0-9]+")


@pytest.mark.parametrize("strict", [True, False])
def test_response_format_strict_remains_protocol_metadata(strict):
    request = make_request(response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object"},
            "strict": strict,
        },
    })
    converted = RequestConverter().convert_chat_completion(request)

    assert request.response_format.json_schema.strict is strict
    assert converted["sampling_config"]["json_schema"] == '{"type":"object"}'


def test_openai_strict_fields_default_to_false():
    request = make_request(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
            },
        },
        tools=[{
            "type": "function",
            "function": {
                "name": "answer",
                "parameters": {"type": "object"},
            },
        }],
    )

    assert request.response_format.json_schema.strict is False
    assert request.tools[0].function.strict is False


def test_legacy_structured_output_regex_is_forwarded():
    converted = RequestConverter().convert_chat_completion(
        make_request(structured_output_regex="[a-z]+")
    )

    sampling_config = converted["sampling_config"]
    assert sampling_config["enable_structured_output"] is True
    assert sampling_config["regex"] == "[a-z]+"


def test_completion_structured_outputs_are_forwarded():
    converted = RequestConverter().convert_completion(
        CompletionRequest(prompt="hello", structured_outputs={"regex": "[a-z]+"})
    )

    sampling_config = converted["sampling_config"]
    assert sampling_config["enable_structured_output"] is True
    assert sampling_config["regex"] == "[a-z]+"


@pytest.mark.parametrize(
    ("tool_parser", "expected_trigger"),
    [
        ("deepseek_v3", "<｜tool▁calls▁begin｜>"),
        ("deepseek_v31", "<｜tool▁calls▁begin｜>"),
        ("deepseek_v32", "<｜DSML｜function_calls>"),
        ("deepseek_v4", "<｜DSML｜tool_calls>"),
        ("glm47", "<tool_call>"),
        ("kimi_k2", "<|tool_calls_section_begin|>"),
        ("llama3_json", '{\\"name\\": '),
        ("llama4_json", '{\\"name\\": '),
    ],
)
def test_strict_auto_uses_xgrammar_model_structural_tag(
    tool_parser, expected_trigger
):
    converted = strict_converter(tool_parser).convert_chat_completion(
        make_request(tools=make_tools(strict=True), tool_choice="auto")
    )

    sampling_config = converted["sampling_config"]
    assert sampling_config["enable_structured_output"] is True
    assert "json_schema" not in sampling_config
    assert expected_trigger in sampling_config["structural_tag"]

    payload = json.loads(sampling_config["structural_tag"])
    schemas = [
        item["json_schema"]
        for item in walk_formats(payload)
        if item.get("type") == "json_schema"
    ]
    assert make_tools()[0]["function"]["parameters"] in schemas


@pytest.mark.parametrize("tool_choice", ["auto", None])
def test_auto_without_strict_keeps_generation_unconstrained(tool_choice):
    converted = strict_converter("deepseek_v3").convert_chat_completion(
        make_request(tools=make_tools(strict=False), tool_choice=tool_choice)
    )

    sampling_config = converted["sampling_config"]
    assert "structural_tag" not in sampling_config
    assert "json_schema" not in sampling_config
    assert "enable_structured_output" not in sampling_config


def test_implicit_auto_with_strict_uses_model_structural_tag():
    request = make_request(tools=make_tools(strict=True))
    assert request.tool_choice == "auto"

    converted = strict_converter("deepseek_v3").convert_chat_completion(request)

    assert "structural_tag" in converted["sampling_config"]


@pytest.mark.parametrize("tool_choice", ["required", {
    "type": "function", "function": {"name": "get_weather"}
}])
def test_forced_choice_uses_native_protocol_even_without_strict(tool_choice):
    converted = strict_converter("deepseek_v3").convert_chat_completion(
        make_request(tools=make_tools(strict=False), tool_choice=tool_choice)
    )

    sampling_config = converted["sampling_config"]
    assert "structural_tag" in sampling_config
    assert "json_schema" not in sampling_config
    assert "<｜tool▁calls▁begin｜>" in sampling_config["structural_tag"]


def test_non_strict_tool_is_schema_unconstrained_inside_active_tag():
    tools = make_tools(strict=True) + [{
        "type": "function",
        "function": {
            "name": "search",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }]
    converted = strict_converter("deepseek_v3").convert_chat_completion(
        make_request(tools=tools, tool_choice="auto")
    )

    payload = json.loads(converted["sampling_config"]["structural_tag"])
    schemas = [
        item["json_schema"]
        for item in walk_formats(payload)
        if item.get("type") == "json_schema"
    ]
    assert tools[0]["function"]["parameters"] in schemas
    assert True in schemas


def test_reasoning_mode_is_composed_by_xgrammar_model_api():
    converted = strict_converter(
        "deepseek_v3", reasoning_parser="deepseek_v3"
    ).convert_chat_completion(
        make_request(
            tools=make_tools(strict=True),
            tool_choice="auto",
            chat_template_kwargs={"thinking": True},
        )
    )

    payload = json.loads(converted["sampling_config"]["structural_tag"])
    assert payload["format"]["type"] == "sequence"
    assert "</think>" in converted["sampling_config"]["structural_tag"]


def test_parser_without_xgrammar_model_uses_legacy_required_json():
    converted = strict_converter("hermes").convert_chat_completion(
        make_request(tools=make_tools(strict=True), tool_choice="required")
    )

    sampling_config = converted["sampling_config"]
    assert "structural_tag" not in sampling_config
    assert "json_schema" in sampling_config


@pytest.mark.parametrize(
    ("tool_parser", "model_output"),
    [
        (
            "deepseek_v3",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
            "<｜tool▁sep｜>get_weather\n```json\n"
            '{"city":"Beijing","unit":"celsius"}'
            "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        ),
        (
            "deepseek_v31",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather"
            '<｜tool▁sep｜>{"city":"Beijing","unit":"celsius"}'
            "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        ),
        (
            "deepseek_v32",
            '\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">Beijing'
            "</｜DSML｜parameter>"
            '<｜DSML｜parameter name="unit" string="true">celsius'
            "</｜DSML｜parameter></｜DSML｜invoke>\n"
            "</｜DSML｜function_calls>",
        ),
        (
            "deepseek_v4",
            '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">Beijing'
            "</｜DSML｜parameter>"
            '<｜DSML｜parameter name="unit" string="true">celsius'
            "</｜DSML｜parameter></｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>",
        ),
        (
            "kimi_k2",
            "<|tool_calls_section_begin|><|tool_call_begin|>"
            "functions.get_weather:0<|tool_call_argument_begin|>"
            '{"city":"Beijing","unit":"celsius"}'
            "<|tool_call_end|><|tool_calls_section_end|>",
        ),
        (
            "glm47",
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "<arg_key>unit</arg_key><arg_value>celsius</arg_value>"
            "</tool_call>",
        ),
        (
            "llama3_json",
            '{"name": "get_weather", "parameters": '
            '{"city":"Beijing","unit":"celsius"}}',
        ),
    ],
)
def test_xgrammar_protocol_is_accepted_by_model_tool_parser(
    monkeypatch, tool_parser, model_output
):
    parser_cls = ToolParserManager.get_tool_parser(tool_parser)
    for attribute in (
        "tool_calls_start_token_id",
        "tool_calls_end_token_id",
        "tool_call_start_token_id",
        "tool_call_end_token_id",
    ):
        if hasattr(parser_cls, attribute):
            monkeypatch.setattr(parser_cls, attribute, 1)

    parsed = parser_cls().extract_tool_calls(
        model_output,
        make_request(tools=make_tools(strict=True), tool_choice="required"),
    )

    assert parsed.tools_called is True
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].function.name == "get_weather"
    assert json.loads(parsed.tool_calls[0].function.arguments) == {
        "city": "Beijing",
        "unit": "celsius",
    }


@pytest.mark.parametrize("tool_choice", [
    "required",
    {"type": "function", "function": {"name": "get_weather"}},
])
def test_forced_choice_with_structural_tag_model_is_detected(
        tool_choice, monkeypatch):
    monkeypatch.setattr(
        "openaiapi.tool_parsers.structural_tag.get_model_structural_tag",
        object(),
    )
    serving = KsanaOpenAIServingChat.__new__(KsanaOpenAIServingChat)
    serving.config = SimpleNamespace(enable_auto_tool_choice=True)
    serving.tool_parser = ToolParserManager.get_tool_parser("deepseek_v3")
    request = make_request(tools=make_tools(), tool_choice=tool_choice)

    assert serving._uses_model_structural_tag(request) is True


def test_structural_tag_eligibility_for_auto_and_unsupported_parsers(
        monkeypatch):
    monkeypatch.setattr(
        "openaiapi.tool_parsers.structural_tag.get_model_structural_tag",
        object(),
    )
    serving = KsanaOpenAIServingChat.__new__(KsanaOpenAIServingChat)
    serving.config = SimpleNamespace(enable_auto_tool_choice=True)

    serving.tool_parser = ToolParserManager.get_tool_parser("hermes")
    request = make_request(tools=make_tools(), tool_choice="required")
    assert serving._uses_model_structural_tag(request) is False

    serving.tool_parser = ToolParserManager.get_tool_parser("deepseek_v3")
    request = make_request(tools=make_tools(strict=False), tool_choice="auto")
    assert serving._uses_model_structural_tag(request) is False

    request = make_request(tools=make_tools(strict=True), tool_choice="auto")
    assert serving._uses_model_structural_tag(request) is True


@pytest.mark.parametrize(
    "tool_parser",
    [
        "deepseek_v3", "deepseek_v31", "deepseek_v32", "deepseek_v4",
        "glm47", "kimi_k2"
    ],
)
def test_native_protocol_tokens_are_preserved_for_forced_choices(
    monkeypatch, tool_parser
):
    monkeypatch.setattr(
        "openaiapi.tool_parsers.structural_tag.get_model_structural_tag",
        object(),
    )
    parser_cls = ToolParserManager.get_tool_parser(tool_parser)
    for attribute in ("tool_calls_start_token_id", "tool_calls_end_token_id"):
        if hasattr(parser_cls, attribute):
            monkeypatch.setattr(parser_cls, attribute, 1)

    request = make_request(
        tools=make_tools(), tool_choice="required", skip_special_tokens=True
    )

    adjusted = parser_cls().adjust_request(request)

    assert adjusted.skip_special_tokens is False


@pytest.mark.parametrize("tool_parser", ["deepseek_v3", "deepseek_v31"])
def test_native_protocol_tokens_remain_filtered_without_structural_tag(
        monkeypatch, tool_parser):
    parser_cls = ToolParserManager.get_tool_parser(tool_parser)
    for attribute in ("tool_calls_start_token_id", "tool_calls_end_token_id"):
        monkeypatch.setattr(parser_cls, attribute, 1)

    request = make_request(
        tools=make_tools(strict=False),
        tool_choice="auto",
        skip_special_tokens=True,
    )

    adjusted = parser_cls().adjust_request(request)

    assert adjusted.skip_special_tokens is True


@pytest.mark.parametrize("tool_parser", ["deepseek_v3", "deepseek_v31"])
def test_native_protocol_tokens_are_preserved_for_strict_auto(
        monkeypatch, tool_parser):
    monkeypatch.setattr(
        "openaiapi.tool_parsers.structural_tag.get_model_structural_tag",
        object(),
    )
    parser_cls = ToolParserManager.get_tool_parser(tool_parser)
    for attribute in ("tool_calls_start_token_id", "tool_calls_end_token_id"):
        monkeypatch.setattr(parser_cls, attribute, 1)

    request = make_request(
        tools=make_tools(strict=True),
        tool_choice="auto",
        skip_special_tokens=True,
    )

    adjusted = parser_cls().adjust_request(request)

    assert adjusted.skip_special_tokens is False


def make_eos_test_serving(
        monkeypatch, tool_parser, decoded_text, uses_structural_tag=True):
    parser_cls = ToolParserManager.get_tool_parser(tool_parser)
    for attribute in ("tool_calls_start_token_id", "tool_calls_end_token_id"):
        monkeypatch.setattr(parser_cls, attribute, 1)

    serving = KsanaOpenAIServingChat.__new__(KsanaOpenAIServingChat)
    serving.tool_parser = parser_cls
    serving.reasoning_parser = None
    serving.tool_call_id_type = "random"
    serving.response_role = "assistant"
    serving.config = SimpleNamespace(enable_auto_tool_choice=True)
    serving.tokenizer = SimpleNamespace(
        decode=lambda _token_ids, skip_special_tokens: decoded_text)
    serving._uses_model_structural_tag = lambda _request: uses_structural_tag
    return serving, parser_cls


@pytest.mark.asyncio
async def test_deepseek_v4_non_streaming_response_splits_reasoning_content(
        monkeypatch):
    parser_cls = ReasoningParserManager.get_reasoning_parser("deepseek_v4")
    monkeypatch.setattr(parser_cls, "think_start_token_id", 1)
    monkeypatch.setattr(parser_cls, "think_end_token_id", 2)

    serving = KsanaOpenAIServingChat.__new__(KsanaOpenAIServingChat)
    serving.tool_parser = None
    serving.reasoning_parser = parser_cls
    serving.tool_call_id_type = "random"
    serving.response_role = "assistant"
    serving.config = SimpleNamespace(
        enable_auto_tool_choice=False,
        reasoning_parser="deepseek_v4",
    )
    serving.tokenizer = SimpleNamespace(
        decode=lambda _token_ids, skip_special_tokens:
            "reasoning text</think>final answer"
    )
    serving._uses_model_structural_tag = lambda _request: False

    status = SimpleNamespace(OK=lambda: True)
    output = SimpleNamespace(output_tokens=[[1]], logprobs=None)
    timing = SimpleNamespace(
        preprocess_time_ms=0,
        waiting_time_ms=0,
        encoder_time_ms=0,
        encoder_send_time_ms=0,
        encoder_queue_time_ms=0,
        encoder_encode_time_ms=0,
        encoder_recv_time_ms=0,
    )
    request = make_request(chat_template_kwargs={"thinking": True})

    response = await serving._chat_completion_full_generator(
        [(status, output, timing)],
        request_id="request-id",
        model_name="test-model",
        request=request,
        prompt_tokens_per_request=[0],
    )

    message = response.choices[0].message
    assert message.reasoning_content == "reasoning text"
    assert message.content == "final answer"
    assert "</think>" not in message.content
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_parser", ["deepseek_v3", "deepseek_v31"])
@pytest.mark.parametrize("uses_structural_tag", [True, False])
async def test_full_response_eos_is_scoped_to_structural_tag(
        monkeypatch, tool_parser, uses_structural_tag):
    eos_token = "<｜end▁of▁sentence｜>"
    decoded_text = f"before{eos_token}after{eos_token}"
    serving, parser_cls = make_eos_test_serving(
        monkeypatch, tool_parser, decoded_text, uses_structural_tag)
    assert parser_cls.EOS_Token == eos_token

    status = SimpleNamespace(OK=lambda: True)
    output = SimpleNamespace(output_tokens=[[1]], logprobs=None)
    timing = SimpleNamespace(
        preprocess_time_ms=0,
        waiting_time_ms=0,
        encoder_time_ms=0,
        encoder_send_time_ms=0,
        encoder_queue_time_ms=0,
        encoder_encode_time_ms=0,
        encoder_recv_time_ms=0,
    )
    request = make_request(
        tools=make_tools(strict=uses_structural_tag), tool_choice="auto")

    response = await serving._chat_completion_full_generator(
        [(status, output, timing)],
        request_id="request-id",
        model_name="test-model",
        request=request,
        prompt_tokens_per_request=[0],
    )

    expected = (
        f"before{eos_token}after" if uses_structural_tag else decoded_text)
    assert response.choices[0].message.content == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_parser", ["deepseek_v3", "deepseek_v31"])
async def test_structural_tag_streaming_does_not_emit_trailing_eos(
        monkeypatch, tool_parser):
    eos_token = "<｜end▁of▁sentence｜>"
    decoded_text = "plain text" + eos_token
    serving, _ = make_eos_test_serving(
        monkeypatch, tool_parser, decoded_text)
    request = make_request(
        tools=make_tools(strict=True), tool_choice="auto", stream=True)

    async def outputs():
        yield SimpleNamespace(output_tokens=[[1]])

    chunks = [
        chunk async for chunk in serving._chat_completion_stream_generator(
            outputs(),
            request_id="request-id",
            model_name="test-model",
            tokenizer=serving.tokenizer,
            request=request,
            prompt_tokens_count=[0],
        )
    ]
    content = ""
    for chunk in chunks:
        for line in chunk.splitlines():
            if not line.startswith("data: {"):
                continue
            payload = json.loads(line.removeprefix("data: "))
            content += payload["choices"][0]["delta"].get("content", "")

    assert content == "plain text"
    assert eos_token not in "".join(chunks)


@pytest.mark.parametrize(
    "tool_parser, model_output",
    [
        (
            "deepseek_v3",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
            "function<｜tool▁sep｜>get_weather\n```json\n"
            '{"city":"Beijing","unit":"celsius"}\n```'
            "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        ),
        (
            "deepseek_v31",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather"
            '<｜tool▁sep｜>{"city":"Beijing","unit":"celsius"}'
            "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        ),
        (
            "kimi_k2",
            "<|tool_calls_section_begin|><|tool_call_begin|>"
            "functions.get_weather:0<|tool_call_argument_begin|>"
            '{"city":"Beijing","unit":"celsius"}'
            "<|tool_call_end|><|tool_calls_section_end|>",
        ),
    ],
)
def test_structural_tag_streaming_accepts_split_markers_without_token_ids(
        monkeypatch, tool_parser, model_output):
    parser_cls = ToolParserManager.get_tool_parser(tool_parser)
    for attribute in (
        "tool_calls_start_token_id",
        "tool_calls_end_token_id",
        "tool_call_start_token_id",
        "tool_call_end_token_id",
    ):
        monkeypatch.setattr(parser_cls, attribute, 999)

    parser = parser_cls()
    parser.model_structural_tag_enabled = True
    request = make_request(tools=make_tools(strict=True), tool_choice="required")

    previous_text = ""
    current_text = ""
    name_seen = False
    argument_parts = []
    content_parts = []

    for chunk in model_output:
        previous_text = current_text
        current_text += chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if delta is None:
            continue
        if delta.content:
            content_parts.append(delta.content)
        if not delta.tool_calls:
            continue
        tool_call = delta.tool_calls[0]
        if tool_call.function and tool_call.function.name:
            name_seen = True
            assert tool_call.function.name == "get_weather"
        if tool_call.function and tool_call.function.arguments:
            argument_parts.append(tool_call.function.arguments)

    assert name_seen is True
    assert json.loads("".join(argument_parts)) == {
        "city": "Beijing",
        "unit": "celsius",
    }
    assert "".join(content_parts) == ""
    assert parser.prev_tool_call_arr


def test_structural_tag_streaming_flushes_unmatched_marker_prefix(monkeypatch):
    parser_cls = ToolParserManager.get_tool_parser("deepseek_v3")
    monkeypatch.setattr(parser_cls, "tool_calls_start_token_id", 999)
    monkeypatch.setattr(parser_cls, "tool_calls_end_token_id", 999)
    parser = parser_cls()
    parser.model_structural_tag_enabled = True
    request = make_request(tools=make_tools(strict=True), tool_choice="auto")

    delta = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="plain<",
        delta_text="plain<",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )

    assert delta.content == "plain"
    assert parser.flush_streaming("plain<").content == "<"


def test_structural_tag_end_marker_search_advances_incrementally(monkeypatch):
    parser_cls = ToolParserManager.get_tool_parser("deepseek_v31")
    monkeypatch.setattr(parser_cls, "tool_calls_start_token_id", 999)
    monkeypatch.setattr(parser_cls, "tool_calls_end_token_id", 999)
    parser = parser_cls()
    parser.model_structural_tag_enabled = True
    request = make_request(tools=make_tools(strict=True), tool_choice="required")
    current_text = parser.tool_calls_start_token + "x" * 4096

    delta = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=current_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )

    assert delta is None
    assert parser._structural_tag_end_search_offset == (
        len(current_text) - len(parser.tool_calls_end_token) + 1
    )
