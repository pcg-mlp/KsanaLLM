# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Reference:
# https://github.com/vllm-project/vllm/blob/v0.16.1rc0/vllm/tool_parsers/glm4_moe_tool_parser.py
"""
GLM-4 Tool Call Parser with re-parse-and-diff streaming approach.

Adapted from vLLM's Glm4MoeModelToolParser. On every streaming call, the parser
re-parses the full ``current_text`` to find ``<tool_call>`` regions, builds the
JSON arguments string for each tool call, and diffs against what was previously
sent to emit only new content.

This approach is robust against MTP (multi-token prediction) where delta_text
may contain multiple tokens spanning state boundaries, which caused bugs in the
previous incremental buffer-based state machine.
"""

import ast
import json
from collections.abc import Sequence
from typing import Any, Optional

import regex as re
from openaiapi.openai_protocol import (ChatCompletionRequest,
                                       ChatCompletionToolsParam,
                                       DeltaFunctionCall, DeltaMessage,
                                       DeltaToolCall,
                                       ExtractedToolCallInformation,
                                       FunctionCall, ToolCall)
from openaiapi.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)

from openaiapi.transformers_utils.chat_utils import make_tool_call_id

from utilize.logger import get_logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Utility functions ported from vllm/tool_parsers/utils.py
# ---------------------------------------------------------------------------


def partial_tag_overlap(text: str, tag: str) -> int:
    """Length of the longest prefix of *tag* that matches a suffix of *text*.

    E.g. text ending in ``"<tool_"`` returns 6 when tag is ``"<tool_call>"``.
    Returns 0 when there is no overlap.
    """
    max_check = min(len(tag) - 1, len(text))
    for k in range(max_check, 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


def extract_types_from_schema(schema: Any) -> list[str]:
    """Extract all possible type strings from a JSON Schema definition.

    Handles ``type`` (string or list), ``enum`` value inference, and
    recursive ``anyOf``/``oneOf``/``allOf``.  Returns ``["string"]``
    when no type information can be determined.
    """
    if schema is None or not isinstance(schema, dict):
        return ["string"]

    types: set[str] = set()

    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            types.add(type_value)
        elif isinstance(type_value, list):
            for t in type_value:
                if isinstance(t, str):
                    types.add(t)

    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        for value in schema["enum"]:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")

    for choice_field in ("anyOf", "oneOf", "allOf"):
        if choice_field in schema and isinstance(schema[choice_field], list):
            for choice in schema[choice_field]:
                types.update(extract_types_from_schema(choice))

    return list(types) if types else ["string"]


def find_tool_properties(
    tools: list[ChatCompletionToolsParam] | None,
    tool_name: str,
) -> dict[str, Any]:
    """Find a tool by name and return its properties dict, or {}."""
    if not tools:
        return {}
    for tool in tools:
        if tool.function.name == tool_name:
            params = tool.function.parameters
            return (params or {}).get("properties", {})
    return {}


# ---------------------------------------------------------------------------


@ToolParserManager.register_module("glm47")
class Glm47MoeToolParser(ToolParser):
    """Tool parser for GLM-4 models using re-parse-and-diff streaming.

    On every streaming call the parser re-parses ``current_text`` to find
    ``<tool_call>`` regions, builds the JSON arguments string for each tool
    call, and diffs against what was previously sent to emit only new content.

    This is robust against MTP where delta_text contains multiple tokens.
    """

    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"

    tool_call_start_token_id: Optional[int] = None
    tool_call_end_token_id: Optional[int] = None

    def __init__(self):
        super().__init__()
        # Stateful streaming fields
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.eos_tokens: list[str] = ["<|user|>", "<|endoftext|>", "<|observation|>"]
        self.arg_key_start: str = "<arg_key>"
        self.arg_key_end: str = "</arg_key>"
        self.arg_val_start: str = "<arg_value>"
        self.arg_val_end: str = "</arg_value>"

        self.tool_calls_start_token = self.tool_call_start_token

        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>\s*(\S+?)\s*(<arg_key>.*)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

        # Pre-compiled pattern for finding the last <arg_key>...</arg_key>
        # before a partial <arg_value> (used in _build_args_json_so_far).
        self._arg_key_pattern = re.compile(
            re.escape(self.arg_key_start) + r"(.*?)" + re.escape(self.arg_key_end),
            re.DOTALL,
        )

        # Streaming state for re-parse-and-diff approach
        self._sent_content_idx: int = 0
        self._tool_call_ids: list[str] = []
        self.temp_previous_text: str = ""
        self.temp_current_text: str = ""

    @staticmethod
    def _deserialize(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        return value

    @staticmethod
    def _json_escape_string_content(s: str) -> str:
        """JSON-escape string content for incremental streaming.

        This escapes the content that goes INSIDE a JSON string (between quotes),
        not including the surrounding quotes themselves.
        """
        if not s:
            return ""
        return json.dumps(s, ensure_ascii=False)[1:-1]

    @staticmethod
    def _is_string_type(
        tool_name: str,
        arg_name: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> bool:
        """Check if *arg_name* of *tool_name* is a string type.

        Uses extract_types_from_schema to handle complex schemas with
        anyOf/oneOf/allOf/enum correctly.
        """
        tool_properties = find_tool_properties(tools, tool_name)
        param_schema = tool_properties.get(arg_name)
        if param_schema is None:
            return False
        param_types = extract_types_from_schema(param_schema)
        return set(param_types) - {"null"} == {"string"}

    @staticmethod
    def _tools_enabled(request: ChatCompletionRequest) -> bool:
        """Return whether tool parsing should be applied for this request."""
        try:
            tools = getattr(request, "tools", None)
            tool_choice = getattr(request, "tool_choice", None)
            return bool(tools) and tool_choice != "none"
        except (AttributeError, TypeError):
            logger.exception("Failed to determine if tools are enabled.")
            return False

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust request parameters for tool call token handling."""
        request = super().adjust_request(request)
        if request.tools and request.tool_choice == "auto":
            # Ensure tool call tokens (<tool_call>, </tool_call>) are not skipped
            # during decoding. Even though they are not marked as special tokens,
            # setting skip_special_tokens=False ensures proper handling in
            # transformers 5.x where decoding behavior may have changed.
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        logger.debug("model_output: %s", model_output)
        for eos_token in self.eos_tokens:
            if eos_token in model_output:
                model_output = model_output.replace(eos_token, "")
        matched_tool_calls = self.func_call_regex.findall(model_output)
        try:
            tool_calls: list[ToolCall] = []
            for match in matched_tool_calls:
                tc_detail = self.func_detail_regex.search(match)
                if not tc_detail:
                    logger.warning(
                        "Failed to parse tool call details from: %s",
                        match,
                    )
                    continue
                tc_name = tc_detail.group(1).strip()
                tc_args = tc_detail.group(2)
                pairs = self.func_arg_regex.findall(tc_args) if tc_args else []
                arg_dct: dict[str, Any] = {}
                for key, value in pairs:
                    arg_key = key.strip()
                    if self._is_string_type(tc_name, arg_key, request.tools):
                        arg_val = value
                    else:
                        arg_val = self._deserialize(value.strip())
                    logger.debug("arg_key = %s, arg_val = %s", arg_key, arg_val)
                    arg_dct[arg_key] = arg_val
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tc_name,
                            arguments=json.dumps(arg_dct, ensure_ascii=False),
                        ),
                    )
                )
        except (IndexError, ValueError, AttributeError, TypeError):
            logger.exception("Failed to extract tool call spec")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        else:
            if len(tool_calls) > 0:
                content = model_output[: model_output.find(self.tool_calls_start_token)]
                if not content or not content.strip():
                    content = None
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=tool_calls, content=content
                )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # ------------------------------------------------------------------
    # Streaming: re-parse-and-diff approach (robust against MTP)
    # ------------------------------------------------------------------

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent non-tool-call text, or None.

        Collects all text outside ``<tool_call>...</tool_call>`` regions,
        including text between consecutive tool calls.  Holds back any
        suffix that could be a partial ``<tool_call>`` tag.
        """
        content_segments: list[str] = []
        pos = self._sent_content_idx

        while pos < len(current_text):
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                tail = current_text[pos:]
                overlap = partial_tag_overlap(tail, self.tool_call_start_token)
                sendable = tail[: len(tail) - overlap] if overlap else tail
                if sendable:
                    content_segments.append(sendable)
                pos = len(current_text) - overlap
                break

            if start > pos:
                content_segments.append(current_text[pos:start])

            end = current_text.find(self.tool_call_end_token, start)
            if end != -1:
                pos = end + len(self.tool_call_end_token)
            else:
                pos = start
                break

        if content_segments:
            self._sent_content_idx = pos
            return "".join(content_segments)
        if pos > self._sent_content_idx:
            self._sent_content_idx = pos
        return None

    def _extract_tool_call_regions(self, text: str) -> list[tuple[str, bool]]:
        """Extract ``(inner_text, is_complete)`` for each ``<tool_call>`` region."""
        results: list[tuple[str, bool]] = []
        pos = 0
        while True:
            start = text.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            inner_start = start + len(self.tool_call_start_token)
            end = text.find(self.tool_call_end_token, inner_start)
            if end != -1:
                results.append((text[inner_start:end], True))
                pos = end + len(self.tool_call_end_token)
            else:
                raw = text[inner_start:]
                overlap = partial_tag_overlap(raw, self.tool_call_end_token)
                if overlap:
                    raw = raw[:-overlap]
                results.append((raw, False))
                break
        return results

    def _extract_tool_name_from_region(self, inner_text: str) -> str | None:
        """Extract the tool name from the beginning of a tool-call region.

        The name is everything before the first ``\\n`` or ``<arg_key>``.
        Returns ``None`` if the name hasn't fully arrived yet.
        """
        nl = inner_text.find("\n")
        ak = inner_text.find(self.arg_key_start)
        candidates = [i for i in [nl, ak] if i != -1]
        if not candidates:
            return None
        cut = min(candidates)
        name = inner_text[:cut].strip()
        return name if name else None

    def _build_args_json_so_far(
        self,
        tool_name: str,
        inner_text: str,
        is_complete: bool,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> str:
        """Build the JSON arguments string from the XML pairs seen so far.

        For complete ``<arg_key>/<arg_value>`` pairs the value is fully
        formatted.  For the last argument whose ``<arg_value>`` has been
        opened but not closed, the partial string content is included
        (JSON-escaped, with an opening ``"`` but no closing ``"``).

        The closing ``}`` is only appended when ``is_complete`` is True
        (i.e. the ``</tool_call>`` tag has arrived).
        """
        pairs = self.func_arg_regex.findall(inner_text)
        logger.debug("pairs: %s", pairs)

        parts: list[str] = []
        for key, value in pairs:
            key = key.strip()
            key_json = json.dumps(key, ensure_ascii=False)
            logger.debug("key: %s, key_json: %s", key, key_json)
            if self._is_string_type(tool_name, key, tools):
                val_json = json.dumps(value, ensure_ascii=False)
                logger.debug("string_type val: %s, val_json: %s", value, val_json)
                parts.append(f"{key_json}: {val_json}")
            else:
                parts.append(f"{key_json}: {value}")
                logger.debug("no string_type val: %s", value)

        # Check for a partial (incomplete) arg value
        last_val_start = inner_text.rfind(self.arg_val_start)
        last_val_end = inner_text.rfind(self.arg_val_end)
        has_partial_value = last_val_start != -1 and (
            last_val_end == -1 or last_val_end < last_val_start
        )
        logger.debug("has_partial_value: %s", has_partial_value)

        if has_partial_value:
            last_key_match = None
            for m in self._arg_key_pattern.finditer(inner_text[:last_val_start]):
                last_key_match = m
            
            logger.debug("last_key_match: %s", last_key_match)
            if last_key_match:
                partial_key = last_key_match.group(1).strip()
                partial_content_start = last_val_start + len(self.arg_val_start)
                partial_content = inner_text[partial_content_start:]
                logger.debug("partial_content: %s", partial_content)

                overlap = partial_tag_overlap(partial_content, self.arg_val_end)
                logger.debug("overlap: %s", overlap)
                if overlap:
                    partial_content = partial_content[:-overlap]
                    logger.debug("partial_content after overlap: %s", partial_content)

                key_json = json.dumps(partial_key, ensure_ascii=False)
                logger.debug("partial_key: %s, key_json: %s", partial_key, key_json)

                if is_complete:
                    if self._is_string_type(tool_name, partial_key, tools):
                        val_json = json.dumps(partial_content, ensure_ascii=False)
                        logger.debug("string_type val: %s, val_json: %s", partial_content, val_json)
                        parts.append(f"{key_json}: {val_json}")
                    else:
                        parts.append(f"{key_json}: {partial_content}")
                        logger.debug("no string_type val: %s", partial_content)
                elif self._is_string_type(tool_name, partial_key, tools):
                    escaped = self._json_escape_string_content(partial_content)
                    logger.debug("escaped: %s", escaped)
                    parts.append(f'{key_json}: "{escaped}')
                else:
                    parts.append(f"{key_json}: {partial_content}")
                    logger.debug("else no string_type val: %s", partial_content)

        if not parts:
            return "{}" if is_complete else ""

        logger.debug("final parts: %s", parts)
        joined = "{" + ", ".join(parts)
        if is_complete:
            joined += "}"
        logger.debug("joined: %s", joined)
        return joined

    def _compute_args_diff(self, index: int, args_so_far: str) -> str | None:
        """Return new argument text not yet sent for tool *index*, or None."""
        prev = self.streamed_args_for_tool[index]
        logger.debug("compare diff prev: %s", prev)
        logger.debug("compare diff args_so_far: %s", args_so_far)
        if not args_so_far or len(args_so_far) <= len(prev):
            return None
        diff = args_so_far[len(prev):]
        logger.debug("compare diff diff: %s", diff)
        self.streamed_args_for_tool[index] = args_so_far
        self.prev_tool_call_arr[index]["arguments"] = args_so_far
        return diff

    def _ensure_tool_state_for(self, index: int) -> None:
        """Grow state arrays so that *index* is valid."""
        while len(self._tool_call_ids) <= index:
            self._tool_call_ids.append(
                make_tool_call_id(id_type="random", func_name=None, idx=None)
            )
        while len(self.streamed_args_for_tool) <= index:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append({})

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        logger.debug("previous_text: %s", previous_text)
        logger.debug("delta_text: %s", delta_text)
        self.temp_previous_text = self.temp_current_text
        self.temp_current_text += delta_text
        previous_text = self.temp_previous_text
        current_text = self.temp_current_text
        logger.debug("real previous_text: %s", previous_text)
        logger.debug("real current_text: %s", current_text)

        # Strip eos tokens from current_text for parsing
        for eos_token in self.eos_tokens:
            if eos_token in current_text:
                current_text = current_text.replace(eos_token, "")

        if not self._tools_enabled(request):
            return DeltaMessage(content=delta_text) if delta_text else None

        # Re-parse-and-diff: parse all tool call regions from full current_text
        content = self._extract_content(current_text)
        regions = self._extract_tool_call_regions(current_text)
        tool_call_deltas: list[DeltaToolCall] = []

        for i, (inner_text, is_complete) in enumerate(regions):
            self._ensure_tool_state_for(i)

            # Extract tool name
            tool_name = self._extract_tool_name_from_region(inner_text)
            if not tool_name:
                break

            # Emit tool name (once per tool call)
            if "name" not in self.prev_tool_call_arr[i]:
                self.prev_tool_call_arr[i]["name"] = tool_name
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=i,
                        id=self._tool_call_ids[i],
                        type="function",
                        function=DeltaFunctionCall(
                            name=tool_name,
                            arguments="",
                        ).model_dump(exclude_none=True),
                    )
                )

            # Build args JSON so far, diff, emit
            logger.debug("tool_name: %s", tool_name)
            logger.debug("inner_text: %s", inner_text)
            logger.debug("is_complete: %s", is_complete)
            args_so_far = self._build_args_json_so_far(
                tool_name, inner_text, is_complete, request.tools
            )
            diff = self._compute_args_diff(i, args_so_far)
            if diff:
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=i,
                        function=DeltaFunctionCall(arguments=diff).model_dump(
                            exclude_none=True
                        ),
                    )
                )

        # Update current_tool_id for serving layer compatibility
        if regions:
            self.current_tool_id = len(regions) - 1

        if content or tool_call_deltas:
            return DeltaMessage(
                content=content,
                tool_calls=tool_call_deltas,
            )
        return None
