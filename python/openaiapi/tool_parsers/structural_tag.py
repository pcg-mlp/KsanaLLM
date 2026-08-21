# Copyright 2026 Tencent Inc.  All rights reserved.

"""Model-native strict tool constraints backed by XGrammar v0.2.5.

Maintainer notes live in ``csrc/runtime/structured_generation/README.md``.
"""

from __future__ import annotations

from typing import Any, Optional

import orjson

try:
    from openaiapi._xgrammar import get_model_structural_tag
except ImportError:
    get_model_structural_tag = None

from openaiapi.openai_protocol import ChatCompletionNamedToolChoiceParam
from openaiapi.utils import get_enable_thinking_from_request

from .abstract_tool_parser import ToolParserManager


def _dump_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        return tool_choice.model_dump(mode="json", exclude_none=True)
    return tool_choice


def uses_model_tool_structural_tag(
    request: Any,
    *,
    enable_auto_tool_choice: bool,
    tool_parser_name: Optional[str] = None,
    structural_tag_model: Optional[str] = None,
) -> bool:
    """Return whether this request will attach a model-native tool structural tag.

    Mirrors the eligibility rules of ``build_model_tool_structural_tag`` without
    building the tag payload.
    """
    tools = getattr(request, "tools", None)
    tool_choice = getattr(request, "tool_choice", None)
    if (
        not enable_auto_tool_choice
        or not tools
        or tool_choice == "none"
    ):
        return False
    if tool_choice in (None, "auto") and not any(
        tool.function.strict is True for tool in tools
    ):
        return False
    if get_model_structural_tag is None:
        return False
    model = structural_tag_model
    if model is None:
        if not tool_parser_name:
            return False
        parser_cls = ToolParserManager.get_tool_parser(tool_parser_name)
        model = getattr(parser_cls, "structural_tag_model", None)
    return model is not None


def build_model_tool_structural_tag(
    request: Any,
    *,
    tool_parser_name: Optional[str],
    reasoning_parser_name: Optional[str],
    enable_auto_tool_choice: bool,
) -> Optional[str]:
    """Build a native model structural tag when strict tool calling needs it.

    ``auto`` remains unconstrained unless at least one function explicitly asks
    for strict schema adherence.  ``required`` and named choices always use the
    native model protocol so they do not fall into the legacy generic-JSON
    wrapper.
    """
    if not uses_model_tool_structural_tag(
        request,
        enable_auto_tool_choice=enable_auto_tool_choice,
        tool_parser_name=tool_parser_name,
    ):
        return None
    assert tool_parser_name is not None
    assert get_model_structural_tag is not None
    parser_cls = ToolParserManager.get_tool_parser(tool_parser_name)
    model = getattr(parser_cls, "structural_tag_model", None)
    assert model is not None
    tools = request.tools
    dumped_tools = [
        tool.model_dump(mode="json", exclude_none=True) for tool in tools
    ]
    structural_tag = get_model_structural_tag(
        model=model,
        tools=dumped_tools,
        tool_choice=_dump_tool_choice(getattr(request, "tool_choice", None)),
        reasoning=get_enable_thinking_from_request(
            request, reasoning_parser_name
        ),
    )
    return orjson.dumps(structural_tag.model_dump(mode="json")).decode()
