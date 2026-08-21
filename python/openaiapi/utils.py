# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""Shared helpers for OpenAI-compatible serving modules."""

from typing import Any, Iterable, Optional

# OpenAI content-part types that indicate a multimodal request.
MULTIMODAL_CONTENT_TYPES = frozenset({"image_url", "video_url", "audio_url"})


def get_enable_thinking_from_request(
    request: Any, reasoning_parser: Optional[str]
) -> bool:
    """Extract request-level thinking mode for a configured reasoning parser.

    Shared by Structural Tag builders and chat serving so prompt rendering and
    constrained decoding agree on the same thinking flag.

    Reference:
    https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/entrypoints/openai/serving_chat.py#L1210
    """
    if not reasoning_parser:
        return False
    kwargs = request.chat_template_kwargs or {}
    if reasoning_parser in ("deepseek_v3", "deepseek_v4"):
        # Models that require explicit enable thinking (thinking=True)
        return kwargs.get("thinking") is True
    if reasoning_parser == "kimi_k2":
        # Thinking by default; disable with thinking=False
        return kwargs.get("thinking") is not False
    if reasoning_parser in ("qwen3", "glm45"):
        # Thinking by default; disable with enable_thinking=False
        return kwargs.get("enable_thinking") is not False
    return True


def is_multimodal_messages(messages: Iterable[Any]) -> bool:
    """Return True if any message contains a multimodal content part.

    A message is considered multimodal when its ``content`` is a list that
    contains at least one part whose ``type`` is in
    :data:`MULTIMODAL_CONTENT_TYPES`. Plain string content is never multimodal.

    Iteration is guarded because ``content`` may be a pydantic ValidatorIterator
    (one-shot) after v2 parsing.
    """
    if not messages:
        return False

    for message in messages:
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

        if content is None or isinstance(content, str):
            continue

        try:
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                else:
                    part_type = getattr(part, "type", "")
                if part_type in MULTIMODAL_CONTENT_TYPES:
                    return True
        except Exception:  # pylint: disable=broad-except
            # Iteration failed (e.g. already-consumed iterator); skip.
            continue

    return False
