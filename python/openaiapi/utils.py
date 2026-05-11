# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""Shared helpers for OpenAI-compatible serving modules."""

from typing import Any, Iterable

# OpenAI content-part types that indicate a multimodal request.
MULTIMODAL_CONTENT_TYPES = frozenset({"image_url", "video_url", "audio_url"})


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
