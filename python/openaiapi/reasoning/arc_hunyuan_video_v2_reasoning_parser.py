# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================

from collections.abc import Sequence
from functools import cached_property
from typing import Optional, Union

from openaiapi.openai_protocol import DeltaMessage
from openaiapi.reasoning import ReasoningParser, ReasoningParserManager
from openaiapi.transformers_utils.chat_utils import AnyTokenizer
from utilize.logger import get_logger

logger = get_logger(__name__)

ANSWER_START_TAG = "<answer>"
ANSWER_END_TAG = "</answer>"


@ReasoningParserManager.register_module("arc_hunyuan_video_v2")
class ArcHunyuanVideoReasoningParser(ReasoningParser):
    """
    Reasoning parser for ARC-Hunyuan-Video model.

    Output format: <think>...</think><answer>...</answer>
    - reasoning_content: text inside <think>...</think>
    - content: text inside <answer>...</answer> (tags stripped)

    Note: <think>/<answer> are not single tokens; uses string-based detection.
    """

    # Class-level attributes so ksana_service can read them without instantiation
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"

    def __init__(self, tokenizer: Optional[AnyTokenizer] = None):
        super().__init__()
        self.model_tokenizer = tokenizer

        # Streaming state, reset per request.
        self._in_answer = False
        self._answer_buffer = ""

        if self.model_tokenizer is None:
            # Lightweight instance (e.g. for reading _in_reasoning default).
            # Token IDs stay at class defaults; vocab lookup is skipped.
            return

        # Try to locate single-token IDs; set None if unavailable.
        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        if self.think_start_token_id is None or self.think_end_token_id is None:
            logger.warning(
                "ArcHunyuanVideo: <think>/</ think> are not single tokens in "
                "this tokenizer. Falling back to string-based matching."
            )
            self.think_start_token_id = None
            self.think_end_token_id = None

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if self.think_end_token_id is not None:
            return self.think_end_token_id in input_ids
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.think_end_token_id is not None:
            if self.think_end_token_id not in input_ids[:-1]:
                return []
            return input_ids[input_ids.index(self.think_end_token_id) + 1 :]
        return []

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------
    def extract_reasoning_content(
        self, model_output: str
    ) -> tuple[Optional[str], Optional[str]]:
        if (
            self.think_start_token not in model_output
            or self.think_end_token not in model_output
        ):
            return None, self._strip_answer_tags(model_output)

        _, _, after_start = model_output.partition(self.think_start_token)
        if self.think_end_token not in after_start:
            return None, model_output

        reasoning_content, _, after_think = after_start.partition(self.think_end_token)

        content = self._strip_answer_tags(after_think)
        return reasoning_content, content if content else None

    # ------------------------------------------------------------------
    # Streaming (string-based with <answer> tag stripping)
    # ------------------------------------------------------------------
    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        self._buffer += delta_text
        current_text = self._buffer

        # --- Phase 0: <think> start tag ---
        if not self.stripped_think_start:
            if self.think_start_token in current_text:
                current_text = current_text.replace(self.think_start_token, "")
                self.stripped_think_start = True
                self._in_reasoning = True
                if not current_text:
                    self._buffer = ""
                    return None
                # fall through with remaining text
            elif self._is_partial_suffix(current_text, self.think_start_token):
                return None  # keep buffering
            else:
                # no <think> tag, emit as content
                self._buffer = ""
                return DeltaMessage(content=current_text)

        # --- Phase 1: inside <think> reasoning block ---
        if self._in_reasoning:
            if self.think_end_token in current_text:
                end_idx = current_text.find(self.think_end_token)
                reasoning_text = current_text[:end_idx]
                self._buffer = ""
                self._in_reasoning = False
                remaining = current_text[end_idx + len(self.think_end_token) :]
                content = self._process_answer_delta(remaining)
                return DeltaMessage(
                    reasoning_content=reasoning_text if reasoning_text else None,
                    content=content,
                )

            # partial </think> tail, hold back
            safe_text, held_back = self._split_partial_suffix(
                current_text, self.think_end_token
            )
            self._buffer = held_back
            if safe_text:
                return DeltaMessage(reasoning_content=safe_text)
            return None  # all held back

        # --- Phase 2: after </think> ---
        self._buffer = ""
        content = self._process_answer_delta(current_text)
        return DeltaMessage(content=content)

    # ------------------------------------------------------------------
    # Answer tag processing helpers
    # ------------------------------------------------------------------
    def _process_answer_delta(self, delta: str) -> Optional[str]:
        """Strip <answer> tags from streaming delta; buffer partial tags."""
        if not delta:
            return None

        self._answer_buffer += delta

        # --- Handle <answer> start tag ---
        if not self._in_answer:
            start_pos = self._answer_buffer.find(ANSWER_START_TAG)
            if start_pos != -1:
                self._in_answer = True
                self._answer_buffer = self._answer_buffer[
                    start_pos + len(ANSWER_START_TAG) :
                ]
                # fall through to check end tag
            elif self._is_partial_suffix(self._answer_buffer, ANSWER_START_TAG):
                return None
            else:
                text = self._answer_buffer
                self._answer_buffer = ""
                return text

        # --- Inside <answer>, handle </answer> end tag ---
        end_pos = self._answer_buffer.find(ANSWER_END_TAG)
        if end_pos != -1:
            text = self._answer_buffer[:end_pos]
            self._answer_buffer = ""
            self._in_answer = False
            return text if text else None

        # partial </answer> tail, hold back
        safe_text, held_back = self._split_partial_suffix(
            self._answer_buffer, ANSWER_END_TAG
        )
        self._answer_buffer = held_back
        return safe_text if safe_text else None

    @staticmethod
    def _strip_answer_tags(text: str) -> str:
        """Remove <answer>...</answer> wrapper for non-streaming output."""
        text = text.strip()
        start_pos = text.find(ANSWER_START_TAG)
        if start_pos == -1:
            return text
        inner = text[start_pos + len(ANSWER_START_TAG) :]
        end_pos = inner.find(ANSWER_END_TAG)
        if end_pos != -1:
            return inner[:end_pos].strip()
        return inner.strip()
