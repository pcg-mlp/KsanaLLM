# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================

from collections.abc import Sequence
from typing import Optional, Union

from openaiapi.openai_protocol import DeltaMessage
from openaiapi.reasoning import ReasoningParser, ReasoningParserManager
from openaiapi.transformers_utils.chat_utils import AnyTokenizer
from utilize.logger import get_logger

logger = get_logger(__name__)

# V1.1 model output format:
#   思考中:\n<reasoning content>\n\n总结如下:\n<answer content>
THINK_START_TAG = "思考中:\n"
THINK_END_TAG = "\n\n总结如下:\n"


@ReasoningParserManager.register_module("arc_hunyuan_video_v1.1")
class ArcHunyuanVideoV11ReasoningParser(ReasoningParser):
    """
    Reasoning parser for ARC-Hunyuan-Video V1.1 model.

    Output format:
        思考中:
        reasoning content...

        总结如下:
        final answer content...

    Markers are plain text strings, not special tokens.
    """

    # Class-level attributes so ksana_service can read them without instantiation
    think_start_token: str = THINK_START_TAG
    think_end_token: str = THINK_END_TAG

    def __init__(self, tokenizer: Optional[AnyTokenizer] = None):
        super().__init__()
        self.model_tokenizer = tokenizer
        # Override base class defaults with V1.1 markers
        self.think_start_token = THINK_START_TAG
        self.think_end_token = THINK_END_TAG

        # Markers are plain text, no token IDs
        self.think_start_token_id = None
        self.think_end_token_id = None

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        # Multi-token markers, not detectable by single token ID
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Multi-token markers, not extractable by token IDs
        return []

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------
    def extract_reasoning_content(
        self, model_output: str
    ) -> tuple[Optional[str], Optional[str]]:
        if THINK_START_TAG not in model_output:
            return None, model_output

        _, _, after_start = model_output.partition(THINK_START_TAG)

        if THINK_END_TAG not in after_start:
            # Truncated before summary marker
            return after_start.strip(), None

        reasoning_content, _, answer_content = after_start.partition(THINK_END_TAG)

        reasoning_content = reasoning_content.strip() if reasoning_content else None
        answer_content = answer_content.strip() if answer_content else None

        return reasoning_content, answer_content

    # ------------------------------------------------------------------
    # Streaming (string-based)
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

        # --- Phase 0: detect start tag ---
        if not self.stripped_think_start:
            if self.think_start_token in current_text:
                current_text = current_text.replace(self.think_start_token, "", 1)
                self.stripped_think_start = True
                self._in_reasoning = True
                if not current_text:
                    self._buffer = ""
                    return None
                # Fall through with remaining text
            elif self._is_partial_suffix(current_text, self.think_start_token):
                return None  # keep buffering
            else:
                # No start tag, emit as content
                self._buffer = ""
                return DeltaMessage(content=current_text)

        # --- Phase 1: inside reasoning block ---
        if self._in_reasoning:
            if self.think_end_token in current_text:
                end_idx = current_text.find(self.think_end_token)
                reasoning_text = current_text[:end_idx]
                self._buffer = ""
                self._in_reasoning = False
                remaining = current_text[end_idx + len(self.think_end_token) :]
                return DeltaMessage(
                    reasoning_content=reasoning_text if reasoning_text else None,
                    content=remaining if remaining else None,
                )

            # Check if tail is a partial end tag
            safe_text, held_back = self._split_partial_suffix(
                current_text, self.think_end_token
            )
            self._buffer = held_back
            if safe_text:
                return DeltaMessage(reasoning_content=safe_text)
            return None  # all held back

        # --- Phase 2: after end tag, content phase ---
        self._buffer = ""
        return DeltaMessage(content=current_text)
