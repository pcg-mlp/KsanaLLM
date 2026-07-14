# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ref:
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/reasoning/qwen3_reasoning_parser.py

from typing import Optional

from openaiapi.reasoning import ReasoningParser, ReasoningParserManager
from utilize.logger import get_logger

logger = get_logger(__name__)


# Qwen3 and GLM versions 4.5, 4.7, 5, etc. all share the same reasoning parser.
@ReasoningParserManager.register_module(["qwen3", "glm45"])
class Qwen3ReasoningParser(ReasoningParser):
    """
    Reasoning parser for the Qwen3 model.

    The Qwen3 model uses <think>...</think> tokens to denote reasoning text
    within its output. The model provides a strict switch to disable reasoning
    output via the 'enable_thinking=False' parameter. This parser extracts the
    reasoning content enclosed by <think> and </think> tokens from the model's
    output.
    """

    # Token ID映射：将字符串token转换为对应的token ID用于高效处理
    think_start_token_id: Optional[int] = None  # "<think>"对应的token ID
    think_end_token_id: Optional[int] = None    # "</think>"对应的token ID

    # 推理标记的字符串表示
    think_start_token: str = "<think>"   # 推理开始标记
    think_end_token: str = "</think>"    # 推理结束标记

    def __init__(self):
        super().__init__()

        if (self.think_start_token_id is None
                or self.think_end_token_id is None):
            raise RuntimeError(
                "Qwen3 reasoning parser could not locate think start/end "
                "tokens in the tokenizer!")

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.think_end_token_id in input_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens
        """
        if self.think_end_token_id not in input_ids[:-1]:
            return []
        else:
            return input_ids[input_ids.index(self.think_end_token_id) + 1:]
