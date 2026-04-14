# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ==============================================================================
# Adapted from vLLM project and SGLang
# [vLLM Project]
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/serving_embedding.py
# [SGLang Project]
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/openai/serving_embedding.py
# ==============================================================================

"""
OpenAI Embeddings API 服务实现

支持功能：
- 文本输入（str / list[str]）
- Token ID 输入（list[int] / list[list[int]]）
- 多模态输入（text + image）
- encoding_format: float / base64
- dimensions 截断
"""

import base64
import time
import uuid
from typing import List, Optional, Tuple, Union
from http import HTTPStatus

import msgpack
import numpy as np
import orjson
from fastapi import Request
from fastapi.responses import Response
from transformers import PreTrainedTokenizerFast

from openaiapi.openai_adapter import KsanaOpenAIServing, ErrorType, OpenAIConfig
from openaiapi.openai_protocol import (
    EmbeddingRequest,
    MultimodalEmbeddingInput,
)
from openaiapi.request_converter import encode_embedding_to_base64

from utilize.logger import get_logger

logger = get_logger(__name__)

# Dtype mapping for embedding extraction.
# bfloat16 is mapped to uint16 because NumPy has no native bfloat16 dtype;
# raw bytes are later reinterpreted via bit-shift to float32.
_EMBEDDING_DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": np.uint16,
    "int32": np.int32,
}

# Supported pooling types for embedding models.
# "cls" and "last" are handled via slice_pos (single token gather).
# "mean" is handled by C++ MeanPooling kernel (averages all tokens).
_SUPPORTED_POOLING_TYPES = {"cls", "last", "mean"}


class KsanaOpenAIServingEmbedding(KsanaOpenAIServing):
    """
    KsanaLLM Embeddings API 实现

    对齐 OpenAI v1/embeddings 接口，支持：
    - 文本与 Token ID 输入
    - 多模态（text + image）输入
    - float / base64 编码格式
    - dimensions 维度截断

    Embedding 全流程（tokenize → forward → extract）完全在此类内完成，
    engine 只需提供 forward() 接口。
    """

    def __init__(self, llm_server, config: Optional[OpenAIConfig] = None):
        super().__init__(llm_server, config)
        self._model_embedding_dim: Optional[int] = None

        # Resolve pooling type and hidden_size from engine_args.
        # These are auto-detected from model files (1_Pooling/config.json)
        # during EngineArgs.from_config_file().
        engine_args = getattr(llm_server, 'engine_args', None)
        pooling_type = getattr(engine_args, 'pooling_type', None) if engine_args else None
        if pooling_type and pooling_type not in _SUPPORTED_POOLING_TYPES:
            logger.warning(
                "Pooling type '%s' is not supported (supported: %s). "
                "Falling back to 'last' token pooling.",
                pooling_type, _SUPPORTED_POOLING_TYPES,
            )
            pooling_type = "last"
        self._pooling_type: str = pooling_type or "last"

        hidden_size = getattr(engine_args, 'hidden_size', None) if engine_args else None
        if hidden_size is not None:
            self._model_embedding_dim = hidden_size
            logger.info("Embedding dim initialized from model config: %d", hidden_size)

        logger.info("Embedding pooling type: %s", self._pooling_type)

    # ------------------------------------------------------------------
    #  OpenAI API entry point
    # ------------------------------------------------------------------

    async def create_embedding(self, request: Request) -> Response:
        """处理 Embeddings API 请求"""
        try:
            request_dict = await request.json()
            embedding_request = EmbeddingRequest(**request_dict)

            # 检查 dimensions 参数
            if embedding_request.dimensions is not None:
                model_dim = self._model_embedding_dim
                if model_dim and embedding_request.dimensions > model_dim:
                    return self.create_error_response(
                        f"Requested dimensions ({embedding_request.dimensions}) exceeds "
                        f"model embedding dimension ({model_dim})",
                        ErrorType.VALIDATION_ERROR,
                        HTTPStatus.BAD_REQUEST,
                    )

            # 转换输入并生成 embeddings
            embeddings, prompt_tokens = await self._generate_embeddings(
                embedding_request, request
            )

            # 处理 dimensions 截断
            if embedding_request.dimensions is not None:
                embeddings = [
                    emb[:embedding_request.dimensions] for emb in embeddings
                ]

            # 构建响应
            request_id = f"embd-{uuid.uuid4().hex}"
            model_name = self._get_model_name(embedding_request.model)

            response = self._build_embedding_response(
                request_id=request_id,
                model_name=model_name,
                embeddings=embeddings,
                prompt_tokens=prompt_tokens,
                encoding_format=embedding_request.encoding_format,
            )

            return Response(
                content=orjson.dumps(
                    response,
                    option=orjson.OPT_SERIALIZE_NUMPY,
                ),
                media_type="application/json",
            )

        except (RuntimeError, TypeError, KeyError, ValueError) as e:
            logger.error("Embeddings API error: %s", e, exc_info=True)
            return self.create_error_response(
                f"Internal server error: {str(e)}",
                ErrorType.INTERNAL_SERVER_ERROR,
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    # ------------------------------------------------------------------
    #  Embedding generation pipeline
    # ------------------------------------------------------------------

    async def _generate_embeddings(
        self,
        request: EmbeddingRequest,
        raw_request: Request,
    ) -> Tuple[List[np.ndarray], int]:
        """
        End-to-end embedding pipeline: parse input → tokenize → forward → extract.

        Returns:
            (embeddings, total_prompt_tokens)
            embeddings are 1-D float32 numpy arrays (not Python lists).
        """
        input_data = request.input
        total_prompt_tokens = 0

        # --- Parse input format ---
        texts: Optional[List[str]] = None
        input_ids_list: Optional[List[List[int]]] = None

        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list):
            # Note: empty list is already rejected by EmbeddingRequest.validate_input,
            # so len(input_data) > 0 is guaranteed here.
            first_item = input_data[0]
            if isinstance(first_item, str):
                texts = input_data
            elif isinstance(first_item, int):
                input_ids_list = [input_data]
            elif isinstance(first_item, list):
                input_ids_list = input_data
            elif isinstance(first_item, (dict, MultimodalEmbeddingInput)):
                texts, _ = self._parse_multimodal_input(input_data)
            else:
                raise ValueError(f"Unsupported input type: {type(first_item)}")
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # --- Build forward requests ---
        truncate = request.truncate_prompt_tokens
        batch_requests, token_counts = self._build_embed_requests(
            texts=texts,
            input_ids_list=input_ids_list,
            truncate_prompt_tokens=truncate,
        )
        if not batch_requests:
            return [], 0

        total_prompt_tokens = sum(token_counts)

        # --- Call engine forward ---
        start = time.perf_counter()
        req_ctx = {"Content-Type": "application/x-msgpack"}
        request_bytes = msgpack.packb({"requests": batch_requests}, use_bin_type=True)
        status, response_bytes = await self.llm_server.model.forward(request_bytes, req_ctx)
        if not status.OK() or response_bytes is None:
            msg = status.GetMessage() if hasattr(status, "GetMessage") else str(status)
            raise RuntimeError(f"Embedding forward failed: {msg}")
        forward_ms = (time.perf_counter() - start) * 1000

        # --- Extract embeddings from response ---
        forward_result = msgpack.unpackb(response_bytes, raw=False)
        responses = forward_result.get("responses") or []

        if len(responses) != len(batch_requests):
            raise RuntimeError(
                f"Response count mismatch: expected {len(batch_requests)}, "
                f"got {len(responses)}"
            )

        embeddings: List[np.ndarray] = []
        for idx, resp in enumerate(responses):
            embedding_np = self._extract_single_embedding(resp)
            if embedding_np is None:
                logger.error("Failed to extract embedding for input %d", idx)
                raise RuntimeError(
                    f"Failed to extract embedding for input {idx}"
                )
            embeddings.append(embedding_np)

        # Update cached embedding dimension
        if embeddings and self._model_embedding_dim is None:
            self._model_embedding_dim = embeddings[0].shape[0]

        logger.debug(
            "EMBED| count=%d forward=%.2fms total_tokens=%d",
            len(embeddings), forward_ms, total_prompt_tokens,
        )

        return embeddings, total_prompt_tokens

    # ------------------------------------------------------------------
    #  Tokenization & request building
    # ------------------------------------------------------------------

    def _build_embed_requests(
        self,
        texts: Optional[List[str]] = None,
        input_ids_list: Optional[List[List[int]]] = None,
        truncate_prompt_tokens: Optional[int] = None,
    ) -> Tuple[List[dict], List[int]]:
        """Tokenize inputs and build forward request dicts.

        Args:
            texts: Raw text strings to tokenize.
            input_ids_list: Pre-tokenized token ID lists.
            truncate_prompt_tokens: Optional max token count for truncation.

        Returns:
            (batch_requests, token_counts) — request dicts for msgpack and
            per-input token counts for usage reporting.

        Raises:
            ValueError: If any input produces an empty token list.
        """
        batch_requests: List[dict] = []
        token_counts: List[int] = []

        if texts:
            inputs = texts
            is_text = True
        elif input_ids_list:
            inputs = input_ids_list
            is_text = False
        else:
            return [], []

        # Tokenize
        if is_text:
            add_special = getattr(
                self.llm_server.pre_post_processor, "add_special_tokens", True
            )
            tokenizer = self.tokenizer
            if isinstance(tokenizer, PreTrainedTokenizerFast):
                batch_encoded = tokenizer(
                    inputs,
                    add_special_tokens=add_special,
                    padding=False,
                    truncation=False,
                )
                all_token_ids = batch_encoded["input_ids"]
            else:
                all_token_ids = [
                    self.llm_server.pre_post_processor.encode(item)
                    for item in inputs
                ]
        else:
            all_token_ids = [list(item) for item in inputs]

        for idx, tokens in enumerate(all_token_ids):
            if truncate_prompt_tokens and len(tokens) > truncate_prompt_tokens:
                tokens = tokens[:truncate_prompt_tokens]

            if not tokens:
                raise ValueError(
                    f"Empty tokens for embedding request at index {idx}"
                )

            token_counts.append(len(tokens))
            seq_len = len(tokens)

            # Generate slice_pos based on pooling strategy.
            # slice_pos maps to logits_idx in C++ (used by AssembleTokensHidden).
            #   "cls"  → first token: [[0, 0]]
            #   "last" → last token:  [[seq_len-1, seq_len-1]]
            #   "mean" → slice_pos is NOT used by the C++ MeanPooling kernel
            #            (it uses input_offset_uint64_tensor to average all tokens),
            #            but a valid value is still required for request construction.
            #            We use last-token position as a harmless placeholder.
            if self._pooling_type == "cls":
                slice_pos = [[0, 0]]
            else:  # "last" or "mean"
                slice_pos = [[seq_len - 1, seq_len - 1]]

            batch_requests.append({
                "input_tokens": tokens,
                "request_target": [{
                    "target_name": "layernorm",
                    "slice_pos": slice_pos,
                    "token_reduce_mode": "GATHER_ALL",
                }],
            })

        return batch_requests, token_counts

    # ------------------------------------------------------------------
    #  Embedding extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_single_embedding(resp: dict) -> Optional[np.ndarray]:
        """Extract a single embedding from one forward response.

        Decodes the base64 tensor payload and converts bfloat16 data to
        float32 via bit-level reinterpretation.

        Args:
            resp: A single response dict from the forward result.

        Returns:
            A 1-D float32 numpy array, or ``None`` on failure.
        """
        response_list = resp.get("response") or []
        if not response_list:
            logger.debug("_extract: 'response' key is empty or missing")
            return None

        first_resp = response_list[0]
        tensor_info = first_resp.get("tensor")
        if tensor_info is None:
            keys = list(first_resp.keys()) if isinstance(first_resp, dict) else type(first_resp)
            logger.error("_extract: 'tensor' key missing in first response item. keys=%s", keys)
            return None

        data = tensor_info.get("data")
        shape = tensor_info.get("shape")
        dtype = tensor_info.get("dtype")
        if data is None or shape is None or dtype is None:
            logger.error("_extract: tensor_info incomplete. data=%s, shape=%s, dtype=%s",
                        type(data).__name__ if data is not None else "None",
                        shape, dtype)
            return None

        np_dtype = _EMBEDDING_DTYPE_MAP.get(dtype)
        if np_dtype is None:
            logger.warning("Unsupported embedding dtype: %s", dtype)
            return None

        arr = np.frombuffer(base64.b64decode(data), dtype=np_dtype)
        arr = arr.reshape(shape)

        # bfloat16 → float32 conversion via bit-level reinterpretation.
        # bfloat16 is the upper 16 bits of float32, so left-shifting by 16
        # and reinterpreting as float32 perfectly restores the original value.
        if dtype == "bfloat16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)

        # Flatten from [1, hidden_dim] to [hidden_dim].
        if arr.ndim > 1:
            arr = arr[0]

        return arr

    # ------------------------------------------------------------------
    #  Multimodal input parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_multimodal_input(
        input_data: List[Union[dict, MultimodalEmbeddingInput]],
    ) -> Tuple[List[str], List[Optional[str]]]:
        """解析多模态输入 → (texts, image_urls)"""
        texts = []
        image_urls = []

        for item in input_data:
            if isinstance(item, dict):
                text = item.get("text", "")
                image = item.get("image")
            elif isinstance(item, MultimodalEmbeddingInput):
                text = item.text or ""
                image = item.image
            else:
                raise ValueError(f"Invalid multimodal input type: {type(item)}")
            texts.append(text)
            image_urls.append(image)

        return texts, image_urls

    # ------------------------------------------------------------------
    #  Response building
    # ------------------------------------------------------------------

    def _build_embedding_response(
        self,
        request_id: str,
        model_name: str,
        embeddings: List[np.ndarray],
        prompt_tokens: int,
        encoding_format: str = "float",
    ) -> dict:
        """构建 Embedding 响应（返回 plain dict 供 orjson 直接序列化）。

        When encoding_format="float", embedding numpy arrays are stored as-is
        in the dict and serialized by orjson with OPT_SERIALIZE_NUMPY, avoiding
        the overhead of .tolist() and Pydantic model_dump() conversion.
        """
        data = []
        for i, embedding in enumerate(embeddings):
            if encoding_format == "base64":
                embedding_data = encode_embedding_to_base64(embedding)
            else:
                # Keep as numpy array — orjson serializes it natively.
                embedding_data = embedding

            data.append({
                "index": i,
                "object": "embedding",
                "embedding": embedding_data,
            })

        return {
            "id": request_id,
            "object": "list",
            "model": model_name,
            "data": data,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens,
            },
        }
