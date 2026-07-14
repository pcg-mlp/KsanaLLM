# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""
测试 KsanaEngineError 异常类及 ksana_service.py 中的错误处理路径。

不需要启动真实模型服务，通过 mock 和 FastAPI TestClient 验证：
1. KsanaEngineError 的基本属性和继承关系
2. ksana_service.py 中 KsanaEngineError vs 普通异常的分支行为
3. 流式场景下的错误响应格式

运行方式 (从 python/ 目录)：
    python3 -m pytest tests/test_ksana_engine_error.py -v
"""

import os
import sys
import types
import logging

import orjson
import pytest

# ---------------------------------------------------------------------------
# exceptions.py 没有任何外部依赖，构造一个轻量 ksana_llm 包来避免
# __init__.py 拉入 torch / libtorch_serving 等重型模块。
# ---------------------------------------------------------------------------
_ksana_pkg = types.ModuleType("ksana_llm")
_ksana_pkg.__path__ = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ksana_llm"))
]
sys.modules.setdefault("ksana_llm", _ksana_pkg)

from ksana_llm.exceptions import KsanaEngineError  # noqa: E402

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

OPENAI_ERROR_KEYS = {"message", "type", "code", "param"}


# ===========================================================================
# 单元测试: KsanaEngineError
# ===========================================================================
class TestKsanaEngineError:
    """测试自定义异常类的基本行为。"""

    def test_inherits_runtime_error(self):
        err = KsanaEngineError("test", 400)
        assert isinstance(err, RuntimeError)

    def test_message_and_code(self):
        err = KsanaEngineError("input too long", 413)
        assert str(err) == "input too long"
        assert err.error_code == 413

    def test_default_error_code(self):
        err = KsanaEngineError("something failed")
        assert err.error_code == 500

    def test_caught_by_except_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise KsanaEngineError("boom", 503)

    def test_caught_by_except_ksana_engine_error(self):
        with pytest.raises(KsanaEngineError) as exc_info:
            raise KsanaEngineError("engine down", 503)
        assert exc_info.value.error_code == 503

    def test_not_caught_by_except_value_error(self):
        """KsanaEngineError should NOT be caught by except ValueError."""
        with pytest.raises(KsanaEngineError):
            try:
                raise KsanaEngineError("test", 400)
            except ValueError:
                pytest.fail("Should not be caught by ValueError")

    def test_isinstance_branch(self):
        """模拟 ksana_service.py 的 isinstance 分支逻辑。"""
        def _extract_error_info(e: Exception):
            if isinstance(e, KsanaEngineError):
                return str(e), e.error_code
            return "Internal server error", 500

        msg, code = _extract_error_info(KsanaEngineError("engine: bad input", 400))
        assert msg == "engine: bad input"
        assert code == 400

        msg, code = _extract_error_info(ValueError("secret traceback info"))
        assert msg == "Internal server error"
        assert code == 500

        msg, code = _extract_error_info(RuntimeError("CUDA OOM"))
        assert msg == "Internal server error"
        assert code == 500


# ===========================================================================
# 集成测试: ksana_service.py 错误处理路径
# ===========================================================================
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _build_error_response(e: Exception) -> JSONResponse:
    """复制 ksana_service.py 中的错误处理逻辑。"""
    if isinstance(e, KsanaEngineError):
        error_code = e.error_code
        error_message = str(e)
    else:
        error_code = 500
        error_message = "Internal server error"
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": error_message,
                "type": "internal_server_error",
                "code": error_code,
                "param": None,
            }
        },
    )


async def _build_streaming_error(e: Exception):
    """复制 ksana_service.py 中的流式错误处理逻辑。"""
    if isinstance(e, KsanaEngineError):
        error_code = e.error_code
        error_message = str(e)
    else:
        error_code = 500
        error_message = "Internal server error"
    error_ret = {
        "error": {
            "message": error_message,
            "type": "internal_server_error",
            "code": error_code,
            "param": None,
        }
    }
    yield orjson.dumps(error_ret) + b"\0"


def _create_error_test_app() -> FastAPI:
    app = FastAPI()

    @app.get("/engine_error")
    async def _engine_error():
        try:
            raise KsanaEngineError("Input length exceeded maximum", 413)
        except Exception as e:  # pylint: disable=broad-except
            return _build_error_response(e)

    @app.get("/generic_error")
    async def _generic_error():
        try:
            raise RuntimeError("CUDA out of memory at 0x7fff")
        except Exception as e:  # pylint: disable=broad-except
            return _build_error_response(e)

    @app.get("/value_error")
    async def _value_error():
        try:
            raise ValueError("invalid literal for int()")
        except Exception as e:  # pylint: disable=broad-except
            return _build_error_response(e)

    @app.get("/stream_engine_error")
    async def _stream_engine_error():
        async def gen():
            try:
                yield orjson.dumps({"texts": ["hello"]}) + b"\0"
                raise KsanaEngineError("Request terminated", 499)
            except Exception as e:  # pylint: disable=broad-except
                async for chunk in _build_streaming_error(e):
                    yield chunk
        return StreamingResponse(gen())

    @app.get("/stream_generic_error")
    async def _stream_generic_error():
        async def gen():
            try:
                yield orjson.dumps({"texts": ["hello"]}) + b"\0"
                raise RuntimeError("segfault in attention kernel")
            except Exception as e:  # pylint: disable=broad-except
                async for chunk in _build_streaming_error(e):
                    yield chunk
        return StreamingResponse(gen())

    return app


@pytest.fixture(scope="module")
def error_client():
    app = _create_error_test_app()
    return TestClient(app, raise_server_exceptions=False)


class TestServiceErrorHandling:
    """验证 ksana_service.py 的错误处理路径。"""

    def test_engine_error_exposes_message(self, error_client):
        """KsanaEngineError → 返回引擎提供的错误消息和错误码。"""
        resp = error_client.get("/engine_error")
        body = resp.json()
        assert body["error"]["message"] == "Input length exceeded maximum"
        assert body["error"]["code"] == 413
        assert body["error"]["type"] == "internal_server_error"
        assert body["error"]["param"] is None

    def test_generic_runtime_error_hides_message(self, error_client):
        """普通 RuntimeError → 返回通用消息，不泄漏内部细节。"""
        resp = error_client.get("/generic_error")
        body = resp.json()
        assert body["error"]["message"] == "Internal server error"
        assert body["error"]["code"] == 500
        assert "CUDA" not in body["error"]["message"]

    def test_value_error_hides_message(self, error_client):
        """ValueError → 返回通用消息。"""
        resp = error_client.get("/value_error")
        body = resp.json()
        assert body["error"]["message"] == "Internal server error"
        assert body["error"]["code"] == 500
        assert "invalid literal" not in body["error"]["message"]

    def test_all_error_responses_have_openai_keys(self, error_client):
        """所有错误响应都包含完整的 OpenAI 错误格式 key。"""
        for route in ["/engine_error", "/generic_error", "/value_error"]:
            resp = error_client.get(route)
            body = resp.json()
            assert "error" in body
            assert OPENAI_ERROR_KEYS <= set(body["error"].keys()), (
                f"{route}: missing keys in error response"
            )


class TestStreamingErrorHandling:
    """验证流式场景下的错误响应。"""

    def _parse_streaming_chunks(self, response):
        """解析 \0 分隔的流式响应。"""
        raw = response.content
        chunks = []
        for part in raw.split(b"\0"):
            if part:
                chunks.append(orjson.loads(part))
        return chunks

    def test_stream_engine_error_exposes_message(self, error_client):
        """流式 KsanaEngineError → 错误帧包含引擎错误消息。"""
        resp = error_client.get("/stream_engine_error")
        chunks = self._parse_streaming_chunks(resp)
        assert len(chunks) == 2
        assert "texts" in chunks[0]
        error_chunk = chunks[1]
        assert error_chunk["error"]["message"] == "Request terminated"
        assert error_chunk["error"]["code"] == 499

    def test_stream_generic_error_hides_message(self, error_client):
        """流式普通异常 → 错误帧返回通用消息，不泄漏内部细节。"""
        resp = error_client.get("/stream_generic_error")
        chunks = self._parse_streaming_chunks(resp)
        assert len(chunks) == 2
        assert "texts" in chunks[0]
        error_chunk = chunks[1]
        assert error_chunk["error"]["message"] == "Internal server error"
        assert error_chunk["error"]["code"] == 500
        assert "segfault" not in error_chunk["error"]["message"]

    def test_stream_error_has_openai_format(self, error_client):
        """流式错误帧也符合 OpenAI 错误格式。"""
        for route in ["/stream_engine_error", "/stream_generic_error"]:
            resp = error_client.get(route)
            chunks = self._parse_streaming_chunks(resp)
            error_chunk = chunks[-1]
            assert "error" in error_chunk
            assert OPENAI_ERROR_KEYS <= set(error_chunk["error"].keys()), (
                f"{route}: missing keys in streaming error"
            )
